//
// Created by creeper on 5/29/24.
//
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/implicit-euler.h>
#include <fem/ipc/distances.h>
#include <fem/system.h>
namespace fem {
constexpr Real kStepSizeScale = 0.9;

IpcImplicitEuler::IpcImplicitEuler(System &system, const Config &config) : IpcIntegrator(system, config) {
  collisionDetector = std::make_unique<ipc::CollisionDetector>();
  edgesBVH = std::make_unique<spatify::LBVH<Real>>();
  trianglesBVH = std::make_unique<spatify::LBVH<Real>>();
}

void IpcImplicitEuler::step(Real dt) {
  x_prev = system().currentConfig();
  VecXd x_t = system().currentConfig();
  Real h = dt;
  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
  VecXd p(system().dof());
  VecXd g(system().dof());
  int iter = 0;
  while (true) {
    g = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
    auto H = spdProjectHessian(h);
//    if (ldlt.compute(H).info() != Eigen::Success)
//      core::ERROR("Failed to perform LDLT decomposition");
//    p = ldlt.solve(-g);
    p = Eigen::ConjugateGradient<SparseMatrix<Real>>(H).solve(-g);
    if (ldlt.info() != Eigen::Success)
      throw std::runtime_error("Failed to solve triangular systems");
    if (p.lpNorm<Eigen::Infinity>() < config.eps * system().meshLengthScale() * h)
      break;
    Real alpha = kStepSizeScale * std::min(1.0, computeStepSizeUpperBound(p));
    if (alpha == 0.0)
      throw std::runtime_error("Invalid state: collision happened within an integration step");
    precomputeConstraintSet(
        {.descentDir = p,
            .toi = alpha,
            .dHat = config.dHat});
    Real E;
    do {
      updateCandidateSolution(x_prev + alpha * p);
      alpha = alpha * 0.5;
      E = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    } while (E > E_prev);
    iter++;
    x_prev = system().currentConfig();
    E_prev = E;
  }
  velocityUpdate(x_t, h);
}

Real IpcImplicitEuler::incrementalPotentialKinematicEnergy(const VecXd &x_t, Real h) const {
  const auto &v_t = system().xdot;
  const auto &f_e = system().f_ext;
  auto x_hat = x_t + h * v_t + h * h * system().massLDLT().solve(f_e);
  return 0.5 * (system().currentConfig() - x_hat).transpose() * system().mass() * (system().currentConfig() - x_hat);
}

VecXd IpcImplicitEuler::incrementalPotentialKinematicEnergyGradient(const VecXd &x_t, Real h) const {
  const auto &v_t = system().xdot;
  const auto &f_e = system().f_ext;
  auto x_hat = x_t + h * v_t + h * h * system().massLDLT().solve(f_e);
  return system().mass() * (system().currentConfig() - x_hat);
}

VecXd symbolicIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const VecXd &x_t, Real h) {
  return euler.barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
}

VecXd numericalIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const VecXd &x_t, Real h) {
  VecXd grad = VecXd::Zero(x_t.size());
  Real dx = 1e-5;
  VecXd current = euler.system().currentConfig();
  for (int i = 0; i < x_t.size(); i++) {
    VecXd x_t_plus = current;
    x_t_plus(i) += dx;
    euler.updateCandidateSolution(x_t_plus);
    Real E_plus = euler.barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    VecXd x_t_minus = current;
    x_t_minus(i) -= dx;
    euler.updateCandidateSolution(x_t_minus);
    Real E_minus = euler.barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    grad(i) = (E_plus - E_minus) / (2 * dx);
  }
  return grad;
}

}