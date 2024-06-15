//
// Created by creeper on 5/29/24.
//
#include <fem/ipc/implicit-euler.h>

namespace fem {
void IpcImplicitEuler::step(Real dt) {
  auto& x_t = x_t_buf;
  x_prev = system().currentConfig();
  x_t = system().currentConfig();
  computeConstraintSet(system().currentConfig(), config.dHat);
  Real h = dt;
  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, active_constraints, h);
  VecXd p(x_t.size());
  while(true) {
    auto H = spdProjectHessian();
    g = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, active_constraints, h);;
    if (ldlt.compute(H).info() != Eigen::Success)
      core::ERROR("Failed to perform LDLT decomposition");
    p = ldlt.solve(-g);
    if (ldlt.info() != Eigen::Success)
      core::ERROR("Failed to solve triangular systems");
    if (p.lpNorm<Eigen::Infinity>() > config.eps)
      break;
    Real alpha = std::min(1.0, computeStepSizeUpperBound(p, active_constraints));
    Real E;
    do {
      updateCandidateSolution(x_prev + alpha * p);
      computeConstraintSet(system().currentConfig(), config.dHat);
      alpha = alpha * 0.5;
      E = barrierAugmentedIncrementalPotentialEnergy(x_t, active_constraints, h);
    } while (E > E_prev);
    x_prev = system().currentConfig();
    E_prev = E;
  }
}
SparseMatrix<Real> IpcImplicitEuler::spdProjectHessian() {
  hessian_builder.reset();
  system().spdProjectHessian(hessian_builder);
  return hessian_builder.build();
}
Real IpcImplicitEuler::incrementalPotentialEnergy(const VecXd &x_t, Real h) const {
  const auto &v_t = system().xdot;
  const auto &f_e = system().f_ext;
  auto x_hat = x_t + h * v_t + h * h * M.inverse() * f_e;
  return 0.5 * (system().currentConfig() - x_hat).transpose() * M * (system().currentConfig() - x_hat)
      + h * h * system().deformationEnergy();
}
Real IpcImplicitEuler::barrierAugmentedIncrementalPotentialEnergy(const VecXd &x_t,
                                                                  const tbb::concurrent_vector<ipc::Constraint> &active_constraints,
                                                                  Real h) {
  Real barrier_energy = 0.0;
  for (const auto &c : active_constraints) {

  }
  return incrementalPotentialEnergy(x_t, h) + kappa * barrier_energy;
}

}