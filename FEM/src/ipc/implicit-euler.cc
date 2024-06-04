//
// Created by creeper on 5/29/24.
//
#include <fem/ipc/implicit-euler.h>

namespace fem {
void IpcImplicitEuler::step(System &system, Real dt) {
  x_t_buf = system.x;
  const VecXd& x_t = x_t_buf;
  auto &x = system.x;
  auto &p = p_buf;
  computeConstraintSet(x, config.dHat);
  Real h = dt;
  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(system, x, x_t, active_constraints, h);
  x_prev = x;
  do {
    auto H = spdProjectHessian(system);
    g = barrierAugmentedIncrementalPotentialEnergyGradient(system, x, x_t, active_constraints, h);
    ldlt.compute(H);
    if (ldlt.info() != Eigen::Success)
      core::ERROR("Failed to perform LDLT decomposition");
    p = ldlt.solve(-g);
    if (ldlt.info() != Eigen::Success)
      core::ERROR("Failed to solve triangular systems");
    Real alpha = std::min(1.0, computeStepSizeUpperBound(x, p, active_constraints));
    Real E;
    do {
      x = x_prev + alpha * p;
      computeConstraintSet(x, config.dHat);
      alpha = alpha * 0.5;
      E = barrierAugmentedIncrementalPotentialEnergy(system, x, x_t, active_constraints, h);
    } while (E > E_prev);
    E_prev = E;
    x_prev = x;
  } while (p.lpNorm<Eigen::Infinity>() / h > config.eps);
}

}