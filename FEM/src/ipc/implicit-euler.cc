//
// Created by creeper on 5/29/24.
//
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/implicit-euler.h>
#include <fem/system.h>

namespace sim::fem {

Real IpcImplicitEuler::incrementalPotentialKinematicEnergy(const VecXd &x_t, Real h) const {
  const auto &v_t = system().xdot;
  auto x_hat = x_t + h * v_t + h * h * system().computeAcceleration();
  return 0.5 * (system().currentConfig() - x_hat).transpose() * system().mass() * (system().currentConfig() - x_hat);
}

VecXd IpcImplicitEuler::incrementalPotentialKinematicEnergyGradient(const VecXd &x_t, Real h) const {
  const auto &v_t = system().xdot;
  auto x_hat = x_t + h * v_t + h * h * system().computeAcceleration();
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