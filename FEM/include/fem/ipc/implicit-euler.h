//
// Created by creeper on 5/28/24.
//

#pragma once
#include <fem/integrator.h>
#include <fem/ipc/integrator.h>

namespace sim::fem {

struct IpcImplicitEuler final : public IpcIntegrator {
  explicit IpcImplicitEuler(System &system, const Config &config = {})
    : IpcIntegrator(system, config) {
  }

private:
  [[nodiscard]] Real incrementalPotentialKinematicEnergy(const VecXd &x_t, Real h) const override;

  void velocityUpdate(const VecXd& x_t, Real h) const override {
    system().xdot = (system().currentConfig() - x_t) / h;
  }

  [[nodiscard]] VecXd incrementalPotentialKinematicEnergyGradient(const VecXd &x_t, Real h) const override;

  friend VecXd symbolicIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const VecXd &x_t, Real h);
  friend VecXd numericalIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const VecXd &x_t, Real h);
};

}