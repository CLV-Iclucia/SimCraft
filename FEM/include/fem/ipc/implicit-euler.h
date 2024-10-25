//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
#include <fem/integrator.h>
#include <fem/ipc/integrator.h>
#include <Core/debug.h>
#include <Maths/sparse-matrix-builder.h>
namespace fem {

struct IpcImplicitEuler final : public IpcIntegrator {
  explicit IpcImplicitEuler(System &system, const Config &config = {});
 private:

  Real incrementalPotentialKinematicEnergy(const VecXd &x_t, Real h) const override;

  void velocityUpdate(const VecXd& x_t, Real h) const override {
    system().xdot = (system().currentConfig() - x_t) / h;
  }

  VecXd incrementalPotentialKinematicEnergyGradient(const VecXd &x_t, Real h) const override;

  friend VecXd symbolicIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const VecXd &x_t, Real h);
  friend VecXd numericalIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const VecXd &x_t, Real h);
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
