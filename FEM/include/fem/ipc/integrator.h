//
// Created by creeper on 10/22/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_INTEGRATOR_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_INTEGRATOR_H_
#include <fem/integrator.h>
namespace fem {

struct ConstraintSet {
  void clear() {
    vtConstraints.clear();
    eeConstraints.clear();
  }
  std::vector<ipc::VertexTriangleConstraint> vtConstraints{};
  std::vector<ipc::EdgeEdgeConstraint> eeConstraints{};
};

struct ConstraintSetPrecomputeRequest {
  const VecXd &descentDir;
  Real toi;
  Real dHat;
};

class IpcIntegrator : public Integrator {
 public:
  struct Config {
    Real dHat = 1e-3;
    Real eps = 1e-2;
    Real contactStiffness = 1e10;
  } config;

  explicit IpcIntegrator(System &system, const Config &config)
      : Integrator(system), config(config), barrier(config.dHat) {}

 protected:

  Real barrierEnergy() {
    Real barrierEnergy = 0.0;
    Real kappa = config.contactStiffness;
    for (const auto &c : constraintSet.vtConstraints)
      barrierEnergy += barrier(c.distanceSqr());
    for (const auto &c : constraintSet.eeConstraints)
      barrierEnergy += c.mollifier() * barrier(c.distanceSqr());
    return kappa * barrierEnergy;
  }

  VecXd barrierEnergyGradient() {
    VecXd gradient = VecXd::Zero(system().dof());
    gradient.setZero();
    Real kappa = config.contactStiffness;
    VecXd current = system().currentConfig();
    for (const auto &c : constraintSet.vtConstraints)
      c.assembleBarrierGradient(barrier, gradient, kappa);
    for (const auto &c : constraintSet.eeConstraints)
      c.assembleMollifiedBarrierGradient(barrier, gradient, kappa);
    return gradient;
  }

  ConstraintSet constraintSet;
  std::unique_ptr<ipc::CollisionDetector> collisionDetector{};
  ipc::LogBarrier barrier;
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_INTEGRATOR_H_
