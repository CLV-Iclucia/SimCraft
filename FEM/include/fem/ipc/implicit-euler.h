//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
#include <fem/integrator.h>
#include <fem/ipc/constraint.h>
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/barrier-functions.h>
#include <tbb/concurrent_vector.h>
#include <Core/log.h>
#include <Core/debug.h>
#include <Maths/sparse-matrix-builder.h>
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
  ConstraintSet constraintSet;
  std::unique_ptr<ipc::CollisionDetector> collisionDetector{};
  ipc::LogBarrier barrier;
};

struct IpcImplicitEuler : public IpcIntegrator {
  explicit IpcImplicitEuler(System &system, const Config &config = {});
  void step(Real dt) override;
 private:
  SparseMatrix<Real> spdProjectHessian(Real h);

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

  Real incrementalPotentialKinematicEnergy(const VecXd &x_t, Real h) const;

  // for now, do not consider dissipative friction forces
  Real barrierAugmentedIncrementalPotentialEnergy(const VecXd &x_t, Real h);

  VecXd incrementalPotentialKinematicEnergyGradient(const VecXd &x_t, Real h);

  VecXd barrierAugmentedIncrementalPotentialEnergyGradient(const VecXd &x_t, Real h);

  void updateCandidateSolution(const VecXd &x) {
    system().updateCurrentConfig(x);
    updateConstraintStatus();
  }

  Real barrierAugmentedPotentialEnergy() {
    return system().potentialEnergy() + barrierEnergy();
  }

  VecXd barrierAugmentedPotentialEnergyGradient() {
    return system().deformationEnergyGradient() + barrierEnergyGradient();
  }

  void updateConstraintStatus();

  Real computeStepSizeUpperBound(const VecXd &p) const;

  void precomputeConstraintSet(const ConstraintSetPrecomputeRequest &config);

  void computeVertexTriangleConstraints(const ConstraintSetPrecomputeRequest &config);

  void computeEdgeEdgeConstraints(const ConstraintSetPrecomputeRequest &config);

  int activeConstraints() {
    int active = 0;
    for (const auto &c : constraintSet.vtConstraints)
      active += (c.distanceSqr() < barrier.dHatSqr());
    for (const auto &c : constraintSet.eeConstraints)
      active += (c.distanceSqr() < barrier.dHatSqr());
    return active;
  }

  maths::SparseMatrixBuilder<Real> sparseBuilder{};
  Eigen::SimplicialLDLT<SparseMatrix<Real>> ldlt{};
  std::unique_ptr<spatify::LBVH<Real>> edgesBVH{};
  std::unique_ptr<spatify::LBVH<Real>> trianglesBVH{};
  VecXd x_prev;

  friend VecXd symbolicIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const VecXd &x_t, Real h);
  friend VecXd numericalIncrementalPotentialEnergyGradient(IpcImplicitEuler &euler, const VecXd &x_t, Real h);
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
