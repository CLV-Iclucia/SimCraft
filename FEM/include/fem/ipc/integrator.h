//
// Created by creeper on 10/22/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_INTEGRATOR_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_INTEGRATOR_H_
#include <Maths/sparse-matrix-builder.h>
#include <Core/debug.h>
#include <tbb/concurrent_vector.h>
#include <fem/ipc/integrator.h>
#include <fem/integrator.h>
#include <fem/ipc/constraint.h>
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/barrier-functions.h>

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
    Real stepSizeScale = 0.9;
  } config;

  explicit IpcIntegrator(System &system, const Config &config)
      : Integrator(system), config(config), barrier(config.dHat) {}

  void step(Real dt) override;
  Eigen::SimplicialLDLT<SparseMatrix<Real>> ldlt{};
 protected:

  [[nodiscard]] Real barrierEnergy() const;

  [[nodiscard]] VecXd barrierEnergyGradient() const;

  virtual void velocityUpdate(const VecXd &x_t, Real h) const = 0;
  [[nodiscard]] virtual Real incrementalPotentialKinematicEnergy(const VecXd &x_t, Real h) const = 0;
  [[nodiscard]] virtual VecXd incrementalPotentialKinematicEnergyGradient(const VecXd &x_t, Real h) const = 0;

  [[nodiscard]] Real barrierAugmentedPotentialEnergy() const {
    return system().potentialEnergy() + barrierEnergy();
  }

  [[nodiscard]] VecXd barrierAugmentedPotentialEnergyGradient() const {
    return system().deformationEnergyGradient() + barrierEnergyGradient();
  }

  [[nodiscard]] Real barrierAugmentedIncrementalPotentialEnergy(const VecXd &x_t, Real h) const {
    return incrementalPotentialKinematicEnergy(x_t, h) + h * h * barrierAugmentedPotentialEnergy();
  }

  [[nodiscard]] VecXd barrierAugmentedIncrementalPotentialEnergyGradient(const VecXd &x_t, Real h) const {
    return incrementalPotentialKinematicEnergyGradient(x_t, h)
        + h * h * barrierAugmentedPotentialEnergyGradient();
  }

  void updateCandidateSolution(const VecXd &x) {
    system().updateCurrentConfig(x);
    updateConstraintStatus();
  }

  [[nodiscard]] Real computeStepSizeUpperBound(const VecXd &p) const {
    auto t = collisionDetector->detect(system(), p);
    return t.value_or(1.0);
  }

  SparseMatrix<Real> spdProjectHessian(Real h);
  void updateConstraintStatus();
  void precomputeConstraintSet(const ConstraintSetPrecomputeRequest &config);
  void computeVertexTriangleConstraints(const ConstraintSetPrecomputeRequest &config);
  void computeEdgeEdgeConstraints(const ConstraintSetPrecomputeRequest &config);

  ConstraintSet constraintSet;
  std::unique_ptr<ipc::CollisionDetector> collisionDetector{};
  ipc::LogBarrier barrier;
  std::unique_ptr<spatify::LBVH<Real>> edgesBVH{};
  std::unique_ptr<spatify::LBVH<Real>> trianglesBVH{};
  maths::SparseMatrixBuilder<Real> sparseBuilder{};
  VecXd x_prev;
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_INTEGRATOR_H_
