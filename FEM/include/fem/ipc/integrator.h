//
// Created by creeper on 10/22/24.
//

#pragma once
#include <Maths/sparse-matrix-builder.h>
#include <Core/debug.h>
#include <fem/integrator.h>
#include <fem/ipc/constraint.h>
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/barrier-functions.h>
#include <Maths/linear-solver.h>

namespace sim::fem {
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
      REFLECT(dHat, eps, contactStiffness, stepSizeScale);
    } config;

    explicit IpcIntegrator(System &system, const Config &config)
      : Integrator(system), config(config), barrier(config.dHat) {
      collisionDetector = std::make_unique<ipc::CollisionDetector>(system);
    }

    void step(Real dt) override;
    std::unique_ptr<maths::LinearSolver> linearSolver;
    static std::unique_ptr<Integrator> create(System &system, const core::JsonNode &json);

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
      collisionDetector->updateBVHs(p, 1.0);
      auto t = collisionDetector->detect(p);
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
    maths::SparseMatrixBuilder<Real> sparseBuilder{};
    VecXd x_prev;
};
}
