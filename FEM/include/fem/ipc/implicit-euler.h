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

class IpcIntegrator : public Integrator {
 public:
  struct Config {
    Real dHat;
    Real eps;
  } config;

  explicit IpcIntegrator(System &system, const Config& config) : Integrator(system), config(config) {}
 protected:
  ConstraintSet constraintSet;
  std::unique_ptr<ipc::CollisionDetector> collisionDetector{};
};

inline Real unconstrainedLinfNorm(const VecXd &p, const std::vector<bool> &constrained) {
  Real norm = 0.0;
  for (int i = 0; i < p.size(); i++)
    if (!constrained[i])
      norm = std::max(norm, std::abs(p[i]));
  return norm;
}

struct ConstraintSetPrecomputeRequest {
  const VecXd &currentConfig;
  const VecXd &descentDir;
  Real alpha;
  Real dHat;
};

struct IpcImplicitEuler : public IpcIntegrator {
  explicit IpcImplicitEuler(System &system, const Config& config);
  void step(Real dt) override;
 private:
  SparseMatrix<Real> spdProjectHessian();
  Real incrementalPotentialEnergy(const VecXd &x_t, Real h) const;
  // for now, do not consider dissipative friction forces
  Real barrierAugmentedIncrementalPotentialEnergy(const VecXd &x_t, Real h);
  auto incrementalPotentialEnergyGradient(const VecXd &x_t, Real h);
  VecXd barrierAugmentedIncrementalPotentialEnergyGradient(const VecXd &x_t, Real h);
  void updateCandidateSolution(const VecXd &x) {
    system().updateCurrentConfig(x);
    constraints_dirty = true;
  }
  void updateConstraintStatus();
  Real computeStepSizeUpperBound(const VecXd &p) const;
  void precomputeConstraintSet(const ConstraintSetPrecomputeRequest &config);
  void computeVertexTriangleConstraints(const ConstraintSetPrecomputeRequest &config);
  void computeEdgeEdgeConstraints(const ConstraintSetPrecomputeRequest &config);
  void adaptStiffness() {

  }
  maths::SparseMatrixBuilder<Real> sparseBuilder;
  Eigen::SimplicialLDLT<SparseMatrix<Real>> ldlt;
  std::unique_ptr<spatify::LBVH<Real>> edgesBVH;
  ipc::LogBarrier barrier;
  Real kappa{};
  core::Logger logger;
  VecXd x_prev, x_t_buf;
  VecXd g;
  std::vector<bool> constrained_cache;
  bool constraints_dirty{false};
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
