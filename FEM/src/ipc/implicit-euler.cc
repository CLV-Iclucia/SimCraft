//
// Created by creeper on 5/29/24.
//
#include <fem/ipc/implicit-euler.h>
#include <fem/ipc/distances.h>
#include <fem/system.h>
namespace fem {
constexpr Real kStepSizeScale = 0.9;

IpcImplicitEuler::IpcImplicitEuler(System &system, const Config &config) : IpcIntegrator(system, config) {
  collisionDetector = std::make_unique<ipc::CollisionDetector>();
  edgesBVH = std::make_unique<spatify::LBVH<Real>>();
}

void IpcImplicitEuler::step(Real dt) {
  auto &x_t = x_t_buf;
  x_prev = system().currentConfig();
  x_t = system().currentConfig();
  Real h = dt;
  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
  VecXd p(x_t.size());
  while (true) {
    auto H = spdProjectHessian();
    g = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
    if (ldlt.compute(H).info() != Eigen::Success)
      core::ERROR("Failed to perform LDLT decomposition");
    p = ldlt.solve(-g);
    if (ldlt.info() != Eigen::Success)
      core::ERROR("Failed to solve triangular systems");
    if (p.lpNorm<Eigen::Infinity>() > config.eps)
      break;
    Real alpha = kStepSizeScale * std::min(1.0, computeStepSizeUpperBound(p));
    precomputeConstraintSet(
        {.currentConfig = system().currentConfig(),
            .descentDir = p,
            .alpha = alpha,
            .dHat = config.dHat});
    Real E;
    do {
      updateCandidateSolution(x_prev + alpha * p);
      alpha = alpha * 0.5;
      E = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    } while (E > E_prev);
    x_prev = system().currentConfig();
    E_prev = E;
  }
}

SparseMatrix<Real> IpcImplicitEuler::spdProjectHessian() {
  sparseBuilder.clear().setRows(system().dof()).setColumns(system().dof());
  system().spdProjectHessian(sparseBuilder);
  for (const auto &c : constraintSet.vtConstraints)
    c.assembleBarrierHessian(barrier, sparseBuilder, kappa);
  for (const auto &c : constraintSet.eeConstraints)
    c.assembleMollifiedBarrierHessian(barrier, sparseBuilder, kappa);
  return sparseBuilder.build();
}

Real IpcImplicitEuler::incrementalPotentialEnergy(const VecXd &x_t, Real h) const {
  const auto &v_t = system().xdot;
  const auto &f_e = system().f_ext;
  auto x_hat = x_t + h * v_t + h * h * system().massLDLT().solve(f_e);
  return 0.5 * (system().currentConfig() - x_hat).transpose() * system().mass() * (system().currentConfig() - x_hat)
      + h * h * system().deformationEnergy();
}

void IpcImplicitEuler::updateConstraintStatus() {
  for (auto &c : constraintSet.vtConstraints)
    c.updateDistanceType();
  for (auto &c : constraintSet.eeConstraints)
    c.updateDistanceType();
}

Real IpcImplicitEuler::barrierAugmentedIncrementalPotentialEnergy(const VecXd &x_t, Real h) {
  if (constraints_dirty) {
    updateConstraintStatus();
    constraints_dirty = false;
  }
  Real barrier_sum = 0.0;
  for (const auto &c : constraintSet.vtConstraints)
    barrier_sum += barrier(c.distance());
  for (const auto &c : constraintSet.eeConstraints)
    barrier_sum += barrier(c.distance());
  return incrementalPotentialEnergy(x_t, h) + kappa * barrier_sum;
}

void IpcImplicitEuler::precomputeConstraintSet(const ConstraintSetPrecomputeRequest &config) {
  if (!constraints_dirty)
    return;
  constraintSet.vtConstraints.clear();
  constraintSet.eeConstraints.clear();
  computeVertexTriangleConstraints(config);
  computeEdgeEdgeConstraints(config);
  constraints_dirty = false;
}

void IpcImplicitEuler::computeVertexTriangleConstraints(const ConstraintSetPrecomputeRequest &config) {
  const auto &[x, p, alpha, d_hat] = config;
  const auto &trianglesBVH = collisionDetector->bvh();
  for (int vertex_index : system().vertices()) {
    auto vertex_pos = x.segment<3>(vertex_index * 3);
    auto vertex_pos_next = vertex_pos + p.segment<3>(vertex_index * 3) * kStepSizeScale;
    BBox<Real, 3> vertex_trajectory_bbox = BBox<Real, 3>({vertex_pos(0), vertex_pos(1), vertex_pos(2)})
        .expand({vertex_pos_next(0), vertex_pos_next(1), vertex_pos_next(2)}).dilate(d_hat);
    trianglesBVH.runSpatialQuery(
        [&](int triangle_idx) -> bool {
          for (int i = 0; i < 3; i++)
            if (system().surfaces()(i, triangle_idx) == vertex_index)
              return false;
          constraintSet.vtConstraints.push_back(
              {.system = system(),
                  .iv = vertex_index,
                  .it = triangle_idx});
          return true;
        }, [&](const BBox<Real, 3> &bbox) -> bool {
          return vertex_trajectory_bbox.overlap(bbox);
        });
  }
}

struct EdgeTrajectoryAccessor {
  const System &system;
  const ConstraintSetPrecomputeRequest &config;
  using CoordType = Real;
  EdgeTrajectoryAccessor(const System &system, const ConstraintSetPrecomputeRequest &config)
      : system(system), config(config) {}
  [[nodiscard]] BBox<Real, 3> bbox(int i) const {
    const auto &[x, p, alpha, d_hat] = config;
    auto v0 = x.segment<3>(system.edges()(0, i) * 3);
    auto v1 = x.segment<3>(system.edges()(1, i) * 3);
    auto v2 = x.segment<3>(system.edges()(0, i) * 3)
        + p.segment<3>(system.edges()(0, i) * 3) * kStepSizeScale;
    auto v3 = x.segment<3>(system.edges()(1, i) * 3)
        + p.segment<3>(system.edges()(1, i) * 3) * kStepSizeScale;
    return BBox<Real, 3>({v0(0), v0(1), v0(2)})
        .expand({v1(0), v1(1), v1(2)})
        .expand({v2(0), v2(1), v2(2)})
        .expand({v3(0), v3(1), v3(2)})
        .dilate(d_hat);
  }
  [[nodiscard]] int size() const {
    return system.numTriangles();
  }
};

void IpcImplicitEuler::computeEdgeEdgeConstraints(const ConstraintSetPrecomputeRequest &config) {
  const auto &[x, p, alpha, d_hat] = config;
  EdgeTrajectoryAccessor edge_accessor(system(), config);
  edgesBVH->update(edge_accessor);
  for (int edge_index = 0; edge_index < system().edges().cols(); edge_index++) {
    auto q_edge_pa = x.segment<3>(system().edges()(0, edge_index) * 3);
    auto q_edge_pb = x.segment<3>(system().edges()(1, edge_index) * 3);
    BBox<Real, 3> edge_trajectory_bbox = BBox<Real, 3>({q_edge_pa(0), q_edge_pa(1), q_edge_pa(2)})
        .expand({q_edge_pb(0), q_edge_pb(1), q_edge_pb(2)});
    edgesBVH->runSpatialQuery(
        [&](int edge_idx) -> bool {
          if (system().checkEdgeAdjacent(edge_index, edge_idx))
            return false;
          constraintSet.eeConstraints.push_back(
              {.system = system(),
                  .ia = edge_index,
                  .ib = edge_idx});
          return true;
        }, [&](const BBox<Real, 3> &bbox) -> bool {
          return edge_trajectory_bbox.overlap(bbox);
        });
  }
}

auto IpcImplicitEuler::incrementalPotentialEnergyGradient(const VecXd &x_t, Real h) {
  const auto &v_t = system().xdot;
  const auto &f_e = system().f_ext;
  auto x_hat = x_t + h * v_t + h * h * system().massLDLT().solve(f_e);
  return system().mass() * (system().currentConfig() - x_hat) + h * h * system().deformationEnergyGradient();
}

VecXd IpcImplicitEuler::barrierAugmentedIncrementalPotentialEnergyGradient(const VecXd &x_t, Real h) {
  if (constraints_dirty) {
    updateConstraintStatus();
    constraints_dirty = false;
  }
  VecXd barrier_energy_gradient = VecXd::Zero(system().currentConfig().size());
  for (const auto &c : constraintSet.vtConstraints)
    c.assembleBarrierGradient(barrier, barrier_energy_gradient, kappa);
  for (const auto &c : constraintSet.eeConstraints)
    c.assembleMollifiedBarrierGradient(barrier, barrier_energy_gradient, kappa);
  return incrementalPotentialEnergyGradient(x_t, h) + kappa * barrier_energy_gradient;
}

Real IpcImplicitEuler::computeStepSizeUpperBound(const VecXd &p) const {
  auto t = collisionDetector->detect(system(), p);
  if (!t)
    return 1.0;
  return *t;
}

}