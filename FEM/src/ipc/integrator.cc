//
// Created by creeper on 10/25/24.
//
#include <fem/system.h>
#include <fem/ipc/distances.h>
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/integrator.h>


namespace fem {

SparseMatrix<fem::Real> IpcIntegrator::spdProjectHessian(Real h) {
  sparseBuilder.clear().setRows(system().dof()).setColumns(system().dof());
  system().spdProjectHessian(sparseBuilder);
  Real kappa = config.contactStiffness;
  for (const auto &c : constraintSet.vtConstraints)
    c.assembleBarrierHessian(barrier, sparseBuilder, kappa);
  for (const auto &c : constraintSet.eeConstraints)
    c.assembleMollifiedBarrierHessian(barrier, sparseBuilder, kappa);
  return sparseBuilder.build() * h * h + system().mass();
}

void IpcIntegrator::updateConstraintStatus() {
  for (auto &c : constraintSet.vtConstraints)
    c.updateDistanceType();
  for (auto &c : constraintSet.eeConstraints)
    c.updateDistanceType();
}

void IpcIntegrator::precomputeConstraintSet(const ConstraintSetPrecomputeRequest &config) {
  constraintSet.vtConstraints.clear();
  constraintSet.eeConstraints.clear();
  computeVertexTriangleConstraints(config);
  computeEdgeEdgeConstraints(config);
}

static BBox<Real, 3> computeVertexTrajectoryBBox(const ipc::Trajectory &trajectory) {
  const auto &[system, p, toi, i] = trajectory;
  const auto &x = system.currentConfig();
  auto v = x.segment<3>(i * 3);
  auto v_hat = v + p.segment<3>(i * 3) * toi;
  return BBox<Real, 3>({v(0), v(1), v(2)})
      .expand({v_hat(0), v_hat(1), v_hat(2)});
}

struct EdgeTrajectoryAccessor {
  const System &system;
  const VecXd &p;
  Real toi = 1.0;
  using CoordType = Real;

  EdgeTrajectoryAccessor(const System &system, const VecXd &p, Real toi) : system(system), p(p), toi(toi) {}
  [[nodiscard]] BBox<Real, 3> bbox(int i) const {
    const auto &x = system.currentConfig();
    int ia = system.edges()(0, i);
    assert(ia < system.numVertices());
    int ib = system.edges()(1, i);
    assert(ib < system.numVertices());
    auto v0 = x.segment<3>(ia * 3);
    auto v1 = x.segment<3>(ib * 3);
    auto v2 = x.segment<3>(ia * 3) + p.segment<3>(ia * 3) * toi;
    auto v3 = x.segment<3>(ib * 3) + p.segment<3>(ib * 3) * toi;
    return BBox<Real, 3>({v0(0), v0(1), v0(2)}).expand({v1(0), v1(1), v1(2)})
        .expand({v2(0), v2(1), v2(2)}).expand({v3(0), v3(1), v3(2)});
  }

  [[nodiscard]] int size() const {
    return system.numTriangles();
  }
};

void IpcIntegrator::computeVertexTriangleConstraints(const ConstraintSetPrecomputeRequest &config) {
  const auto &[p, toi, dHat] = config;
  trianglesBVH->update(ipc::SystemTriangleTrajectoryAccessor(system(), p, toi));
  for (int i = 0; i < system().numVertices(); i++) {
    BBox<Real, 3> vertex_trajectory_bbox = computeVertexTrajectoryBBox({system(), p, toi, i}).dilate(dHat);
    trianglesBVH->runSpatialQuery(
        [&](int triangle_idx) -> bool {
          for (int j = 0; j < 3; j++)
            if (system().surfaces()(j, triangle_idx) == i)
              return false;
          constraintSet.vtConstraints.push_back(
              {.system = system(),
                  .iv = i,
                  .it = triangle_idx});
          return true;
        }, [&](const BBox<Real, 3> &bbox) -> bool {
          return vertex_trajectory_bbox.overlap(bbox);
        });
  }
}

void IpcIntegrator::computeEdgeEdgeConstraints(const ConstraintSetPrecomputeRequest &config) {
  const auto &[p, toi, dHat] = config;
  const auto &x = system().currentConfig();
  EdgeTrajectoryAccessor edge_accessor(system(), p, toi);
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

VecXd IpcIntegrator::barrierEnergyGradient() const {
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

Real IpcIntegrator::barrierEnergy() const {
  Real barrierEnergy = 0.0;
  Real kappa = config.contactStiffness;
  for (const auto &c : constraintSet.vtConstraints)
    barrierEnergy += barrier(c.distanceSqr());
  for (const auto &c : constraintSet.eeConstraints)
    barrierEnergy += c.mollifier() * barrier(c.distanceSqr());
  return kappa * barrierEnergy;
}
}