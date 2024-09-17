//
// Created by creeper on 5/29/24.
//
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/implicit-euler.h>
#include <fem/ipc/distances.h>
#include <fem/system.h>
namespace fem {
constexpr Real kStepSizeScale = 0.9;

IpcImplicitEuler::IpcImplicitEuler(System &system, const Config &config) : IpcIntegrator(system, config) {
  collisionDetector = std::make_unique<ipc::CollisionDetector>();
  edgesBVH = std::make_unique<spatify::LBVH<Real>>();
  trianglesBVH = std::make_unique<spatify::LBVH<Real>>();
}

void IpcImplicitEuler::step(Real dt) {
  x_prev = system().currentConfig();
  VecXd x_t = system().currentConfig();
  Real h = dt;
  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
  VecXd p(system().dof());
  int iter = 0;
  while (true) {
    g = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
    auto H = spdProjectHessian(h);
    if (ldlt.compute(H).info() != Eigen::Success)
      core::ERROR("Failed to perform LDLT decomposition");
    p = ldlt.solve(-g);
    if (ldlt.info() != Eigen::Success)
      core::ERROR("Failed to solve triangular systems");
    if (p.lpNorm<Eigen::Infinity>() < config.eps * system().meshLengthScale() * h)
      break;
    Real alpha = kStepSizeScale * std::min(1.0, computeStepSizeUpperBound(p));
    if (alpha == 0.0)
      throw std::runtime_error("Invalid state: collision happened within an integration step");
    precomputeConstraintSet(
        {.descentDir = p,
            .toi = alpha,
            .dHat = config.dHat});
    Real E;
    do {
      updateCandidateSolution(x_prev + alpha * p);
      alpha = alpha * 0.5;
      E = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    } while (E > E_prev);
    iter++;
    x_prev = system().currentConfig();
    E_prev = E;
  }
  system().xdot = (system().currentConfig() - x_t) / h;
}

SparseMatrix<Real> IpcImplicitEuler::spdProjectHessian(Real h) {
  if (constraintsDirty) {
    updateConstraintStatus();
    constraintsDirty = false;
  }
  sparseBuilder.clear().setRows(system().dof()).setColumns(system().dof());
  system().spdProjectHessian(sparseBuilder);
  Real kappa = config.contactStiffness;
  for (const auto &c : constraintSet.vtConstraints)
    c.assembleBarrierHessian(barrier, sparseBuilder, kappa);
  for (const auto &c : constraintSet.eeConstraints)
    c.assembleMollifiedBarrierHessian(barrier, sparseBuilder, kappa);
  return sparseBuilder.build() * h * h + system().mass();
}

Real IpcImplicitEuler::incrementalPotentialKinematicEnergy(const VecXd &x_t, Real h) const {
  const auto &v_t = system().xdot;
  const auto &f_e = system().f_ext;
  auto x_hat = x_t + h * v_t + h * h * system().massLDLT().solve(f_e);
  return 0.5 * (system().currentConfig() - x_hat).transpose() * system().mass() * (system().currentConfig() - x_hat);
}

void IpcImplicitEuler::updateConstraintStatus() {
  for (auto &c : constraintSet.vtConstraints)
    c.updateDistanceType();
  for (auto &c : constraintSet.eeConstraints)
    c.updateDistanceType();
}

Real IpcImplicitEuler::barrierAugmentedIncrementalPotentialEnergy(const VecXd &x_t, Real h) {
  return incrementalPotentialKinematicEnergy(x_t, h) + h * h * barrierAugmentedPotentialEnergy();
}

void IpcImplicitEuler::precomputeConstraintSet(const ConstraintSetPrecomputeRequest &config) {
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

void IpcImplicitEuler::computeVertexTriangleConstraints(const ConstraintSetPrecomputeRequest &config) {
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

void IpcImplicitEuler::computeEdgeEdgeConstraints(const ConstraintSetPrecomputeRequest &config) {
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

VecXd IpcImplicitEuler::incrementalPotentialKinematicEnergyGradient(const VecXd &x_t, Real h) {
  const auto &v_t = system().xdot;
  const auto &f_e = system().f_ext;
  auto x_hat = x_t + h * v_t + h * h * system().massLDLT().solve(f_e);
  return system().mass() * (system().currentConfig() - x_hat);
}

VecXd IpcImplicitEuler::barrierAugmentedIncrementalPotentialEnergyGradient(const VecXd &x_t, Real h) {
  return incrementalPotentialKinematicEnergyGradient(x_t, h)
      + h * h * barrierAugmentedPotentialEnergyGradient();
}

Real IpcImplicitEuler::computeStepSizeUpperBound(const VecXd &p) const {
  auto t = collisionDetector->detect(system(), p);
  if (!t)
    return 1.0;
  return *t;
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