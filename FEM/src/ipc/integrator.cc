//
// Created by creeper on 10/25/24.
//
#include <Maths/linear-solver.h>
#include <fem/integrator-factory.h>
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/distances.h>
#include <fem/ipc/implicit-euler.h>
#include <fem/ipc/integrator.h>
#include <fem/system.h>
#include <tbb/parallel_for.h>

namespace sim::fem {

static int ipc_auto_reg = ([]() {
    IntegratorFactory::instance().registerCreator(
        "ipc",
        [](System &system, const core::JsonNode &json) {
          return IpcIntegrator::create(system, json);
        });
  }(), 0);

void IpcIntegrator::step(Real dt) {
  x_prev = system().currentConfig();
  VecXd x_t = system().currentConfig();
  Real h = dt;
  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
  VecXd p(system().dof());
  VecXd g(system().dof());
  int iter = 0;
  while (true) {
    g = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
    auto H = spdProjectHessian(h);
    p = linearSolver->solve(H, -g);
    if (!linearSolver->success())
      throw std::runtime_error("Failed to solve triangular systems");
    if (p.lpNorm<Eigen::Infinity>() <
        config.eps * system().meshLengthScale() * h)
      break;
    Real alpha =
        config.stepSizeScale * std::min(1.0, computeStepSizeUpperBound(p));
    if (alpha == 0.0)
      throw std::runtime_error(
          "Invalid state: collision happened within an integration step");
    precomputeConstraintSet(
        {.descentDir = p, .toi = alpha, .dHat = config.dHat});
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
  velocityUpdate(x_t, h);
}

SparseMatrix<Real> IpcIntegrator::spdProjectHessian(Real h) {
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

void IpcIntegrator::precomputeConstraintSet(
    const ConstraintSetPrecomputeRequest &config) {
  constraintSet.vtConstraints.clear();
  constraintSet.eeConstraints.clear();
  collisionDetector->updateBVHs(config.descentDir, config.toi);
  computeVertexTriangleConstraints(config);
  computeEdgeEdgeConstraints(config);
}

void IpcIntegrator::computeVertexTriangleConstraints(
    const ConstraintSetPrecomputeRequest &config) {
  const auto &[p, toi, dHat] = config;

  tbb::parallel_for(0, system().numVertices(), [&](int vertexIdx) {
    auto vertexTrajectoryBBox =
        system()
            .geometryManager()
            .getTrajectoryAccessor(system().currentConfig(), p, toi)
            .vertexBBox(vertexIdx);
    vertexTrajectoryBBox = vertexTrajectoryBBox.dilate(dHat);
    const auto &primitive = system().primitive(
        system().geometryManager().getVertexPrimitiveId(vertexIdx));
    collisionDetector->trianglesBVH().runSpatialQuery(
        [&](int triangleIdx) -> bool {
          if (system().triangleContainsVertex(triangleIdx, vertexIdx))
            return false;
          auto triangle = system().geometryManager().getTriangle(triangleIdx);
          const auto &x = system().currentConfig();

          constraintSet.vtConstraints.push_back({
              .triangle = triangle,
              .xv = primitive.cview(x),
              .xt = primitive.cview(x),
              .iv = vertexIdx,
              .type = ipc::PointTriangleDistanceType::Unknown,
          });

          constraintSet.vtConstraints.back().updateDistanceType();

          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return vertexTrajectoryBBox.overlap(bbox);
        });
  });
}

void IpcIntegrator::computeEdgeEdgeConstraints(
    const ConstraintSetPrecomputeRequest &config) {
  const auto &[p, toi, dHat] = config;

  tbb::parallel_for(0, system().numEdges(), [&](int edgeIdx) {
    auto edgeTrajectoryBBox =
        system()
            .geometryManager()
            .getTrajectoryAccessor(system().currentConfig(), p, toi)
            .edgeBBox(edgeIdx);
    edgeTrajectoryBBox = edgeTrajectoryBBox.dilate(dHat);

    collisionDetector->edgesBVH().runSpatialQuery(
        [&](int otherEdgeIdx) -> bool {
          if (system().checkEdgeAdjacent(edgeIdx, otherEdgeIdx))
            return false;

          const auto &ea = system().geometryManager().getEdge(edgeIdx);
          const auto &eb = system().geometryManager().getEdge(otherEdgeIdx);
          auto pr_a_id = system().globalEdgeToPrimitive(edgeIdx);
          auto pr_b_id = system().globalEdgeToPrimitive(otherEdgeIdx);
          const auto &pr_a = system().primitive(pr_a_id);
          const auto &pr_b = system().primitive(pr_b_id);
          const auto &x = system().currentConfig();
          const auto &X = system().referenceConfig();

          constraintSet.eeConstraints.push_back({
              .ea = ea,
              .eb = eb,
              .xa = pr_a.cview(x),
              .Xa = pr_a.cview(X),
              .xb = pr_b.cview(x),
              .Xb = pr_b.cview(X),
              .type = ipc::EdgeEdgeDistanceType::Unknown,
          });

          constraintSet.eeConstraints.back().updateDistanceType();

          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return edgeTrajectoryBBox.overlap(bbox);
        });
  });
}

VecXd IpcIntegrator::barrierEnergyGradient() const {
  VecXd gradient = VecXd::Zero(system().dof());
  gradient.setZero();
  Real kappa = config.contactStiffness;
  for (const auto &c : constraintSet.vtConstraints)
    c.assembleBarrierGradient(barrier, gradient, kappa);
  for (const auto &c : constraintSet.eeConstraints)
    c.assembleMollifiedBarrierGradient(barrier, gradient, kappa);
  return gradient;
}

std::unique_ptr<Integrator> IpcIntegrator::create(System &system,
                                                  const core::JsonNode &json) {
  std::unordered_map<std::string,
                     std::function<std::unique_ptr<IpcIntegrator>(
                         System &, const IpcIntegrator::Config &cfg)>>
      integratorCreators = {
          {"implicit-euler",
           [](System &system, const Config &cfg) {
             return std::make_unique<IpcImplicitEuler>(system, cfg);
           }},
      };

  if (!json.is<core::JsonDict>())
    throw std::runtime_error("Expected a JSON object for IpcIntegrator");
  const auto &dict = json.as<core::JsonDict>();
  if (!dict.contains("type"))
    throw std::runtime_error("IpcIntegrator missing type field");
  const auto &subtype = dict.at("type").as<std::string>();

  auto config = core::deserialize<Config>(dict.at("config"));
  auto integrator = integratorCreators.at(subtype)(system, config);

  if (dict.contains("linearSolver"))
    integrator->linearSolver =
        maths::createLinearSolver(dict.at("linearSolver"));
  return integrator;
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
} // namespace sim::fem