//
// Created by creeper on 10/25/24.
//
#include <Maths/linear-solver.h>
#include <Maths/block-solvers/block-pcg.h>
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
  x_prev = system().x;
  maths::BlockVector<3> x_t = system().x;
  Real h = dt;
  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
  maths::BlockVector<3> p(x_t.numBlocks());
  int iter = 0;
  while (true) {
    VecXd g = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
    auto H_eigen = spdProjectHessian(h);

    // --- Block solver boundary ---
    auto H_block = maths::BlockSparseMatrix<3>::fromEigen(H_eigen);
    maths::BlockVector<3> negG(x_t.numBlocks());
    negG.asEigen() = -g;

    p.setZero();
    auto result = solver->solve(H_block, negG, p);
    if (!result.converged)
      spdlog::warn("BlockPCG: {} iters, residual={}", result.iterations, result.residualNorm);
    // --- end Block solver boundary ---

    if (p.asEigen().template lpNorm<Eigen::Infinity>() <
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
      maths::BlockVector<3> x_candidate = x_prev;
      x_candidate.axpy(alpha, p);
      updateCandidateSolution(x_candidate);
      alpha = alpha * 0.5;
      E = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    } while (E > E_prev);
    iter++;
    x_prev = system().x;
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
  collisionDetector->updateBVHs(config.descentDir.asEigen(), config.toi);
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
            .getTrajectoryAccessor(system().x, p, toi)
            .vertexBBox(vertexIdx);
    vertexTrajectoryBBox = vertexTrajectoryBBox.dilate(dHat);
    const auto &primitive = system().primitive(
        system().geometryManager().getVertexPrimitiveId(vertexIdx));
    collisionDetector->trianglesBVH().runSpatialQuery(
        [&](int triangleIdx) -> bool {
          if (system().triangleContainsVertex(triangleIdx, vertexIdx))
            return false;
          auto triangle = system().geometryManager().getTriangle(triangleIdx);

          constraintSet.vtConstraints.push_back({
              .triangle = triangle,
              .xv = primitive.cview(system().x),
              .xt = primitive.cview(system().x),
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
            .getTrajectoryAccessor(system().x, p, toi)
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

          constraintSet.eeConstraints.push_back({
              .ea = ea,
              .eb = eb,
              .xa = pr_a.cview(system().x),
              .Xa = pr_a.cview(system().X),
              .xb = pr_b.cview(system().x),
              .Xb = pr_b.cview(system().X),
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

  // Create block solver (replaces legacy linearSolver creation)
  int maxIter = 1000;
  Real tol = 1e-6;
  if (dict.contains("solver")) {
    const auto &sDict = dict.at("solver").as<core::JsonDict>();
    if (sDict.contains("maxIterations")) maxIter = sDict.at("maxIterations").as<int>();
    if (sDict.contains("tolerance")) tol = sDict.at("tolerance").as<Real>();
  }
  integrator->solver = std::make_unique<maths::BlockPCGSolver>(maxIter, tol);
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