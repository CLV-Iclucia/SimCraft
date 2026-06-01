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
  Real t_next = system().currentTime() + dt;
  system().advanceKinematicBodies(t_next);

  collisionDetector->setKinematicBodies(&system().kinematicBodies());

  system().constraints().enforcePosition(system(), t_next);
  maths::BlockVector<3> x_t = system().x;
  x_prev = system().x;

  Real h = dt;
  spdlog::info("[IPC] computing initial energy...");
  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
  spdlog::info("[IPC] E_prev = {}", E_prev);
  maths::BlockVector<3> p(x_t.numBlocks());
  int iter = 0;
  while (true) {
    // Compute negative gradient directly as BlockVector<3>
    spdlog::info("[IPC] iter {}: computing gradient...", iter);
    auto negG = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
    negG *= -1.0;

    system().constraints().zeroConstrainedGradient(negG);
    // Compute Hessian directly as BlockSparseMatrix<3>
    spdlog::info("[IPC] iter {}: computing Hessian...", iter);
    auto H_block = spdProjectHessian(h);

    spdlog::info("[IPC] iter {}: solving linear system...", iter);
    p.setZero();
    spdlog::info("[IPC] nnz: {}", H_block.blocks().size());
    auto result = solver->solve(H_block, negG, p);
    if (!result.converged)
      spdlog::warn("BlockPCG: {} iters, residual={}", result.iterations, result.residualNorm);

    system().constraints().projectToFreeSpace(p);

    std::cout << "norm: " << p.infNorm() << " " << negG.infNorm() << std::endl;
    if (p.infNorm() <
        config.eps * system().meshLengthScale() * h)
      break;

    spdlog::info("[IPC] iter {}: computing step size upper bound...", iter);
    Real alphaElastic = computeStepSizeUpperBound(p);
    Real alphaKinematic = 1.0;
    if (!system().kinematicBodies().empty()) {
      auto toiKin = collisionDetector->detectDeformableVsKinematic(p, dt);
      if (toiKin) alphaKinematic = *toiKin;
    }
    Real alpha = config.stepSizeScale * std::min({1.0, alphaElastic, alphaKinematic});
    spdlog::info("[IPC] iter {}: alpha={}", iter, alpha);

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
      system().constraints().enforcePosition(system(), t_next);
      alpha = alpha * 0.5;
      E = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    } while (E > E_prev);
    iter++;
    x_prev = system().x;
    std::cout << E << " " << E_prev << std::endl;
    E_prev = E;

    if (onNewtonIter) onNewtonIter(iter);
  }
  velocityUpdate(x_t, h);

  system().constraints().enforceVelocity(system(), t_next);

  system().advanceTime(dt);

  Real T = system().kineticEnergy();
  Real V = system().potentialEnergy();
  Real Vg = system().gravitationalPotentialEnergy();
  Real total = T + V + Vg;
  spdlog::info("[IPC] t={:.4f}  T={:.6e}  V_elastic={:.6e}  V_gravity={:.6e}  Total={:.6e}",
               system().currentTime(), T, V, Vg, total);
  // Diagnostic: log first vertex position to verify system.x is changing
  if (system().x.numBlocks() > 0) {
    auto v0 = system().x[0];
    spdlog::debug("[IPC] step done: x[0]=({:.6f},{:.6f},{:.6f})", v0.x, v0.y, v0.z);
  }
}

maths::BlockSparseMatrix<3> IpcIntegrator::spdProjectHessian(Real h) const
{
  int nBlocks = system().x.numBlocks();
  maths::BlockSparseMatrix<3> H(nBlocks, nBlocks);
  H.setSymmetric(true);

  // Elastic Hessian
  system().spdProjectHessian(H);

  // Barrier Hessian (elastic-elastic)
  Real kappa = config.contactStiffness;
  for (const auto &c : constraintSet.vtConstraints)
    c.assembleBarrierHessian(barrier, H, kappa);
  for (const auto &c : constraintSet.eeConstraints)
    c.assembleMollifiedBarrierHessian(barrier, H, kappa);

  // Barrier Hessian (elastic-kinematic)
  for (const auto &c : constraintSet.kinematicVTConstraints)
    c.assembleBarrierHessian(barrier, H, kappa);

  // H_total = h² * H_elastic_barrier + M
  H.scale(h * h);
  H.addFrom(system().blockMass());

  return H;
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
  constraintSet.kinematicVTConstraints.clear();
  
  collisionDetector->updateBVHs(config.descentDir, config.toi);
  computeVertexTriangleConstraints(config);
  computeEdgeEdgeConstraints(config);
  
  // 检测运动学体约束
  if (!system().kinematicBodies().empty()) {
    computeKinematicVTConstraints(config);
  }
}

void IpcIntegrator::computeKinematicVTConstraints(
    const ConstraintSetPrecomputeRequest &config) {
  const auto &[p, toi, dHat] = config;
  
  // 遍历所有运动学体
  for (size_t bodyIdx = 0; bodyIdx < system().kinematicBodies().size(); bodyIdx++) {
    const auto& body = system().kinematicBodies()[bodyIdx];
    auto* mg = std::get_if<KinematicBody::MeshGeometry>(&body.geometry);
    if (!mg) continue;  // SDF 碰撞在 barrier 层单独处理
    
    const auto& triangles = mg->mesh->triangles;
    
    for (int vertexIdx = 0; vertexIdx < system().numVertices(); vertexIdx++) {
      // 弹性体顶点轨迹 bbox
      auto startPos = system().x[vertexIdx];
      auto endPos = startPos + p[vertexIdx] * toi;
      BBox<Real, 3> vertexBBox;
      vertexBBox.expand({startPos.x, startPos.y, startPos.z});
      vertexBBox.expand({endPos.x, endPos.y, endPos.z});
      
      // 遍历运动学体的三角形
      for (size_t triIdx = 0; triIdx < triangles.size(); triIdx++) {
        const auto& tri = triangles[triIdx];
        
        // 计算运动学三角形在当前时刻的位置
        glm::dvec3 ka = body.currentVertices[tri.x];
        glm::dvec3 kb = body.currentVertices[tri.y];
        glm::dvec3 kc = body.currentVertices[tri.z];
        
        // 简单距离检测：如果弹性体顶点接近运动学三角形，添加约束
        Real distSqr = ipc::distanceSqrPointTriangle(startPos, ka, kb, kc);
        
        if (distSqr < dHat * dHat) {
          constraintSet.kinematicVTConstraints.push_back({
              .x = system().x,
              .deformableVertex = vertexIdx,
              .ka = ka,
              .kb = kb,
              .kc = kc,
              .type = ipc::PointTriangleDistanceType::Unknown,
          });
          constraintSet.kinematicVTConstraints.back().updateDistanceType();
        }
      }
    }
  }
}

void IpcIntegrator::computeVertexTriangleConstraints(
    const ConstraintSetPrecomputeRequest &config) {
  const auto &[p, toi, dHat] = config;
  const int nVerts = system().numVertices();

  tbb::enumerable_thread_specific<std::vector<ipc::VertexTriangleConstraint>> threadLocalVT;

  tbb::parallel_for(0, nVerts, [&](int vertexIdx) {
    auto vertexTrajectoryBBox =
        system().geometryManager()
            .getTrajectoryAccessor(system().x, p, toi)
            .vertexBBox(vertexIdx);
    vertexTrajectoryBBox = vertexTrajectoryBBox.dilate(dHat);

    auto &local = threadLocalVT.local();
    collisionDetector->trianglesBVH().runSpatialQuery(
        [&](int triangleIdx) -> bool {
          if (system().triangleContainsVertex(triangleIdx, vertexIdx))
            return false;

          auto globalTri = system().geometryManager().getGlobalTriangle(triangleIdx);
          local.push_back({
              .x = system().x,
              .globalVertex = vertexIdx,
              .globalTriVerts = {globalTri.x, globalTri.y, globalTri.z},
              .type = ipc::PointTriangleDistanceType::Unknown,
          });
          local.back().updateDistanceType();
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return vertexTrajectoryBBox.overlap(bbox);
        });
  });

  // Sequential merge
  for (auto &local : threadLocalVT)
    for (auto &c : local)
      constraintSet.vtConstraints.push_back(c);
}

void IpcIntegrator::computeEdgeEdgeConstraints(
    const ConstraintSetPrecomputeRequest &config) {
  const auto &[p, toi, dHat] = config;
  const int nEdges = system().numEdges();

  tbb::enumerable_thread_specific<std::vector<ipc::EdgeEdgeConstraint>> threadLocalEE;

  tbb::parallel_for(0, nEdges, [&](int edgeIdx) {
    auto edgeTrajectoryBBox =
        system().geometryManager()
            .getTrajectoryAccessor(system().x, p, toi)
            .edgeBBox(edgeIdx);
    edgeTrajectoryBBox = edgeTrajectoryBBox.dilate(dHat);

    auto &local = threadLocalEE.local();
    collisionDetector->edgesBVH().runSpatialQuery(
        [&](int otherEdgeIdx) -> bool {
          if (system().checkEdgeAdjacent(edgeIdx, otherEdgeIdx))
            return false;

          auto globalEa = system().geometryManager().getGlobalEdge(edgeIdx);
          auto globalEb = system().geometryManager().getGlobalEdge(otherEdgeIdx);

          local.push_back({
              .x = system().x,
              .X = system().X,
              .globalEdgeA = {globalEa.x, globalEa.y},
              .globalEdgeB = {globalEb.x, globalEb.y},
              .type = ipc::EdgeEdgeDistanceType::Unknown,
          });
          local.back().updateDistanceType();
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return edgeTrajectoryBBox.overlap(bbox);
        });
  });

  // Sequential merge
  for (auto &local : threadLocalEE)
    for (auto &c : local)
      constraintSet.eeConstraints.push_back(c);
}

maths::BlockVector<3> IpcIntegrator::barrierEnergyGradient() const {
  maths::BlockVector<3> gradient(system().x.numBlocks());
  gradient.setZero();
  Real kappa = config.contactStiffness;
  for (const auto &c : constraintSet.vtConstraints)
    c.assembleBarrierGradient(barrier, gradient, kappa);
  for (const auto &c : constraintSet.eeConstraints)
    c.assembleMollifiedBarrierGradient(barrier, gradient, kappa);
  for (const auto &c : constraintSet.kinematicVTConstraints)
    c.assembleBarrierGradient(barrier, gradient, kappa);
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
  // 运动学约束
  for (const auto &c : constraintSet.kinematicVTConstraints)
    barrierEnergy += barrier(c.distanceSqr());
  return kappa * barrierEnergy;
}
} // namespace sim::fem