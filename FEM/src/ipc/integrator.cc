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
  // ===== 运动学体: 推进到下一时刻 =====
  Real t_next = system().currentTime() + dt;
  system().advanceKinematicBodies(t_next);

  // ===== 设置 collision detector 的运动学体 =====
  collisionDetector->setKinematicBodies(&system().kinematicBodies());

  x_prev = system().x;
  maths::BlockVector<3> x_t = system().x;

  // ===== enforce 约束到目标位置 =====
  system().constraints().enforcePosition(system().x, t_next);

  Real h = dt;
  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
  maths::BlockVector<3> p(x_t.numBlocks());
  int iter = 0;
  while (true) {
    // Compute negative gradient directly as BlockVector<3>
    auto negG = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
    negG *= Real(-1);

    // ===== 投影梯度（清零被约束分量）=====
    system().constraints().zeroConstrainedGradient(negG);

    // Compute Hessian directly as BlockSparseMatrix<3>
    auto H_block = spdProjectHessian(h);

    p.setZero();
    auto result = solver->solve(H_block, negG, p);
    if (!result.converged)
      spdlog::warn("BlockPCG: {} iters, residual={}", result.iterations, result.residualNorm);

    // ===== 投影搜索方向 =====
    system().constraints().projectToFreeSpace(p);

    if (p.infNorm() <
        config.eps * system().meshLengthScale() * h)
      break;

    // ===== 计算步长上限 (弹性体 + 运动学体) =====
    Real alphaElastic = computeStepSizeUpperBound(p);
    Real alphaKinematic = 1.0;
    if (!system().kinematicBodies().empty()) {
      auto toiKin = collisionDetector->detectDeformableVsKinematic(p, dt);
      if (toiKin) alphaKinematic = *toiKin;
    }
    Real alpha = config.stepSizeScale * std::min({1.0, alphaElastic, alphaKinematic});

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
      // ===== line search 后再次 enforce =====
      system().constraints().enforcePosition(system().x, t_next);
      alpha = alpha * 0.5;
      E = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
    } while (E > E_prev);
    iter++;
    x_prev = system().x;
    E_prev = E;
  }
  velocityUpdate(x_t, h);

  // ===== enforce 速度约束 =====
  system().constraints().enforceVelocity(system().xdot, t_next);

  // ===== 推进时间 =====
  system().advanceTime(dt);
}

maths::BlockSparseMatrix<3> IpcIntegrator::spdProjectHessian(Real h) {
  int nBlocks = system().x.numBlocks();
  maths::BlockSparseMatrix<3> H(nBlocks, nBlocks);

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
    
    // 使用运动学体的 BVH 进行空间查询
    // 为简化，先遍历所有顶点
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

  tbb::parallel_for(0, system().numVertices(), [&](int vertexIdx) {
    auto vertexTrajectoryBBox =
        system()
            .geometryManager()
            .getTrajectoryAccessor(system().x, p, toi)
            .vertexBBox(vertexIdx);
    vertexTrajectoryBBox = vertexTrajectoryBBox.dilate(dHat);
    collisionDetector->trianglesBVH().runSpatialQuery(
        [&](int triangleIdx) -> bool {
          if (system().triangleContainsVertex(triangleIdx, vertexIdx))
            return false;

          auto globalTri = system().geometryManager().getGlobalTriangle(triangleIdx);

          constraintSet.vtConstraints.push_back({
              .x = system().x,
              .globalVertex = vertexIdx,
              .globalTriVerts = {globalTri.x, globalTri.y, globalTri.z},
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

          auto globalEa = system().geometryManager().getGlobalEdge(edgeIdx);
          auto globalEb = system().geometryManager().getGlobalEdge(otherEdgeIdx);

          constraintSet.eeConstraints.push_back({
              .x = system().x,
              .X = system().X,
              .globalEdgeA = {globalEa.x, globalEa.y},
              .globalEdgeB = {globalEb.x, globalEb.y},
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

maths::BlockVector<3> IpcIntegrator::barrierEnergyGradient() const {
  maths::BlockVector<3> gradient(system().x.numBlocks());
  gradient.setZero();
  Real kappa = config.contactStiffness;
  for (const auto &c : constraintSet.vtConstraints)
    c.assembleBarrierGradient(barrier, gradient, kappa);
  for (const auto &c : constraintSet.eeConstraints)
    c.assembleMollifiedBarrierGradient(barrier, gradient, kappa);
  // 运动学约束梯度
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