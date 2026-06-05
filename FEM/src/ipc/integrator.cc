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
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>


namespace sim::fem {

namespace {

bool isFiniteVec3(const glm::dvec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool isFiniteMat3(const glm::dmat3& value) {
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r)
      if (!std::isfinite(value[c][r]))
        return false;
  return true;
}

void ensureFiniteValue(std::string_view label, Real value) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(std::format("[IPC] {} is not finite: {}", label, value));
  }
}

void ensureFiniteAllBlocks(std::string_view label, const maths::BlockVector<3>& value) {
  for (int i = 0; i < value.numBlocks(); ++i) {
    const auto& block = value[i];
    if (!isFiniteVec3(block)) {
      throw std::runtime_error(std::format(
          "[IPC] {} has non-finite block {} = ({}, {}, {})",
          label, i, block.x, block.y, block.z));
    }
  }
}

template <std::size_t N>
void ensureFiniteTouchedBlocks(std::string_view label,
                               const maths::BlockVector<3>& value,
                               const std::array<int, N>& touchedBlocks) {
  for (int blockIdx : touchedBlocks) {
    if (blockIdx < 0 || blockIdx >= value.numBlocks()) {
      throw std::runtime_error(std::format(
          "[IPC] {} touched invalid block index {}", label, blockIdx));
    }
    const auto& block = value[blockIdx];
    if (!isFiniteVec3(block))
    {
      assert(false);
    }
  }
}

void ensureFiniteAllMatrixEntries(std::string_view label,
                                  const maths::BlockSparseMatrix<3>& value) {
  const auto& blocks = value.blocks();
  const auto& rowIndices = value.rowIndices();
  const auto& colIndices = value.colIndices();
  for (int entry = 0; entry < value.numEntries(); ++entry) {
    if (!isFiniteMat3(blocks[entry])) {
      throw std::runtime_error(std::format(
          "[IPC] {} has non-finite block entry {} at ({}, {})",
          label, entry, rowIndices[entry], colIndices[entry]));
    }
  }
}

void ensureFiniteNewMatrixEntries(std::string_view label,
                                  const maths::BlockSparseMatrix<3>& value,
                                  int firstNewEntry) {
  const auto& blocks = value.blocks();
  const auto& rowIndices = value.rowIndices();
  const auto& colIndices = value.colIndices();
  for (int entry = firstNewEntry; entry < value.numEntries(); ++entry) {
    if (!isFiniteMat3(blocks[entry])) {
      throw std::runtime_error(std::format(
          "[IPC] {} produced non-finite block entry {} at ({}, {})",
          label, entry, rowIndices[entry], colIndices[entry]));
    }
  }
}

std::string toString(ipc::PointTriangleDistanceType type) {
  switch (type) {
    case ipc::PointTriangleDistanceType::P_A: return "P_A";
    case ipc::PointTriangleDistanceType::P_B: return "P_B";
    case ipc::PointTriangleDistanceType::P_C: return "P_C";
    case ipc::PointTriangleDistanceType::P_AB: return "P_AB";
    case ipc::PointTriangleDistanceType::P_BC: return "P_BC";
    case ipc::PointTriangleDistanceType::P_CA: return "P_CA";
    case ipc::PointTriangleDistanceType::P_ABC: return "P_ABC";
    case ipc::PointTriangleDistanceType::Unknown: return "Unknown";
  }
  return "Unknown";
}

std::string toString(ipc::EdgeEdgeDistanceType type) {
  switch (type) {
    case ipc::EdgeEdgeDistanceType::A_C: return "A_C";
    case ipc::EdgeEdgeDistanceType::A_D: return "A_D";
    case ipc::EdgeEdgeDistanceType::B_C: return "B_C";
    case ipc::EdgeEdgeDistanceType::B_D: return "B_D";
    case ipc::EdgeEdgeDistanceType::AB_C: return "AB_C";
    case ipc::EdgeEdgeDistanceType::AB_D: return "AB_D";
    case ipc::EdgeEdgeDistanceType::A_CD: return "A_CD";
    case ipc::EdgeEdgeDistanceType::B_CD: return "B_CD";
    case ipc::EdgeEdgeDistanceType::AB_CD: return "AB_CD";
    case ipc::EdgeEdgeDistanceType::Unknown: return "Unknown";
  }
  return "Unknown";
}

std::string toString(ipc::ConstraintKind kind) {
  switch (kind) {
    case ipc::ConstraintKind::PP: return "PP";
    case ipc::ConstraintKind::PE: return "PE";
    case ipc::ConstraintKind::PT: return "PT";
    case ipc::ConstraintKind::EE: return "EE";
  }
  return "Unknown";
}

std::string describeConstraintPair(std::size_t idx,
                                   const ipc::ConstraintPair& c) {
  switch (c.type) {
    case ipc::ConstraintKind::PP:
      return std::format("pair[{}] type={} indices=({}, {})",
                         idx,
                         toString(c.type),
                         c.indices[0],
                         c.indices[1]);
    case ipc::ConstraintKind::PE:
      return std::format("pair[{}] type={} indices=({}, {}, {})",
                         idx,
                         toString(c.type),
                         c.indices[0],
                         c.indices[1],
                         c.indices[2]);
    case ipc::ConstraintKind::PT:
    case ipc::ConstraintKind::EE:
      return std::format("pair[{}] type={} indices=({}, {}, {}, {})",
                         idx,
                         toString(c.type),
                         c.indices[0],
                         c.indices[1],
                         c.indices[2],
                         c.indices[3]);
  }
  return std::format("pair[{}] type=Unknown", idx);
}

void ensureFiniteConstraintPairGradient(std::string_view label,
                                        const maths::BlockVector<3>& value,
                                        const ipc::ConstraintPair& pair) {
  switch (pair.type) {
    case ipc::ConstraintKind::PP: {
      const std::array<int, 2> touched = {pair.indices[0], pair.indices[1]};
      ensureFiniteTouchedBlocks(label, value, touched);
      return;
    }
    case ipc::ConstraintKind::PE: {
      const std::array<int, 3> touched = {pair.indices[0], pair.indices[1], pair.indices[2]};
      ensureFiniteTouchedBlocks(label, value, touched);
      return;
    }
    case ipc::ConstraintKind::PT:
    case ipc::ConstraintKind::EE: {
      const std::array<int, 4> touched = {
          pair.indices[0], pair.indices[1], pair.indices[2], pair.indices[3]};
      ensureFiniteTouchedBlocks(label, value, touched);
      return;
    }
  }
}



} // namespace

static int ipc_auto_reg = ([]() {
    IntegratorFactory::instance().registerCreator(
        "ipc",
        [](System &system, const core::JsonNode &json) {
          return IpcIntegrator::create(system, json);
        });
  }(), 0);

void IpcIntegrator::step(Real dt) {
  ensureFiniteValue("time step dt", dt);
  Real t_next = system().currentTime() + dt;
  system().advanceKinematicBodies(t_next);

  collisionDetector->setKinematicBodies(&system().colliders());

  system().constraints().enforcePosition(system(), t_next);
  maths::BlockVector<3> x_t = system().x;
  ensureFiniteAllBlocks("initial configuration x_t", x_t);
  x_prev = system().x;

  if (!hasInitializedActiveConstraints) {
    maths::BlockVector<3> zeroDirection(x_t.numBlocks());
    zeroDirection.setZero();
    precomputeCollisionPairs(zeroDirection, 0.0);
    refreshActiveConstraintPairs();
    hasInitializedActiveConstraints = true;
  }

  Real h = dt;
  spdlog::info("[IPC] computing initial energy...");

  Real E_prev = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
  ensureFiniteValue("initial barrier augmented energy", E_prev);
  spdlog::info("[IPC] E_prev = {}", E_prev);
  maths::BlockVector<3> p(x_t.numBlocks());
  int iter = 0;
  while (true) {
    // Compute negative gradient directly as BlockVector<3>
    spdlog::info("[IPC] iter {}: computing gradient...", iter);
    auto negG = barrierAugmentedIncrementalPotentialEnergyGradient(x_t, h);
    negG *= -1.0;

    system().constraints().zeroConstrainedGradient(negG);
    ensureFiniteAllBlocks(std::format("Newton rhs after constraints at iter {}", iter), negG);

    // Compute Hessian directly as BlockSparseMatrix<3>
    spdlog::info("[IPC] iter {}: computing Hessian...", iter);
    auto H = spdProjectHessian(h);
    ensureFiniteAllMatrixEntries(std::format("Newton Hessian before solve at iter {}", iter), H);

    spdlog::info("[IPC] iter {}: solving linear system...", iter);
    p.setZero();
    spdlog::info("[IPC] nnz: {}", H.blocks().size());
    auto result = solver->solve(H, negG, p);
    ensureFiniteValue(std::format("BlockPCG residual at iter {}", iter), result.residualNorm);
    if (!result.converged)
      spdlog::warn("BlockPCG: {} iters, residual={}", result.iterations, result.residualNorm);
    else
      spdlog::info("BlockPCG: {} iters, residual={}", result.iterations, result.residualNorm);

    system().constraints().projectToFreeSpace(p);
    ensureFiniteAllBlocks(std::format("projected Newton direction at iter {}", iter), p);

    std::cout << "norm: " << p.infNorm() << " " << negG.infNorm() << std::endl;
    if (p.infNorm() <
        config.eps * system().meshLengthScale() * h)
      break;

    spdlog::info("[IPC] iter {}: computing step size upper bound...", iter);
    Real alphaElastic = computeStepSizeUpperBound(p);
    ensureFiniteValue(std::format("alphaElastic at iter {}", iter), alphaElastic);
    Real alphaKinematic = 1.0;
    if (!system().colliders().empty()) {
      if (auto toiColliders = collisionDetector->detectDeformableVsKinematic(p, dt)) alphaKinematic = *toiColliders;
    }
    ensureFiniteValue(std::format("alphaKinematic at iter {}", iter), alphaKinematic);
    Real alpha = config.stepSizeScale * std::min({1.0, alphaElastic, alphaKinematic});
    ensureFiniteValue(std::format("alpha at iter {}", iter), alpha);
    spdlog::info("[IPC] iter {}: alpha={}", iter, alpha);

    if (alpha == 0.0)
      throw std::runtime_error(
          "Invalid state: collision happened within an integration step");
    precomputeCollisionPairs(p, alpha);
    Real E;
    do {
      maths::BlockVector<3> x_candidate = x_prev;
      x_candidate.axpy(alpha, p);
      ensureFiniteAllBlocks(std::format("line-search candidate x at iter {}", iter), x_candidate);
      updateCandidateSolution(x_candidate);
      system().constraints().enforcePosition(system(), t_next);
      alpha = alpha * 0.5;
      ensureFiniteValue(std::format("backtracked alpha at iter {}", iter), alpha);
      E = barrierAugmentedIncrementalPotentialEnergy(x_t, h);
      ensureFiniteValue(std::format("line-search energy at iter {}", iter), E);
    } while (E > E_prev);

    iter++;
  }
  velocityUpdate(x_t, h);

  system().constraints().enforceVelocity(system(), t_next);

  system().advanceTime(dt);

  Real T = system().kineticEnergy();
  Real V = system().potentialEnergy();
  Real Vg = system().gravitationalPotentialEnergy();
  Real total = T + V + Vg;
  ensureFiniteValue("kinetic energy", T);
  ensureFiniteValue("elastic potential energy", V);
  ensureFiniteValue("gravity potential energy", Vg);
  ensureFiniteValue("total energy", total);
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
  ensureFiniteValue("Hessian scale h", h);
  int nBlocks = system().x.numBlocks();
  maths::BlockSparseMatrix<3> H(nBlocks, nBlocks);
  H.setSymmetric(true);

  // Elastic Hessian
  system().spdProjectHessian(H);
  ensureFiniteAllMatrixEntries("elastic Hessian assembly", H);

  Real kappa = config.contactStiffness;
  ensureFiniteValue("contact stiffness", kappa);
  const int activeConstraintPairCount = constraintPairs.typeOffsets.back();
  for (int i = 0; i < activeConstraintPairCount; ++i) {
    const auto& pair = constraintPairs.pairs[i];
    const auto label = describeConstraintPair(i, pair);
    const int prevEntries = H.numEntries();
    try {
      ipc::constraintPairBarrierHessian(pair, system().x, system().X, H, barrier_, kappa);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::format(
          "[IPC] Hessian assembly failed for {}: {}", label, e.what()));
    }
    ensureFiniteNewMatrixEntries(std::format("Hessian assembly for {}", label), H, prevEntries);
  }



  // H_total = h² * H_elastic_barrier + M
  H.scale(h * h);
  H.addFrom(system().blockMass());
  ensureFiniteAllMatrixEntries("final Newton Hessian", H);

  return H;
}

void IpcIntegrator::refreshActiveConstraintPairs() {
  for (auto &c : collisionPairs.vtPairs)
    c.updateDistanceState();
  for (auto &c : collisionPairs.eePairs)
    c.updateDistanceState();
  for (auto &c : collisionPairs.colliderVTPairs)
    c.updateDistanceState();
  
  int ppCount = 0, peCount = 0, ptCount = 0, eeCount = 0;
  int colliderPpCount = 0, colliderPeCount = 0, colliderPtCount = 0;
  
  Real dHatSqr = barrier_.dHatSqr();
  
  for (const auto& cp : collisionPairs.vtPairs) {
    if (cp.isActive(dHatSqr)) {
      switch (cp.type) {
        case ipc::PointTriangleDistanceType::P_A:
        case ipc::PointTriangleDistanceType::P_B:
        case ipc::PointTriangleDistanceType::P_C:
          ppCount++; break;
        case ipc::PointTriangleDistanceType::P_AB:
        case ipc::PointTriangleDistanceType::P_BC:
        case ipc::PointTriangleDistanceType::P_CA:
          peCount++; break;
        case ipc::PointTriangleDistanceType::P_ABC:
          ptCount++; break;
        default: break;
      }
    }
  }
  
  for (const auto& cp : collisionPairs.eePairs) {
    if (cp.isActive(dHatSqr)) {
      switch (cp.type) {
        case ipc::EdgeEdgeDistanceType::A_C:
        case ipc::EdgeEdgeDistanceType::A_D:
        case ipc::EdgeEdgeDistanceType::B_C:
        case ipc::EdgeEdgeDistanceType::B_D:
          ppCount++; break;
        case ipc::EdgeEdgeDistanceType::AB_C:
        case ipc::EdgeEdgeDistanceType::AB_D:
        case ipc::EdgeEdgeDistanceType::A_CD:
        case ipc::EdgeEdgeDistanceType::B_CD:
          peCount++; break;
        case ipc::EdgeEdgeDistanceType::AB_CD:
          eeCount++; break;
        default: break;
      }
    }
  }
  
  for (const auto& cp : collisionPairs.colliderVTPairs) {
    if (cp.isActive(dHatSqr)) {
      switch (cp.type) {
        case ipc::PointTriangleDistanceType::P_A:
        case ipc::PointTriangleDistanceType::P_B:
        case ipc::PointTriangleDistanceType::P_C:
          colliderPpCount++; break;
        case ipc::PointTriangleDistanceType::P_AB:
        case ipc::PointTriangleDistanceType::P_BC:
        case ipc::PointTriangleDistanceType::P_CA:
          colliderPeCount++; break;
        case ipc::PointTriangleDistanceType::P_ABC:
          colliderPtCount++; break;
        default: break;
      }
    }
  }
  
  int totalCount = ppCount + peCount + ptCount + eeCount;
  int totalColliderCount = colliderPpCount + colliderPeCount + colliderPtCount;
  
  if (constraintPairs.pairs.size() < totalCount)
    constraintPairs.pairs.resize(totalCount);
  if (constraintPairs.colliderPairs.size() < totalColliderCount)
    constraintPairs.colliderPairs.resize(totalColliderCount);
  
  int ppIdx = 0, peIdx = ppCount, ptIdx = ppCount + peCount, eeIdx = ppCount + peCount + ptCount;
  int colliderPpIdx = 0, colliderPeIdx = colliderPpCount, colliderPtIdx = colliderPpCount + colliderPeCount;
  
  for (const auto& cp : collisionPairs.vtPairs) {
    if (!cp.isActive(dHatSqr)) continue;
    
    ipc::ConstraintPair c;
    switch (cp.type) {
      case ipc::PointTriangleDistanceType::P_A:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[0];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_B:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[1];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_C:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[2];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_AB:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[0];
        c.indices[2] = cp.globalTriVerts[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_BC:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[1];
        c.indices[2] = cp.globalTriVerts[2];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_CA:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[2];
        c.indices[2] = cp.globalTriVerts[0];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_ABC:
        c.type = ipc::ConstraintKind::PT;
        c.indices[0] = cp.globalVertex;
        c.indices[1] = cp.globalTriVerts[0];
        c.indices[2] = cp.globalTriVerts[1];
        c.indices[3] = cp.globalTriVerts[2];
        constraintPairs.pairs[ptIdx++] = c;
        break;
      default: break;
    }
  }
  
  for (const auto& cp : collisionPairs.eePairs) {
    if (!cp.isActive(dHatSqr)) continue;
    
    ipc::ConstraintPair c;
    switch (cp.type) {
      case ipc::EdgeEdgeDistanceType::A_C:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalEdgeA[0];
        c.indices[1] = cp.globalEdgeB[0];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::A_D:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalEdgeA[0];
        c.indices[1] = cp.globalEdgeB[1];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::B_C:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalEdgeA[1];
        c.indices[1] = cp.globalEdgeB[0];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::B_D:
        c.type = ipc::ConstraintKind::PP;
        c.indices[0] = cp.globalEdgeA[1];
        c.indices[1] = cp.globalEdgeB[1];
        constraintPairs.pairs[ppIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::AB_C:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalEdgeB[0];
        c.indices[1] = cp.globalEdgeA[0];
        c.indices[2] = cp.globalEdgeA[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::AB_D:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalEdgeB[1];
        c.indices[1] = cp.globalEdgeA[0];
        c.indices[2] = cp.globalEdgeA[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::A_CD:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalEdgeA[0];
        c.indices[1] = cp.globalEdgeB[0];
        c.indices[2] = cp.globalEdgeB[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::B_CD:
        c.type = ipc::ConstraintKind::PE;
        c.indices[0] = cp.globalEdgeA[1];
        c.indices[1] = cp.globalEdgeB[0];
        c.indices[2] = cp.globalEdgeB[1];
        constraintPairs.pairs[peIdx++] = c;
        break;
      case ipc::EdgeEdgeDistanceType::AB_CD:
        c.type = ipc::ConstraintKind::EE;
        c.indices[0] = cp.globalEdgeA[0];
        c.indices[1] = cp.globalEdgeA[1];
        c.indices[2] = cp.globalEdgeB[0];
        c.indices[3] = cp.globalEdgeB[1];
        constraintPairs.pairs[eeIdx++] = c;
        break;
      default: break;
    }
    
    // 同时添加到旧的约束集（用于 barrier 计算）
    // TODO Phase 3: Remove old constraint set completely
    // constraintSet.eeConstraints.push_back({
    //     .x = system().x,
    //     .X = system().X,
    //     .globalEdgeA = cp.globalEdgeA,
    //     .globalEdgeB = cp.globalEdgeB,
    //     .type = cp.type,
    // });
  }
  
  for (const auto& cp : collisionPairs.colliderVTPairs) {
    if (!cp.isActive(dHatSqr)) continue;
    
    ipc::ColliderConstraintPair c;
    c.writableIndices[0] = cp.deformableVertex;
    
    switch (cp.type) {
      case ipc::PointTriangleDistanceType::P_A:
        c.type = ipc::ConstraintKind::PP;
        c.colliderIndices[0] = 0;
        c.colliderIndices[1] = -1;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPpIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_B:
        c.type = ipc::ConstraintKind::PP;
        c.colliderIndices[0] = 1;
        c.colliderIndices[1] = -1;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPpIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_C:
        c.type = ipc::ConstraintKind::PP;
        c.colliderIndices[0] = 2;
        c.colliderIndices[1] = -1;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPpIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_AB:
        c.type = ipc::ConstraintKind::PE;
        c.colliderIndices[0] = 0;
        c.colliderIndices[1] = 1;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPeIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_BC:
        c.type = ipc::ConstraintKind::PE;
        c.colliderIndices[0] = 1;
        c.colliderIndices[1] = 2;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPeIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_CA:
        c.type = ipc::ConstraintKind::PE;
        c.colliderIndices[0] = 2;
        c.colliderIndices[1] = 0;
        c.colliderIndices[2] = -1;
        constraintPairs.colliderPairs[colliderPeIdx++] = c;
        break;
      case ipc::PointTriangleDistanceType::P_ABC:
        c.type = ipc::ConstraintKind::PT;
        c.colliderIndices[0] = 0;
        c.colliderIndices[1] = 1;
        c.colliderIndices[2] = 2;
        constraintPairs.colliderPairs[colliderPtIdx++] = c;
        break;
      default: break;
    }
    
  }
  
  // 更新 type offsets
  constraintPairs.typeOffsets = {0, ppCount, ppCount + peCount, ppCount + peCount + ptCount, totalCount};
  constraintPairs.colliderTypeOffsets = {0, colliderPpCount, colliderPpCount + colliderPeCount, totalColliderCount};
  
 // constraintPairs.pairs.resize(totalCount);
 // constraintPairs.colliderPairs.resize(totalColliderCount);
}

void IpcIntegrator::precomputeCollisionPairs(const maths::BlockVector<3>& p, Real alpha) {
  collisionPairs.clear();
  
  collisionDetector->updateBVHs(p, alpha);
  computeVertexTriangleCollisionPairs(p, alpha);
  computeEdgeEdgeCollisionPairs(p, alpha);
  
}

void IpcIntegrator::computeVertexTriangleCollisionPairs(const maths::BlockVector<3>& p, Real alpha) {
  Real dHat = config.dHat;
  const int nVerts = system().numVertices();

  tbb::enumerable_thread_specific<std::vector<ipc::VertexTriangleCollisionPair>> threadLocalVT;

  tbb::parallel_for(0, nVerts, [&](int vertexIdx) {
    auto vertexTrajectoryBBox =
        system().geometryManager()
            .getTrajectoryAccessor(system().x, p, alpha)
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
          local.back().updateDistanceState();
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return vertexTrajectoryBBox.overlap(bbox);
        });
  });

  for (auto &local : threadLocalVT)
    for (auto &c : local)
      collisionPairs.vtPairs.push_back(c);
}

void IpcIntegrator::computeEdgeEdgeCollisionPairs(const maths::BlockVector<3>& p, Real alpha) {
  Real dHat = config.dHat;
  const int nEdges = system().numEdges();

  tbb::enumerable_thread_specific<std::vector<ipc::EdgeEdgeCollisionPair>> threadLocalEE;

  tbb::parallel_for(0, nEdges, [&](int edgeIdx) {
    auto edgeTrajectoryBBox =
        system().geometryManager()
            .getTrajectoryAccessor(system().x, p, alpha)
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
          local.back().updateDistanceState();
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return edgeTrajectoryBBox.overlap(bbox);
        });
  });

  for (auto &local : threadLocalEE)
    for (auto &c : local)
      collisionPairs.eePairs.push_back(c);
}

maths::BlockVector<3> IpcIntegrator::barrierEnergyGradient() const {
  maths::BlockVector<3> gradient(system().x.numBlocks());
  gradient.setZero();
  Real kappa = config.contactStiffness;
  ensureFiniteValue("contact stiffness", kappa);

  const int activeConstraintPairCount = constraintPairs.typeOffsets.back();
  for (int i = 0; i < activeConstraintPairCount; ++i) {
    const auto& pair = constraintPairs.pairs[i];
    const auto label = describeConstraintPair(i, pair);
    try {
      ipc::constraintPairBarrierGradient(pair, system().x, system().X, gradient, barrier_, kappa);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::format(
          "[IPC] Gradient assembly failed for {}: {}", label, e.what()));
    }
    ensureFiniteConstraintPairGradient(std::format("Gradient assembly for {}", label), gradient, pair);
  }

  ensureFiniteAllBlocks("full barrier gradient", gradient);
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
  Real energy = 0.0;
  Real kappa = config.contactStiffness;
  ensureFiniteValue("contact stiffness", kappa);

  const int activeConstraintPairCount = constraintPairs.typeOffsets.back();
  for (int i = 0; i < activeConstraintPairCount; ++i) {
    const auto& pair = constraintPairs.pairs[i];
    const auto label = describeConstraintPair(i, pair);
    Real localEnergy;
    try {
      localEnergy = ipc::constraintPairBarrierEnergy(pair, system().x, system().X, barrier_, kappa);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::format(
          "[IPC] Barrier energy failed for {}: {}", label, e.what()));
    }
    ensureFiniteValue(std::format("barrier energy for {}", label), localEnergy);
    energy += localEnergy;
    ensureFiniteValue("accumulated barrier energy", energy);
  }

  return energy;
}

} // namespace sim::fem
