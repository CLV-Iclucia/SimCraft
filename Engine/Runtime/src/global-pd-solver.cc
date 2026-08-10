#include <Runtime/global-pd-solver.h>

#include <Core/profiler.h>
#include <Runtime/types.h>
#include <spdlog/spdlog.h>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ksk::runtime {
namespace {

DofBuffer createDofBufferLike(const DofBuffer& reference, int scalarCount)
{
  if (reference.isCPU()) {
    return DofBuffer::CPU(scalarCount);
  }
  return DofBuffer::GPU(*reference.device(), scalarCount);
}

#define KSK_RECORD_FRAME_STAT(stats, key, value) \
  do {                                           \
    if ((stats) != nullptr) {                    \
      (stats)->frameSet((key), (value));         \
    }                                            \
  } while (false)

#define KSK_ADD_FRAME_STAT(stats, key, value) \
  do {                                        \
    if ((stats) != nullptr) {                 \
      (stats)->frameAdd((key), (value));      \
    }                                         \
  } while (false)

#define KSK_AVERAGE_FRAME_STAT(stats, key, value) \
  do {                                            \
    if ((stats) != nullptr) {                     \
      (stats)->frameAverage((key), (value));      \
    }                                             \
  } while (false)

}  // namespace

struct GlobalProjectiveDynamicsSolver::Impl {
  std::vector<std::unique_ptr<ProjectiveDynamicsSubsystemSolver>>
      subsystem_apis_;
  int scalar_count_ = 0;
  int geometry_point_count_ = 0;

  // Constant global matrix A = M + h^2 L and its Cholesky factorization.
  // The design keeps A constant across PD iterations and (topology unchanged)
  // across frames, so the factorization is cached and reused.
  Eigen::SparseMatrix<double> matrix_;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factorized_;
  bool matrix_ready_ = false;
  double cached_dt_ = -1.0;

  void rebuildGlobalMatrix(SimulationContext& simulation, double dt);
  void solveProjectiveSystem(const DofBuffer& rhs, DofBuffer& deltaQ);
};

GlobalProjectiveDynamicsSolver::GlobalProjectiveDynamicsSolver()
    : impl_(std::make_unique<Impl>())
{
}

GlobalProjectiveDynamicsSolver::~GlobalProjectiveDynamicsSolver() = default;

GlobalProjectiveDynamicsSolver::GlobalProjectiveDynamicsSolver(
    GlobalProjectiveDynamicsSolver&&) noexcept = default;

GlobalProjectiveDynamicsSolver&
GlobalProjectiveDynamicsSolver::operator=(GlobalProjectiveDynamicsSolver&&) noexcept =
    default;

int GlobalProjectiveDynamicsSolver::scalarCount() const noexcept
{
  return impl_->scalar_count_;
}

int GlobalProjectiveDynamicsSolver::geometryPointCount() const noexcept
{
  return impl_->geometry_point_count_;
}

void GlobalProjectiveDynamicsSolver::Impl::rebuildGlobalMatrix(
    SimulationContext& simulation, double dt)
{
  if (matrix_ready_ && cached_dt_ == dt) {
    return;
  }
  const int scalarCount = simulation.scene().dofs.totalScalars;
  std::vector<SparseMatrixEntry> entries;
  for (const auto& api : subsystem_apis_) {
    api->appendProjectiveMatrixEntries(entries);
  }

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(entries.size());
  for (const SparseMatrixEntry& entry : entries) {
    triplets.emplace_back(entry.row, entry.col, entry.value);
  }

  matrix_.resize(scalarCount, scalarCount);
  matrix_.setFromTriplets(triplets.begin(), triplets.end());
  factorized_.compute(matrix_);
  matrix_ready_ = (factorized_.info() == Eigen::Success);
  cached_dt_ = dt;

  if (!matrix_ready_) {
    spdlog::warn(
        "GlobalProjectiveDynamicsSolver: factorization of the constant "
        "projective matrix failed (scalars={}, nnz={})",
        scalarCount, matrix_.nonZeros());
  }
}

void GlobalProjectiveDynamicsSolver::Impl::solveProjectiveSystem(
    const DofBuffer& rhs, DofBuffer& deltaQ)
{
  if (!rhs.isCPU() || !deltaQ.isCPU()) {
    throw std::runtime_error(
        "GlobalProjectiveDynamicsSolver: GPU solve is not implemented");
  }
  deltaQ.cpu() = factorized_.solve(rhs.cpu());
}

void GlobalProjectiveDynamicsSolver::prepare(SimulationContext& simulation)
{
  KSK_PROFILE_SCOPE("GlobalPDSolver/Prepare");
  impl_->subsystem_apis_.clear();
  impl_->matrix_ready_ = false;
  impl_->cached_dt_ = -1.0;

  for (auto& subsystem : simulation.subsystems()) {
    auto api = subsystem->createProjectiveDynamicsSolver();
    if (!api) {
      throw std::runtime_error(
          "subsystem does not support ProjectiveDynamics solver");
    }
    impl_->subsystem_apis_.push_back(std::move(api));
  }

  impl_->scalar_count_ = simulation.scene().dofs.totalScalars;
  impl_->geometry_point_count_ =
      static_cast<int>(simulation.scene().geometry.pointCount());
}

RuntimeStepResult GlobalProjectiveDynamicsSolver::step(
    SimulationContext& simulation, double dt, double time,
    GlobalSolverStatsCollector* stats)
{
  KSK_PROFILE_SCOPE_COLOR("GlobalPDSolver/Step",
                          ksk::core::profiler_colors::kCyan);

  // Re-prepare if the subsystem list changed since the last step.
  if (impl_->subsystem_apis_.size() != simulation.subsystems().size()) {
    prepare(simulation);
  }

  const GlobalSolverConfig& config = *simulation.scene().solverConfig;
  const ProjectiveDynamicsSolverConfig* pd_config =
      RF::Cast<ProjectiveDynamicsSolverConfig>(&config);
  const int maxIterations =
      pd_config != nullptr ? pd_config->maxPdIterations : 50;
  const double residualTolerance =
      pd_config != nullptr ? pd_config->pdResidualTolerance : 1.0e-6;
  const bool enableCollision =
      pd_config != nullptr ? pd_config->enablePdCollision : false;
  const int maxResolveIterations =
      pd_config != nullptr ? pd_config->maxCollisionResolveIterations : 5;

  RuntimeStepResult result;
  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  auto& subsystems = simulation.subsystems();

  if (stats != nullptr) {
    stats->beginFrame();
  }
  KSK_RECORD_FRAME_STAT(stats, "step.time", time);

  // --- begin step: common subsystem lifecycle (solver-independent) ---
  {
    KSK_PROFILE_SCOPE("GlobalPDSolver/Step/BeginSubsystemSteps");
    for (const auto& subsystem : subsystems) {
      const DofRange range = subsystem->dofRange();
      subsystem->readState(q.slice(range), qdot.slice(range));
      subsystem->beginStep(q.slice(range), qdot.slice(range), dt);
      subsystem->updateGeometry(simulation.scene().geometry);
    }
  }

  // --- per-subsystem elastic operator setup (once per step) ---
  {
    KSK_PROFILE_SCOPE("GlobalPDSolver/Step/SetupProjectiveOperators");
    for (const auto& api : impl_->subsystem_apis_) {
      api->updateProjectiveConstraints(time + dt, dt);
      api->prepareProjectiveOperator(dt);
    }
  }

  // Constant matrix M + h^2 L: assemble + factor once, reuse across iterations.
  {
    KSK_PROFILE_SCOPE("GlobalPDSolver/Step/RebuildGlobalMatrix");
    impl_->rebuildGlobalMatrix(simulation, dt);
  }

  // ========================================================================
  // Phase 1: PD L-G iterations (no collision)
  // ========================================================================
  const int scalarCount = simulation.scene().dofs.totalScalars;
  for (int iteration = 0; iteration < maxIterations; ++iteration) {
    KSK_PROFILE_SCOPE("GlobalPDSolver/Step/PdIteration");

    // Local step.
    {
      KSK_PROFILE_SCOPE("GlobalPDSolver/Step/PdIteration/ProjectLocalStep");
      for (const auto& api : impl_->subsystem_apis_) {
        const DofRange range = api->dofRange();
        api->projectLocalStep(q.slice(range), qdot.slice(range), dt);
      }
    }

    // Global step right-hand side.
    DofBuffer rhs = createDofBufferLike(q, scalarCount);
    rhs.setZero();
    {
      KSK_PROFILE_SCOPE("GlobalPDSolver/Step/PdIteration/AssembleRhs");
      for (const auto& api : impl_->subsystem_apis_) {
        api->assembleProjectiveRhs(rhs.slice(api->dofRange()));
      }
    }

    // deltaQ = (M + h^2 L)^-1 * rhs
    DofBuffer deltaQ = createDofBufferLike(q, scalarCount);
    {
      KSK_PROFILE_SCOPE("GlobalPDSolver/Step/PdIteration/SolveGlobalSystem");
      impl_->solveProjectiveSystem(rhs, deltaQ);
    }
    q.addScaled(1.0, deltaQ);

    {
      KSK_PROFILE_SCOPE("GlobalPDSolver/Step/PdIteration/SyncGeometry");
      for (const auto& subsystem : subsystems) {
        const DofRange range = subsystem->dofRange();
        subsystem->readState(q.slice(range), qdot.slice(range));
        subsystem->updateGeometry(simulation.scene().geometry);
      }
    }

    result.iterations = iteration + 1;
    {
      KSK_PROFILE_SCOPE("GlobalPDSolver/Step/PdIteration/ResidualCheck");
      double residual = 0.0;
      for (const auto& api : impl_->subsystem_apis_) {
        residual += api->evaluateProjectiveResidual(
            q.slice(api->dofRange()), dt);
      }
      result.finalGradientNorm = residual;
      KSK_AVERAGE_FRAME_STAT(stats, "pd.residual.average", residual);
      if (std::isfinite(residual) && residual <= residualTolerance) {
        result.converged = true;
        result.terminationReason = "pd_residual_tolerance";
        KSK_ADD_FRAME_STAT(stats, "pd.converged.count", 1);
        break;
      }
    }
  }

  // ========================================================================
  // Phase 2: post-processing collision resolution (position corrections on q)
  // ========================================================================
  if (enableCollision) {
    KSK_PROFILE_SCOPE("GlobalPDSolver/Step/ResolveCollisions");
    // TODO: collect pinned (V_pin) vertices from the scene constraints. The
    // collision pipeline is fully wired; topology/pin data can be gathered per
    // subsystem when PD collision is enabled.
    const std::unordered_set<PointIdx> pinnedVertices;
    resolvePdCollisions(simulation, pinnedVertices, maxResolveIterations);
  }

  // --- accept step ---
  {
    KSK_PROFILE_SCOPE("GlobalPDSolver/Step/AcceptSubsystemSteps");
    for (const auto& subsystem : subsystems) {
      const DofRange range = subsystem->dofRange();
      subsystem->readState(q.slice(range), qdot.slice(range));
      subsystem->acceptStep(q.slice(range), qdot.slice(range), dt);
      subsystem->updateGeometry(simulation.scene().geometry);
    }
  }

  if (!result.converged) {
    result.terminationReason =
        result.terminationReason.empty() ? "max_pd_iterations"
                                         : result.terminationReason;
  }

  return result;
}

}  // namespace ksk::runtime
