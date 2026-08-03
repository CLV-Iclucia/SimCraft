#include <Core/profiler.h>
#include <Eigen/IterativeLinearSolvers>
#include <Runtime/contact-barrier-energy.h>
#include <Runtime/global-solver.h>
#include <spdlog/spdlog.h>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ksk::runtime {
namespace {

#ifndef NDEBUG
void validateSimulationStateForStep(const SimulationContext& simulation) {
  const RuntimeScene& scene = simulation.scene();
  const int scalar_count = scene.dofs.totalScalars;
  const int q_count = simulation.q().scalarCount();
  const int qdot_count = simulation.qdot().scalarCount();

  if (simulation.subsystems().empty()) {
    throw std::runtime_error(
        "GlobalGaussNewtonSolver::step received a SimulationContext with no "
        "subsystems. Build it with buildSimulation/buildSimulationRunner "
        "from a populated RuntimeSceneDesc before stepping.");
  }
  if (scalar_count <= 0) {
    throw std::runtime_error(
        "GlobalGaussNewtonSolver::step received a SimulationContext with zero "
        "DOFs. Check that scene objects were assigned to subsystems before "
        "buildSimulation/buildSimulationRunner.");
  }
  if (q_count != scalar_count || qdot_count != scalar_count) {
    throw std::runtime_error(
        "GlobalGaussNewtonSolver::step received inconsistent DOF buffers: "
        "scene.dofs.totalScalars=" +
        std::to_string(scalar_count) +
        ", q.scalarCount=" + std::to_string(q_count) +
        ", qdot.scalarCount=" + std::to_string(qdot_count) + ".");
  }
}
#endif

GeometryBuffer createGeometryDirectionBuffer(const DofBuffer& direction,
                                             int point_count) {
  if (direction.isCPU()) return GeometryBuffer::CPU(point_count);
  return GeometryBuffer::GPU(*direction.device(), point_count);
}

DofBuffer createDofBufferLike(const DofBuffer& reference, int scalar_count) {
  if (reference.isCPU()) {
    return DofBuffer::CPU(scalar_count);
  }
  return DofBuffer::GPU(*reference.device(), scalar_count);
}

double computeSceneScale(const GlobalGeometryManager& geometry) {
  if (geometry.points.empty()) {
    return 1.0;
  }

  glm::dvec3 lower(std::numeric_limits<double>::max());
  glm::dvec3 upper(std::numeric_limits<double>::lowest());
  for (int i = 0; i < geometry.points.size(); i++) {
    const auto& point = geometry.points[i];
    lower = glm::min(lower, point.x);
    upper = glm::max(upper, point.x);
  }

  const double scale = glm::length(upper - lower);
  if (std::isfinite(scale) && scale > 0.0) {
    return scale;
  }
  return 1.0;
}

double computeDofInfNorm(const DofBuffer& buffer) {
  if (buffer.isCPU()) {
    return buffer.cpu().cwiseAbs().maxCoeff();
  }
  throw std::runtime_error("GPU DofBuffer infNorm is not implemented");
}

double computeDofViewDot(ConstDofView a, ConstDofView b) {
  if (!a.isCPU() || !b.isCPU()) {
    throw std::runtime_error("GPU DofView dot is not implemented");
  }
  if (a.scalarCount() != b.scalarCount()) {
    throw std::invalid_argument("DofView dot requires matching scalar counts");
  }
  double dot = 0.0;
  for (int i = 0; i < a.scalarCount(); ++i) {
    dot += a[i] * b[i];
  }
  return dot;
}

double computeDofViewNorm(ConstDofView view) {
  const double squared_norm = computeDofViewDot(view, view);
  if (!std::isfinite(squared_norm) || squared_norm < 0.0) {
    return squared_norm;
  }
  return std::sqrt(squared_norm);
}

double computeDofViewInfNorm(ConstDofView view) {
  if (!view.isCPU()) {
    throw std::runtime_error("GPU DofView infNorm is not implemented");
  }
  double norm = 0.0;
  for (int i = 0; i < view.scalarCount(); ++i) {
    norm = std::max(norm, std::abs(view[i]));
  }
  return norm;
}

const char* solveModeName(GlobalSolveMode mode) noexcept {
  switch (mode) {
    case GlobalSolveMode::Decoupled:
      return "decoupled";
    case GlobalSolveMode::Coupled:
      return "coupled";
  }
  return "<unknown>";
}

std::string subsystemTypeName(const RuntimeScene& scene,
                              SubsystemIdx subsystem) {
  for (const SubsystemBatch& batch : scene.subsystemBatches) {
    if (subsystem >= batch.firstSubsystem &&
        subsystem < batch.firstSubsystem + batch.subsystemCount) {
      return batch.type;
    }
  }
  return {};
}

struct ContactBreakdown {
  int total = 0;
  std::array<int, 4> count{};
  std::array<double, 4> energy{};
  double maxThickness = 0.0;
  double maxStiffness = 0.0;
  double minDhat = 0.0;
};

struct ObjectiveBreakdown {
  double subsystem = 0.0;
  double contact = 0.0;
  double friction = 0.0;
  double total = 0.0;
  ContactBreakdown contacts;
};

int contactCaseIndex(ContactCase type) {
  return static_cast<int>(type);
}

ContactStencils singleContactStencil(const ContactStencil& contact) {
  ContactStencils stencils;
  stencils.push_back(contact);
  return stencils;
}

ContactBreakdown computeContactBreakdown(const RuntimeScene& scene,
                                         bool include_energy) {
  ContactBreakdown breakdown;
  breakdown.total = static_cast<int>(scene.contacts.size());
  if (scene.contacts.empty()) {
    breakdown.minDhat = 0.0;
    return breakdown;
  }

  bool first_contact = true;
  for (int contact_index = 0; contact_index < scene.contacts.size();
       ++contact_index) {
    const ContactStencil& contact =
        scene.contacts[static_cast<size_t>(contact_index)];
    const int index = contactCaseIndex(contact.type);
    breakdown.count[static_cast<size_t>(index)]++;
    breakdown.maxThickness =
        std::max(breakdown.maxThickness, contact.thickness);
    breakdown.maxStiffness =
        std::max(breakdown.maxStiffness, contact.stiffness);
    breakdown.minDhat =
        first_contact ? contact.dHat : std::min(breakdown.minDhat, contact.dHat);
    first_contact = false;
    double contact_energy = 0.0;
    if (include_energy) {
      contact_energy =
          computeContactEnergy(scene.geometry, singleContactStencil(contact));
      breakdown.energy[static_cast<size_t>(index)] += contact_energy;
    }
  }
  return breakdown;
}

void recordContactBreakdownStats(GlobalSolverStatsCollector* stats,
                                 std::string_view prefix,
                                 const ContactBreakdown& breakdown) {
  if (stats == nullptr) {
    return;
  }
  stats->frameSet(std::string(prefix) + ".contacts.total", breakdown.total);
  stats->frameSet(std::string(prefix) + ".contacts.pp",
                  breakdown.count[contactCaseIndex(ContactCase::PP)]);
  stats->frameSet(std::string(prefix) + ".contacts.pe",
                  breakdown.count[contactCaseIndex(ContactCase::PE)]);
  stats->frameSet(std::string(prefix) + ".contacts.pt",
                  breakdown.count[contactCaseIndex(ContactCase::PT)]);
  stats->frameSet(std::string(prefix) + ".contacts.ee",
                  breakdown.count[contactCaseIndex(ContactCase::EE)]);
  stats->frameSet(std::string(prefix) + ".contact_energy.pp",
                  breakdown.energy[contactCaseIndex(ContactCase::PP)]);
  stats->frameSet(std::string(prefix) + ".contact_energy.pe",
                  breakdown.energy[contactCaseIndex(ContactCase::PE)]);
  stats->frameSet(std::string(prefix) + ".contact_energy.pt",
                  breakdown.energy[contactCaseIndex(ContactCase::PT)]);
  stats->frameSet(std::string(prefix) + ".contact_energy.ee",
                  breakdown.energy[contactCaseIndex(ContactCase::EE)]);
  stats->frameSet(std::string(prefix) + ".contacts.max_thickness",
                  breakdown.maxThickness);
  stats->frameSet(std::string(prefix) + ".contacts.max_stiffness",
                  breakdown.maxStiffness);
  stats->frameSet(std::string(prefix) + ".contacts.min_dhat",
                  breakdown.minDhat);
}

ObjectiveBreakdown evaluateObjectiveBreakdown(
    SimulationContext& simulation, double dt, bool include_contact_breakdown);
ObjectiveBreakdown evaluateSubsystemObjectiveAndUpdateGeometry(
    SimulationContext& simulation, double dt);
void appendContactObjectiveBreakdown(SimulationContext& simulation,
                                     bool include_contact_breakdown,
                                     ObjectiveBreakdown& result);

void warnInvalidPreconditionerDotBreakdown(SimulationContext& simulation,
                                           const DofBuffer& residual,
                                           const DofBuffer& z,
                                           const char* phase, int pcg_iteration,
                                           double rz, double residual_norm,
                                           double rhs_norm) {
  spdlog::warn(
      "PCG preconditioner produced invalid rz: phase={} pcg_iteration={} "
      "rz={} residual_norm={} rhs_norm={} contacts={}",
      phase, pcg_iteration, rz, residual_norm, rhs_norm,
      simulation.scene().contacts.size());
  for (const auto& subsystem : simulation.subsystems()) {
    const DofRange range = subsystem->dofRange();
    const ConstDofView local_residual = residual.slice(range);
    const ConstDofView local_z = z.slice(range);
    const std::string type_name =
        subsystemTypeName(simulation.scene(), subsystem->id());
    spdlog::warn(
        "PCG preconditioner block: subsystem={} type={} scalar_offset={} "
        "scalar_count={} rz={} residual_norm={} z_norm={} "
        "residual_inf={} z_inf={}",
        subsystem->id(), type_name.empty() ? "<unknown>" : type_name,
        range.scalarOffset, range.scalarCount,
        computeDofViewDot(local_residual, local_z),
        computeDofViewNorm(local_residual), computeDofViewNorm(local_z),
        computeDofViewInfNorm(local_residual), computeDofViewInfNorm(local_z));
  }
}

void synchronizeGeometryFromState(SimulationContext& simulation) {
  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  for (const auto& subsystem : simulation.subsystems()) {
    const DofRange range = subsystem->dofRange();
    subsystem->readState(q.slice(range), qdot.slice(range));
    subsystem->updateGeometry(simulation.scene().geometry);
  }
}

void captureFrictionReferenceGeometry(SimulationContext& simulation) {
  RuntimeScene& scene = simulation.scene();
  scene.frictionReferencePositions.clear();

  if (!scene.solverConfig.enableFriction) {
    return;
  }

  scene.frictionReferencePositions.resize(
      static_cast<size_t>(scene.geometry.pointCount()));
  for (int point = 0; point < scene.geometry.pointCount(); ++point) {
    scene.frictionReferencePositions[point] =
        scene.geometry.worldPosition(PointIdx{point});
  }
}

void appendContactGeometryHessian(ContactGeometryHessian& target,
                                  ContactGeometryHessian&& source) {
  target.blocks.reserve(target.blocks.size() + source.blocks.size());
  for (ContactGeometryHessianBlock& block : source.blocks) {
    target.blocks.push_back(std::move(block));
  }
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

#define KSK_MAX_FRAME_STAT(stats, key, value) \
  do {                                        \
    if ((stats) != nullptr) {                 \
      (stats)->frameMax((key), (value));      \
    }                                         \
  } while (false)

}  // namespace

void GlobalSolverStatsCollector::beginFrame() {
  frames_.emplace_back();
  frame_average_counts_.emplace_back();
}

void GlobalSolverStatsCollector::clear() noexcept {
  global_.clear();
  global_average_counts_.clear();
  frames_.clear();
  frame_average_counts_.clear();
}

void GlobalSolverStatsCollector::frameSet(std::string key, double value) {
  currentFrame()[std::move(key)] = value;
}

void GlobalSolverStatsCollector::frameAdd(std::string key, double value) {
  add(currentFrame(), std::move(key), value);
}

void GlobalSolverStatsCollector::frameMax(std::string key, double value) {
  max(currentFrame(), std::move(key), value);
}

void GlobalSolverStatsCollector::frameMin(std::string key, double value) {
  min(currentFrame(), std::move(key), value);
}

void GlobalSolverStatsCollector::frameAverage(std::string key, double value) {
  average(currentFrame(), frame_average_counts_.back(), std::move(key), value);
}

void GlobalSolverStatsCollector::globalSet(std::string key, double value) {
  global_[std::move(key)] = value;
}

void GlobalSolverStatsCollector::globalAdd(std::string key, double value) {
  add(global_, std::move(key), value);
}

void GlobalSolverStatsCollector::globalMax(std::string key, double value) {
  max(global_, std::move(key), value);
}

void GlobalSolverStatsCollector::globalMin(std::string key, double value) {
  min(global_, std::move(key), value);
}

void GlobalSolverStatsCollector::globalAverage(std::string key, double value) {
  average(global_, global_average_counts_, std::move(key), value);
}

void GlobalSolverStatsCollector::add(StatMap& stats, std::string key,
                                     double value) {
  stats[std::move(key)] += value;
}

void GlobalSolverStatsCollector::max(StatMap& stats, std::string key,
                                     double value) {
  auto [it, inserted] = stats.try_emplace(std::move(key), value);
  if (!inserted) {
    it->second = std::max(it->second, value);
  }
}

void GlobalSolverStatsCollector::min(StatMap& stats, std::string key,
                                     double value) {
  auto [it, inserted] = stats.try_emplace(std::move(key), value);
  if (!inserted) {
    it->second = std::min(it->second, value);
  }
}

void GlobalSolverStatsCollector::average(StatMap& stats,
                                         std::map<std::string, int>& counts,
                                         std::string key, double value) {
  const std::string count_key = key;
  const int count = ++counts[count_key];
  auto [it, inserted] = stats.try_emplace(std::move(key), value);
  if (!inserted) {
    it->second += (value - it->second) / static_cast<double>(count);
  }
}

GlobalSolverStatsCollector::StatMap&
GlobalSolverStatsCollector::currentFrame() {
  if (frames_.empty()) {
    beginFrame();
  }
  return frames_.back();
}

void GlobalGaussNewtonSolver::prepare(const RuntimeScene& scene) {
  scalar_count_ = scene.dofs.totalScalars;
  geometry_point_count_ = static_cast<int>(scene.geometry.points.size());
}

RuntimeStepResult GlobalGaussNewtonSolver::step(
    SimulationContext& simulation, double dt, double time,
    GlobalSolverStatsCollector* stats) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/Step",
                          ksk::core::profiler_colors::kBlue);

#ifndef NDEBUG
  {
    KSK_PROFILE_SCOPE("GlobalSolver/Step/ValidateState");
    validateSimulationStateForStep(simulation);
  }
#endif

  {
    KSK_PROFILE_SCOPE("GlobalSolver/Step/Prepare");
    prepare(simulation.scene());
  }

  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  auto& subsystems = simulation.subsystems();

  {
    KSK_PROFILE_SCOPE("GlobalSolver/Step/BeginSubsystemSteps");
    for (const auto& subsystem : subsystems) {
      const DofRange range = subsystem->dofRange();
      subsystem->readState(q.slice(range), qdot.slice(range));
      subsystem->beginStep(q.slice(range), qdot.slice(range), dt);
      subsystem->updateGeometry(simulation.scene().geometry);
    }
    captureFrictionReferenceGeometry(simulation);
  }

  RuntimeStepResult result;
  const GlobalSolverConfig& config = simulation.scene().solverConfig;
  const int scalar_count = simulation.scene().dofs.totalScalars;

  if (stats) {
    stats->beginFrame();
  }

  KSK_RECORD_FRAME_STAT(stats, "step.time", time);

  if (!config.enableContact) {
    simulation.scene().contacts.clear();
  }

  result.terminationReason = "max_newton_iterations";
  std::optional<DofBuffer> previous_newton_direction;

  for (int iteration = 0; iteration < config.maxNewtonIterations; iteration++) {
    KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration");
    DofBuffer gradient;
    DofBuffer direction;

    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/CreateBuffers");
      gradient = createDofBufferLike(q, scalar_count);
      direction = createDofBufferLike(q, scalar_count);
    }

    if (config.enablePCGWarmStart && previous_newton_direction) {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/WarmStartDirection");
      direction.copyFrom(*previous_newton_direction);
      KSK_ADD_FRAME_STAT(stats, "pcg.warm_start.available.count", 1);
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.warm_start.direction_norm.average",
                             direction.norm());
    }

    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/UpdateSubsystemStates");
      for (const auto& subsystem : subsystems) {
        const DofRange range = subsystem->dofRange();
        subsystem->readState(q.slice(range), qdot.slice(range));
        subsystem->updateInternalConstraints(time + dt, dt);
        subsystem->updateGeometry(simulation.scene().geometry);
      }
    }

    if (config.enableContact) {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/RebuildActiveContacts");
      contact_detector_.rebuildActiveContacts(simulation);
    } else
      simulation.scene().contacts.clear();

    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/AssembleLocal");  // NOLINT(*-reserved-identifier)
      for (const auto& subsystem : subsystems) {
        const DofRange range = subsystem->dofRange();
        {
          KSK_PROFILE_SCOPE(
              "GlobalSolver/Step/NewtonIteration/PrepareLocalOperator");
          subsystem->prepareLocalOperator(dt);
        }
        {
          KSK_PROFILE_SCOPE(
              "GlobalSolver/Step/NewtonIteration/AssembleLocalGradient");
          subsystem->assembleLocalGradient(gradient.slice(range));
        }
      }
    }

    if (config.enableContact) {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/AssembleContactGradient");
      assembleContactGradient(simulation, gradient);
    }

    if (config.enableContact && config.enableFriction) {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/AssembleContactFrictionGradient");
      assembleContactFrictionGradient(simulation, gradient);
    }

    std::optional<ContactGeometryHessian> contact_hessian;
    if (config.enableContact && !simulation.scene().contacts.empty()) {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/AssembleContactHessian");
      contact_hessian = computeContactHessianWrtGeometry(
          simulation.scene().geometry, simulation.scene().contacts);
      if (config.enableFriction) {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/Step/NewtonIteration/AssembleContactFrictionHessian");
        appendContactGeometryHessian(
            *contact_hessian,
            computeContactFrictionHessianWrtGeometry(
                simulation.scene().geometry, simulation.scene().contacts));
      }
    }

    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/GradientConvergenceCheck");
      result.finalGradientNorm = gradient.norm();
      result.iterations = iteration + 1;
      KSK_AVERAGE_FRAME_STAT(stats, "newton.gradient_norm.average",
                             result.finalGradientNorm);
      KSK_MAX_FRAME_STAT(stats, "newton.gradient_norm.max",
                         result.finalGradientNorm);
    }
    if (result.finalGradientNorm <= config.newtonGradientTolerance) {
      result.converged = true;
      result.terminationReason = "gradient_tolerance";
      KSK_ADD_FRAME_STAT(stats, "newton.converged_by_gradient.count", 1);
      KSK_ADD_FRAME_STAT(stats, "newton.termination.gradient_tolerance.count",
                         1);
      break;
    }

    DofBuffer q_before;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/CloneState");
      q_before = q.clone();
    }
    ObjectiveBreakdown objective_before_breakdown;
    double objective_before = 0.0;
    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/EvaluateObjectiveBefore");
      objective_before_breakdown =
          evaluateSubsystemObjectiveAndUpdateGeometry(simulation, dt);
      if (config.enableContact) {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/Step/NewtonIteration/"
            "RebuildActiveContactsForObjectiveBefore");
        contact_detector_.rebuildActiveContacts(simulation);
      }
      appendContactObjectiveBreakdown(simulation, config.enableContact,
                                      objective_before_breakdown);
      objective_before = objective_before_breakdown.total;
      KSK_AVERAGE_FRAME_STAT(stats, "objective.before.average",
                             objective_before);
      KSK_AVERAGE_FRAME_STAT(stats, "objective.before.subsystem.average",
                             objective_before_breakdown.subsystem);
      KSK_AVERAGE_FRAME_STAT(stats, "objective.before.contact.average",
                             objective_before_breakdown.contact);
      KSK_AVERAGE_FRAME_STAT(stats, "objective.before.friction.average",
                             objective_before_breakdown.friction);
      recordContactBreakdownStats(stats, "objective.before",
                                  objective_before_breakdown.contacts);
    }

    bool solved_direction = false;
    std::string direction_failure_reason;
    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/SolveNewtonDirection");
      solved_direction =
          solveNewtonDirection(simulation, gradient, direction, stats,
                               contact_hessian ? &*contact_hessian : nullptr,
                               &direction_failure_reason);
    }

    if (!solved_direction) {
      result.finalStepNorm = 0.0;
      result.terminationReason = direction_failure_reason.empty()
                                     ? "newton_direction_failed"
                                     : direction_failure_reason;
      KSK_ADD_FRAME_STAT(stats, "newton.termination.direction_failed.count", 1);
      break;
    }

    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/PostSolveRefreshBaseline");
      q.copyFrom(q_before);
      synchronizeGeometryFromState(simulation);
      objective_before_breakdown =
          evaluateSubsystemObjectiveAndUpdateGeometry(simulation, dt);
      if (config.enableContact) {
        contact_detector_.rebuildActiveContacts(simulation);
      } else {
        simulation.scene().contacts.clear();
      }

      appendContactObjectiveBreakdown(simulation, config.enableContact,
                                      objective_before_breakdown);
      objective_before = objective_before_breakdown.total;
    }

    double direction_norm = 0.0;
    {
      KSK_PROFILE_SCOPE(
         "GlobalSolver/Step/NewtonIteration/DirectionConvergenceCheck");
      double scene_scale = 1.0;
      double direction_inf_norm = 0.0;
      direction_norm = direction.norm();
      direction_inf_norm = computeDofInfNorm(direction);
      scene_scale = computeSceneScale(simulation.scene().geometry);
      const double step_tolerance =
          config.newtonStepTolerance * scene_scale * dt;
      KSK_AVERAGE_FRAME_STAT(stats, "newton.direction_norm.average",
                             direction_norm);
      KSK_AVERAGE_FRAME_STAT(stats, "newton.direction_inf_norm.average",
                             direction_inf_norm);
      KSK_AVERAGE_FRAME_STAT(stats, "newton.scene_scale.average", scene_scale);
      KSK_AVERAGE_FRAME_STAT(stats, "newton.scaled_step_tolerance.average",
                             step_tolerance);
      if (direction_inf_norm < step_tolerance) {
        result.converged = true;
        result.finalStepNorm = direction_inf_norm;
        result.terminationReason = "step_tolerance";
        KSK_ADD_FRAME_STAT(stats, "newton.converged_by_step.count", 1);
        KSK_ADD_FRAME_STAT(stats, "newton.termination.step_tolerance.count", 1);
        break;
      }
    }

    Real ccd_step_bound = 1.0;

    if (config.enableContact) {
      {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/Step/NewtonIteration/ContactCcdDirectionSearch");
        ccd_step_bound =
            contact_detector_.computeCCDStepBound(simulation, direction);
      }
      KSK_AVERAGE_FRAME_STAT(stats, "newton.ccd_step_upper_bound.average",
                             ccd_step_bound);
    }

    double directional_derivative = 0.0;
    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/DescentDirectionCheck");
      directional_derivative = gradient.dot(direction);
      KSK_AVERAGE_FRAME_STAT(stats, "newton.directional_derivative.average",
                             directional_derivative);
    }
    if (directional_derivative >= 0.0) {
      {
        KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/RestoreState");
        q.copyFrom(q_before);
        synchronizeGeometryFromState(simulation);
        if (config.enableContact) {
          contact_detector_.rebuildActiveContacts(simulation);
        } else {
          simulation.scene().contacts.clear();
        }
      }
      result.finalStepNorm = 0.0;
      result.terminationReason = "non_descent_direction";
      KSK_ADD_FRAME_STAT(stats,
                         "newton.termination.non_descent_direction.count", 1);
      break;
    }

    double alpha = 1.0;
    double initial_alpha = 1.0;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/LineSearchSetup");
      const double ccd_step_scale =
          std::clamp(config.ccdStepSizeScale, 0.0, 1.0);
      const double ccd_step_limit =
          std::clamp(ccd_step_bound, 0.0, 1.0);
      alpha = ccd_step_limit * ccd_step_scale;
      initial_alpha = alpha;
      KSK_AVERAGE_FRAME_STAT(stats, "line_search.initial_alpha.average",
                             initial_alpha);
    }
    if (!std::isfinite(alpha) || alpha <= 0.0) {
      {
        KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/RestoreState");
        q.copyFrom(q_before);
        synchronizeGeometryFromState(simulation);
        if (config.enableContact) {
          contact_detector_.rebuildActiveContacts(simulation);
        } else {
          simulation.scene().contacts.clear();
        }
      }
      result.finalStepNorm = 0.0;
      result.terminationReason = "invalid_alpha";
      KSK_ADD_FRAME_STAT(stats, "newton.termination.invalid_alpha.count", 1);
      break;
    }

    bool accepted = false;
    int line_search_attempts = 0;
    for (int line_search = 0; line_search < config.maxLineSearchIterations;
         line_search++) {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/Step/NewtonIteration/LineSearchIteration");
      {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/Step/NewtonIteration/LineSearch/ApplyTrialState");
        q.assignLinearCombination(q_before, alpha, direction);
      }

      {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/Step/NewtonIteration/LineSearch/SynchronizeGeometry");
        synchronizeGeometryFromState(simulation);
      }

      ObjectiveBreakdown objective_trial_breakdown;
      double objective_trial = 0.0;
      {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/Step/NewtonIteration/LineSearch/EvaluateObjective");
        objective_trial_breakdown =
            evaluateSubsystemObjectiveAndUpdateGeometry(simulation, dt);
      }
      if (config.enableContact) {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/Step/NewtonIteration/LineSearch/"
            "RebuildActiveContacts");
        contact_detector_.rebuildActiveContacts(simulation);
      } else {
        simulation.scene().contacts.clear();
      }
      {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/Step/NewtonIteration/LineSearch/"
            "EvaluateContactObjective");
        appendContactObjectiveBreakdown(simulation, config.enableContact,
                                        objective_trial_breakdown);
        objective_trial = objective_trial_breakdown.total;
      }
      {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/Step/NewtonIteration/LineSearch/AcceptanceTest");
        // Trial contact sets are rebuilt, so contact objectives use GIPC-style
        // monotone decrease instead of an Armijo slope from the previous set.
        const double armijo =
            config.enableContact ? 0.0 : config.lineSearchArmijo;
        const double armijo_rhs =
            objective_before + armijo * alpha * directional_derivative;
        line_search_attempts = line_search + 1;
        KSK_AVERAGE_FRAME_STAT(stats, "line_search.alpha.average", alpha);
        KSK_AVERAGE_FRAME_STAT(stats, "line_search.objective.average",
                               objective_trial);
        KSK_AVERAGE_FRAME_STAT(stats, "line_search.armijo.average", armijo_rhs);
        KSK_AVERAGE_FRAME_STAT(stats, "line_search.subsystem_energy.average",
                               objective_trial_breakdown.subsystem);
        KSK_AVERAGE_FRAME_STAT(stats, "line_search.contact_energy.average",
                               objective_trial_breakdown.contact);
        KSK_AVERAGE_FRAME_STAT(stats, "line_search.friction_energy.average",
                               objective_trial_breakdown.friction);
        KSK_MAX_FRAME_STAT(stats, "line_search.contacts.max",
                           objective_trial_breakdown.contacts.total);

        if (std::isfinite(objective_trial) && objective_trial <= armijo_rhs) {
          accepted = true;
          break;
        }

        alpha *= 0.5;
      }
    }
    KSK_AVERAGE_FRAME_STAT(stats, "line_search.attempts.average",
                           line_search_attempts);
    KSK_MAX_FRAME_STAT(stats, "line_search.attempts.max", line_search_attempts);
    KSK_ADD_FRAME_STAT(stats, "line_search.accepted.count", accepted);

    if (!accepted) {
      {
        KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/RestoreState");
        q.copyFrom(q_before);
        synchronizeGeometryFromState(simulation);
        if (config.enableContact)
          contact_detector_.rebuildActiveContacts(simulation);
        else
          simulation.scene().contacts.clear();
      }
      result.finalStepNorm = 0.0;
      result.terminationReason = "line_search_failed";
      KSK_ADD_FRAME_STAT(stats, "newton.termination.line_search_failed.count",
                         1);
      KSK_RECORD_FRAME_STAT(stats, "line_search.failure.iteration", iteration);
      KSK_RECORD_FRAME_STAT(stats, "line_search.failure.objective_before",
                            objective_before);
      KSK_RECORD_FRAME_STAT(stats, "line_search.failure.directional_derivative",
                            directional_derivative);
      KSK_RECORD_FRAME_STAT(stats, "line_search.failure.ccd_step_bound",
                            ccd_step_bound);
      KSK_RECORD_FRAME_STAT(stats, "line_search.failure.initial_alpha",
                            initial_alpha);
      recordContactBreakdownStats(stats, "line_search.failure.before",
                                  objective_before_breakdown.contacts);
      break;
    }

    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/StepNorm");
      result.finalStepNorm = alpha * direction_norm;
    }

    if (config.enablePCGWarmStart) {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/SaveWarmStart");
      previous_newton_direction = direction.clone();
    }
  }

  {
    KSK_PROFILE_SCOPE("GlobalSolver/Step/ReadFinalSubsystemStates");
    for (const auto& subsystem : subsystems) {
      const DofRange range = subsystem->dofRange();
      subsystem->readState(q.slice(range), qdot.slice(range));
    }
  }

  {
    KSK_PROFILE_SCOPE("GlobalSolver/Step/AcceptSubsystemSteps");
    for (const auto& subsystem : subsystems) {
      const DofRange range = subsystem->dofRange();
      subsystem->acceptStep(q.slice(range), qdot.slice(range), dt);
      subsystem->updateGeometry(simulation.scene().geometry);
    }
  }
  KSK_RECORD_FRAME_STAT(stats, "step.result.iterations", result.iterations);
  KSK_RECORD_FRAME_STAT(stats, "step.result.final_gradient_norm",
                        result.finalGradientNorm);
  KSK_RECORD_FRAME_STAT(stats, "step.result.final_step_norm",
                        result.finalStepNorm);
  KSK_RECORD_FRAME_STAT(stats, "step.result.converged", result.converged);
  if (!result.converged) {
    spdlog::warn(
        "Global solver did not converge: reason={} iterations={} grad={} "
        "step_norm={} dt={} contacts={} scalar_count={} geometry_points={}",
        result.terminationReason, result.iterations, result.finalGradientNorm,
        result.finalStepNorm, dt, simulation.scene().contacts.size(),
        scalar_count, geometry_point_count_);
  }

  return result;
}

double GlobalGaussNewtonSolver::evaluateObjective(SimulationContext& simulation,
                                                  double dt) {
  return evaluateObjectiveBreakdown(simulation, dt, false).total;
}

namespace {
ObjectiveBreakdown evaluateSubsystemObjectiveAndUpdateGeometry(
    SimulationContext& simulation, double dt) {
  KSK_PROFILE_SCOPE("GlobalSolver/EvaluateObjective/Subsystems");
  ObjectiveBreakdown result;
  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  for (const auto& subsystem : simulation.subsystems()) {
    const DofRange range = subsystem->dofRange();
    {
      KSK_PROFILE_SCOPE("GlobalSolver/EvaluateObjective/ReadSubsystemState");
      subsystem->readState(q.slice(range), qdot.slice(range));
    }
    {
      KSK_PROFILE_SCOPE("GlobalSolver/EvaluateObjective/SubsystemObjective");
      result.subsystem +=
          subsystem->evaluateObjective(q.slice(range), qdot.slice(range), dt);
    }
    {
      KSK_PROFILE_SCOPE("GlobalSolver/EvaluateObjective/UpdateGeometry");
      subsystem->updateGeometry(simulation.scene().geometry);
    }
  }
  result.total = result.subsystem;
  return result;
}

void appendContactObjectiveBreakdown(SimulationContext& simulation,
                                     bool include_contact_breakdown,
                                     ObjectiveBreakdown& result) {
  if (!simulation.scene().solverConfig.enableContact) {
    result.total = result.subsystem + result.contact + result.friction;
    return;
  }
  {
    KSK_PROFILE_SCOPE("GlobalSolver/EvaluateObjective/ContactEnergy");
    result.contact = computeContactEnergy(simulation.scene().geometry,
                                          simulation.scene().contacts);
  }
  if (simulation.scene().solverConfig.enableFriction) {
    KSK_PROFILE_SCOPE("GlobalSolver/EvaluateObjective/ContactFrictionEnergy");
    result.friction =
        computeContactFrictionEnergy(simulation.scene().geometry,
                                     simulation.scene().contacts);
  }
  result.contacts =
      computeContactBreakdown(simulation.scene(), include_contact_breakdown);
  result.total = result.subsystem + result.contact + result.friction;
}

ObjectiveBreakdown evaluateObjectiveBreakdown(
    SimulationContext& simulation, double dt, bool include_contact_breakdown) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/EvaluateObjective",
                          ksk::core::profiler_colors::kPurple);
  ObjectiveBreakdown result =
      evaluateSubsystemObjectiveAndUpdateGeometry(simulation, dt);
  appendContactObjectiveBreakdown(simulation, include_contact_breakdown, result);
  return result;
}
}  // namespace

void GlobalGaussNewtonSolver::assembleContactGradient(
    SimulationContext& simulation, DofBuffer& gradient) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/AssembleContactGradient",
                          ksk::core::profiler_colors::kRed);
  const ContactStencils& contacts = simulation.scene().contacts;
  if (!simulation.scene().solverConfig.enableContact) {
    return;
  }
  if (contacts.empty()) {
    return;
  }

  ContactPotentialGradient contact_gradient;
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/AssembleContactGradient/ComputeGeometryGradient");
    contact_gradient =
        computeContactGradient(simulation.scene().geometry, contacts);
  }
  if (contact_gradient.points.empty()) {
    return;
  }

  for (const auto& subsystem : simulation.subsystems()) {
    KSK_PROFILE_SCOPE("GlobalSolver/AssembleContactGradient/ScatterSubsystem");
    std::vector<PointIdx> points;
    std::vector<glm::dvec3> point_gradient;
    auto subsystem_id = subsystem->id();

    for (int i = 0; i < contact_gradient.points.size(); i++) {
      const PointIdx point = contact_gradient.points[i];
      if (simulation.scene().geometry.pointOwner(point).subsystem !=
          subsystem_id) {
        continue;
      }
      points.push_back(point);
      point_gradient.push_back(contact_gradient.gradient.cpu()[i]);
    }

    if (points.empty()) {
      continue;
    }
    subsystem->scatterContactGradient(
        points,
        GeometryBuffer::FromCPU(std::move(point_gradient)).view().asConst(),
        gradient.slice(subsystem->dofRange()));
  }
}

void GlobalGaussNewtonSolver::assembleContactFrictionGradient(
    SimulationContext& simulation, DofBuffer& gradient) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/AssembleContactFrictionGradient",
                          ksk::core::profiler_colors::kRed);
  const ContactStencils& contacts = simulation.scene().contacts;
  if (!simulation.scene().solverConfig.enableContact ||
      !simulation.scene().solverConfig.enableFriction) {
    return;
  }
  if (contacts.empty()) {
    return;
  }

  ContactPotentialGradient friction_gradient;
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/AssembleContactFrictionGradient/"
        "ComputeGeometryGradient");
    friction_gradient = computeContactFrictionGradientWrtGeometry(
        simulation.scene().geometry, contacts);
  }
  if (friction_gradient.points.empty()) {
    return;
  }

  for (const auto& subsystem : simulation.subsystems()) {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/AssembleContactFrictionGradient/ScatterSubsystem");
    std::vector<PointIdx> points;
    std::vector<glm::dvec3> point_gradient;
    auto subsystem_id = subsystem->id();

    for (int i = 0; i < friction_gradient.points.size(); i++) {
      const PointIdx point = friction_gradient.points[i];
      if (simulation.scene().geometry.pointOwner(point).subsystem !=
          subsystem_id) {
        continue;
      }
      points.push_back(point);
      point_gradient.push_back(friction_gradient.gradient.cpu()[i]);
    }

    if (points.empty()) {
      continue;
    }
    subsystem->scatterContactGradient(
        points,
        GeometryBuffer::FromCPU(std::move(point_gradient)).view().asConst(),
        gradient.slice(subsystem->dofRange()));
  }
}

bool GlobalGaussNewtonSolver::solveNewtonDirection(
    SimulationContext& simulation, const DofBuffer& gradient,
    DofBuffer& direction, GlobalSolverStatsCollector* stats,
    const ContactGeometryHessian* contactHessian, std::string* failureReason) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/SolveNewtonDirection",
                          ksk::core::profiler_colors::kBlue);
  const GlobalSolverConfig& config = simulation.scene().solverConfig;
  const int scalar_count = simulation.scene().dofs.totalScalars;
  if (scalar_count == 0) {
    direction.setZero();
    KSK_ADD_FRAME_STAT(stats, "solve.mode.zero_scalar_system.count", 1);
    return true;
  }

  const GlobalSolveMode solve_mode = config.solveMode;
  KSK_ADD_FRAME_STAT(
      stats, std::string("solve.mode.") + solveModeName(solve_mode) + ".count",
      1);
  switch (solve_mode) {
    case GlobalSolveMode::Decoupled:
      return solveDecoupledDirection(simulation, gradient, direction, stats,
                                     contactHessian, failureReason);
    case GlobalSolveMode::Coupled: {
      std::string coupled_failure_reason;
      const bool solved =
          solveCoupledDirection(simulation, gradient, direction, stats,
                                contactHessian, &coupled_failure_reason);
      if (solved) {
        return true;
      }
      KSK_ADD_FRAME_STAT(stats, "coupled.fallback_decoupled.count", 1);
      spdlog::warn(
          "Coupled solve failed: reason={}; falling back to decoupled solve",
          coupled_failure_reason.empty() ? "<unknown>"
                                         : coupled_failure_reason);
      return solveDecoupledDirection(simulation, gradient, direction, stats,
                                     contactHessian, failureReason);
    }
  }

  if (failureReason != nullptr) {
    *failureReason = "unknown_global_solve_mode";
  }
  return false;
}

bool GlobalGaussNewtonSolver::solveDecoupledDirection(
    SimulationContext& simulation, const DofBuffer& gradient,
    DofBuffer& direction, GlobalSolverStatsCollector* stats,
    const ContactGeometryHessian* contactHessian, std::string* failureReason) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/SolveDecoupledDirection",
                          ksk::core::profiler_colors::kBlue);
  const GlobalSolverConfig& config = simulation.scene().solverConfig;
  const int scalar_count = simulation.scene().dofs.totalScalars;

  if (!config.enableContact || simulation.scene().contacts.empty()) {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveDecoupledDirection/BlockDiagonalSubsystems");
    const bool solved =
        solveBlockDiagonalSubsystemDirection(simulation, gradient, direction);
    if (!solved && failureReason != nullptr) {
      *failureReason = "block_diagonal_direction_failed";
    }
    return solved;
  }

  bool use_warm_start = false;
  double initial_direction_norm = 0.0;
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveDecoupledDirection/InitializePcgDirection");
    initial_direction_norm = direction.norm();
    use_warm_start =
        config.enablePCGWarmStart && std::isfinite(initial_direction_norm) &&
        initial_direction_norm > 0.0;
  }

  if (!use_warm_start) {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveDecoupledDirection/ZeroInitialGuess");
    direction.setZero();
  }

  DofBuffer residual;
  double rhs_norm = 0.0;
  double initial_residual_norm = 0.0;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/BuildResidual");
    residual = createDofBufferLike(gradient, scalar_count);
    residual.assignScaled(-1.0, gradient);
    rhs_norm = residual.norm();

    if (use_warm_start) {
      DofBuffer initial_product = createDofBufferLike(gradient, scalar_count);
      {
        KSK_PROFILE_SCOPE(
            "GlobalSolver/SolveNewtonDirection/WarmStartApplyGlobalMatrix");
        applyGlobalMatrix(simulation, direction, initial_product,
                          contactHessian);
      }
      residual.addScaled(-1.0, initial_product);
      const double warm_residual_norm = residual.norm();
      if (std::isfinite(warm_residual_norm)) {
        KSK_ADD_FRAME_STAT(stats, "pcg.warm_start.used.count", 1);
        KSK_AVERAGE_FRAME_STAT(
            stats, "pcg.warm_start.initial_residual_norm.average",
            warm_residual_norm);
        KSK_AVERAGE_FRAME_STAT(
            stats, "pcg.warm_start.initial_residual_ratio.average",
            rhs_norm > 0.0 ? warm_residual_norm / rhs_norm : 0.0);
      } else {
        KSK_ADD_FRAME_STAT(stats, "pcg.warm_start.rejected.count", 1);
        direction.setZero();
        residual.assignScaled(-1.0, gradient);
        use_warm_start = false;
      }
    }
    initial_residual_norm = residual.norm();
  }

  if (rhs_norm == 0.0) {
    direction.setZero();
    return true;
  }

  const double tolerance = config.pcgTolerance * std::max(1.0, rhs_norm);
  if (initial_residual_norm <= tolerance) {
    KSK_AVERAGE_FRAME_STAT(stats, "pcg.iterations.average", 0);
    KSK_AVERAGE_FRAME_STAT(stats, "pcg.final_residual_norm.average",
                           initial_residual_norm);
    KSK_ADD_FRAME_STAT(stats, "pcg.converged.count", 1);
    return true;
  }

  DofBuffer z = createDofBufferLike(gradient, scalar_count);
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveNewtonDirection/InitialPreconditioner");
    applyPreconditioner(simulation, residual, z);
  }

  DofBuffer search;
  double rz_old = 0.0;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/InitialDotProducts");
    search = z.clone();
    rz_old = residual.dot(z);
  }

#ifndef NDEBUG
  if (!std::isfinite(rz_old) || rz_old <= 0.0) {
    warnInvalidPreconditionerDotBreakdown(simulation, residual, z, "initial", 0,
                                          rz_old, rhs_norm, rhs_norm);

    if (failureReason != nullptr) *failureReason = "pcg_invalid_initial_rz";

    return false;
  }
#endif
  DofBuffer matrix_search = createDofBufferLike(gradient, scalar_count);
  auto record_verified_residual = [&](const char* label) {
    if (stats == nullptr) return;

    (void)label;
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/VerifyResidual");
    DofBuffer verification = createDofBufferLike(gradient, scalar_count);
    applyGlobalMatrix(simulation, direction, verification, contactHessian);
    verification.addScaled(1.0, gradient);
  };

  for (int iteration = 0; iteration < config.maxPcgIterations; ++iteration) {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/PcgIteration");
    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/SolveNewtonDirection/Pcg/ApplyGlobalMatrix");
      applyGlobalMatrix(simulation, search, matrix_search, contactHessian);
    }

    double denominator = 0.0;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/DenominatorDot");
      denominator = search.dot(matrix_search);
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) {
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.iterations.average", iteration + 1);
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.invalid_denominator.average",
                             denominator);
      KSK_ADD_FRAME_STAT(stats, "pcg.failure.invalid_denominator.count", 1);
      if (failureReason != nullptr) {
        *failureReason = "pcg_invalid_denominator";
      }
      return false;
    }

    const double alpha = rz_old / denominator;
    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/SolveNewtonDirection/Pcg/UpdateDirectionResidual");
      direction.addScaled(alpha, search);
      residual.addScaled(-alpha, matrix_search);
    }

    double residual_norm = 0.0;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/ResidualNorm");
      residual_norm = residual.norm();
    }
    KSK_AVERAGE_FRAME_STAT(stats, "pcg.denominator.average", denominator);
    KSK_AVERAGE_FRAME_STAT(stats, "pcg.alpha.average", alpha);
    if (residual_norm <= tolerance) {
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.iterations.average", iteration + 1);
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.final_residual_norm.average",
                             residual_norm);
      KSK_ADD_FRAME_STAT(stats, "pcg.converged.count", 1);
      record_verified_residual("converged");
      return true;
    }

    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/SolveNewtonDirection/Pcg/ApplyPreconditioner");
      applyPreconditioner(simulation, residual, z);
    }
    double rz_new = residual.dot(z);

#ifndef NDEBUG
    if (!std::isfinite(rz_new) || rz_new <= 0.0) {
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.iterations.average", iteration + 1);
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.invalid_rz.average", rz_new);
      KSK_ADD_FRAME_STAT(stats, "pcg.failure.invalid_rz.count", 1);
      warnInvalidPreconditionerDotBreakdown(simulation, residual, z,
                                            "iteration", iteration + 1, rz_new,
                                            residual_norm, rhs_norm);
      if (failureReason != nullptr) {
        *failureReason = "pcg_invalid_rz";
      }
      return false;
    }
#endif

    const double beta = rz_new / rz_old;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/UpdateSearch");
      search.assignLinearCombination(z, beta, search);
    }
    rz_old = rz_new;
  }

  const double final_residual_norm = residual.norm();
  const bool converged = final_residual_norm <= tolerance;
  KSK_ADD_FRAME_STAT(stats, "pcg.converged.count", converged);
//  record_verified_residual("max_iterations");
  if (!converged && failureReason != nullptr) {
    *failureReason = "pcg_max_iterations";
  }
  return converged;
}

bool GlobalGaussNewtonSolver::solveCoupledDirection(
    SimulationContext& simulation, const DofBuffer& gradient,
    DofBuffer& direction, GlobalSolverStatsCollector* stats,
    const ContactGeometryHessian* contactHessian, std::string* failureReason) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/SolveCoupledDirection",
                          ksk::core::profiler_colors::kPurple);
  const GlobalSolverConfig& config = simulation.scene().solverConfig;
  const int scalar_count = simulation.scene().dofs.totalScalars;
  if (!gradient.isCPU() || !direction.isCPU()) {
    if (failureReason != nullptr) {
      *failureReason = "coupled_solve_requires_cpu_buffers";
    }
    return false;
  }

  direction.setZero();
  if (gradient.norm() == 0.0) {
    return true;
  }

  using SparseMatrix = Eigen::SparseMatrix<double>;
  using Triplet = Eigen::Triplet<double>;
  std::vector<Triplet> triplets;
  std::vector<SparseMatrixEntry> local_entries;
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveCoupledDirection/AppendLocalMatrixEntries");
    for (const auto& subsystem : simulation.subsystems()) {
      subsystem->appendLocalMatrixEntries(local_entries);
    }
  }
  triplets.reserve(local_entries.size() +
                   static_cast<size_t>(scalar_count) * 4);
  for (const SparseMatrixEntry& entry : local_entries) {
    triplets.emplace_back(entry.row, entry.col, entry.value);
  }

  if (config.enableContact && !simulation.scene().contacts.empty()) {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveCoupledDirection/AssembleContactMatrixColumns");
    DofBuffer basis = createDofBufferLike(gradient, scalar_count);
    DofBuffer column = createDofBufferLike(gradient, scalar_count);
    for (int col = 0; col < scalar_count; ++col) {
      basis.setZero();
      basis.cpu()[col] = 1.0;
      column.setZero();
      applyGlobalContactHessian(simulation, basis, column, contactHessian);
      const Eigen::VectorXd& values = column.cpu();
      for (int row = 0; row < scalar_count; ++row) {
        const double value = values[row];
        if (value != 0.0) {
          triplets.emplace_back(row, col, value);
        }
      }
    }
  }
  for (int i = 0; i < scalar_count; ++i) {
    triplets.emplace_back(i, i, config.coupledDiagonalRegularization);
  }

  SparseMatrix matrix;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveCoupledDirection/CompressMatrix");
    matrix.resize(scalar_count, scalar_count);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
  }
  KSK_AVERAGE_FRAME_STAT(stats, "coupled.matrix_nnz.average",
                         matrix.nonZeros());
  KSK_MAX_FRAME_STAT(stats, "coupled.matrix_nnz.max", matrix.nonZeros());

  Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper | Eigen::Lower> solver;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveCoupledDirection/Factorize");
    solver.compute(matrix);
  }
  if (solver.info() != Eigen::Success) {
    KSK_ADD_FRAME_STAT(stats, "coupled.failure.factorize.count", 1);
    if (failureReason != nullptr) {
      *failureReason = "coupled_factorization_failed";
    }
    return false;
  }

  Eigen::VectorXd solution;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveCoupledDirection/Solve");
    solution = solver.solve(-gradient.cpu());
  }
  if (solver.info() != Eigen::Success || !solution.allFinite()) {
    KSK_ADD_FRAME_STAT(stats, "coupled.failure.solve.count", 1);
    if (failureReason != nullptr) {
      *failureReason = "coupled_solve_failed";
    }
    return false;
  }

  direction.cpu() = std::move(solution);
  KSK_ADD_FRAME_STAT(stats, "coupled.solved.count", 1);
  return std::isfinite(direction.norm());
}

bool GlobalGaussNewtonSolver::solveSingleSubsystemDirection(
    SimulationContext& simulation, const DofBuffer& gradient,
    DofBuffer& direction) {
  KSK_PROFILE_SCOPE("GlobalSolver/SolveSingleSubsystemDirection");
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveSingleSubsystemDirection/SetZero");
    direction.setZero();
  }

  DofBuffer residual;
  double residual_norm = 0.0;
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveSingleSubsystemDirection/BuildResidual");
    residual =
        createDofBufferLike(gradient, simulation.scene().dofs.totalScalars);
    residual.assignScaled(-1.0, gradient);
    residual_norm = residual.norm();
  }
  if (residual_norm == 0.0) {
    return true;
  }

  const DofRange range = simulation.subsystems().front()->dofRange();
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveSingleSubsystemDirection/SolveLocalSystem");
    simulation.subsystems().front()->solveLocalSystem(residual.slice(range),
                                                      direction.slice(range));
  }
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveSingleSubsystemDirection/DirectionNorm");
    return std::isfinite(direction.norm());
  }
}

bool GlobalGaussNewtonSolver::solveBlockDiagonalSubsystemDirection(
    SimulationContext& simulation, const DofBuffer& gradient,
    DofBuffer& direction) {
  KSK_PROFILE_SCOPE("GlobalSolver/SolveBlockDiagonalSubsystemDirection");
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveBlockDiagonalSubsystemDirection/SetZero");
    direction.setZero();
  }

  DofBuffer residual;
  double residual_norm = 0.0;
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveBlockDiagonalSubsystemDirection/BuildResidual");
    residual =
        createDofBufferLike(gradient, simulation.scene().dofs.totalScalars);
    residual.assignScaled(-1.0, gradient);
    residual_norm = residual.norm();
  }
  if (residual_norm == 0.0) {
    return true;
  }

  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveBlockDiagonalSubsystemDirection/SolveLocalSystems");
    for (const auto& subsystem : simulation.subsystems()) {
      const DofRange range = subsystem->dofRange();
      subsystem->solveLocalSystem(residual.slice(range),
                                  direction.slice(range));
    }
  }

  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/SolveBlockDiagonalSubsystemDirection/DirectionNorm");
    return std::isfinite(direction.norm());
  }
}

void GlobalGaussNewtonSolver::applyGlobalMatrix(
    SimulationContext& simulation, const DofBuffer& x, DofBuffer& y,
    const ContactGeometryHessian* contactHessian) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/ApplyGlobalMatrix",
                          ksk::core::profiler_colors::kCyan);
  {
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyGlobalMatrix/SetZero");
    y.setZero();
  }
  for (const auto& subsystem : simulation.subsystems()) {
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyGlobalMatrix/ApplyLocalMatrix");
    const DofRange range = subsystem->dofRange();
    subsystem->applyLocalMatrix(x.slice(range), y.slice(range));
  }

  const ContactStencils& contacts = simulation.scene().contacts;
  if (simulation.scene().solverConfig.enableContact && !contacts.empty()) {
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyGlobalMatrix/ApplyContactHessian");
    applyGlobalContactHessian(simulation, x, y, contactHessian);
  }
}

void GlobalGaussNewtonSolver::applyGlobalContactHessian(
    SimulationContext& simulation, const DofBuffer& x, DofBuffer& y,
    const ContactGeometryHessian* contactHessian) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/ApplyGlobalContactHessian",
                          ksk::core::profiler_colors::kOrange);
  const ContactStencils& contacts = simulation.scene().contacts;
  if (!simulation.scene().solverConfig.enableContact) {
    return;
  }
  if (contacts.empty()) {
    return;
  }

  GeometryBuffer geometry_direction;
  {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/ApplyGlobalContactHessian/CreateGeometryDirection");
    geometry_direction = createGeometryDirectionBuffer(
        x, simulation.scene().geometry.pointCount());
  }
  for (const auto& subsystem : simulation.subsystems()) {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/ApplyGlobalContactHessian/MapDirectionToGeometry");
    subsystem->mapLocalDirectionToGeometry(x.slice(subsystem->dofRange()),
                                           geometry_direction.view());
  }

  struct ContactProductBucket {
    std::vector<PointIdx> points;
    std::vector<glm::dvec3> values;
  };

  std::vector<ContactProductBucket> product_buckets(
      simulation.subsystems().size());

  if (contactHessian != nullptr) {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/ApplyGlobalContactHessian/ApplyCachedGeometryHessian");
    const int point_count = simulation.scene().geometry.pointCount();
    std::vector<glm::dvec3> point_products(static_cast<size_t>(point_count),
                                           glm::dvec3(0.0));
    std::vector<unsigned char> touched(static_cast<size_t>(point_count), 0);
    std::vector<PointIdx> product_points;
    product_points.reserve(std::min(contactHessian->blocks.size(),
                                    static_cast<size_t>(point_count)));
    const std::vector<glm::dvec3>& direction_values = geometry_direction.cpu();
    for (const ContactGeometryHessianBlock& block : contactHessian->blocks) {
      assert(block.row >= 0);
      assert(block.row < point_count);
      assert(block.col >= 0);
      assert(block.col < point_count);
      if (touched[static_cast<size_t>(block.row)] == 0) {
        touched[static_cast<size_t>(block.row)] = 1;
        product_points.push_back(block.row);
      }
      point_products[static_cast<size_t>(block.row)] +=
          block.value * direction_values[static_cast<size_t>(block.col)];
    }

    if (product_points.empty()) {
      return;
    }
    if (!product_buckets.empty()) {
      const auto average_bucket_size =
          product_points.size() / product_buckets.size() + 1;
      for (auto& bucket : product_buckets) {
        bucket.points.reserve(average_bucket_size);
        bucket.values.reserve(average_bucket_size);
      }
    }
    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/ApplyGlobalContactHessian/BucketProducts");
      for (const PointIdx point : product_points) {
        const SubsystemIdx subsystem_id =
            simulation.scene().geometry.pointOwner(point).subsystem;
        assert(subsystem_id >= 0);
        assert(subsystem_id <
               static_cast<SubsystemIdx>(product_buckets.size()));
        auto& bucket = product_buckets[static_cast<size_t>(subsystem_id)];
        bucket.points.push_back(point);
        bucket.values.push_back(point_products[static_cast<size_t>(point)]);
      }
    }
  } else {
    ContactPotentialGradient contact_hessian_product;
    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/ApplyGlobalContactHessian/"
          "ComputeGeometryHessianProduct");
      contact_hessian_product = computeContactHessianProductWrtGeometry(
          simulation.scene().geometry, contacts,
          geometry_direction.view().asConst());
    }
    ContactPotentialGradient friction_hessian_product;
    if (simulation.scene().solverConfig.enableFriction) {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/ApplyGlobalContactHessian/"
          "ComputeGeometryFrictionHessianProduct");
      friction_hessian_product =
          computeContactFrictionHessianProductWrtGeometry(
              simulation.scene().geometry, contacts,
              geometry_direction.view().asConst());
    }
    if (contact_hessian_product.points.empty() &&
        friction_hessian_product.points.empty()) {
      return;
    }

    const size_t total_points = contact_hessian_product.points.size() +
                                friction_hessian_product.points.size();
    if (!product_buckets.empty()) {
      const auto average_bucket_size =
          total_points / product_buckets.size() + 1;
      for (auto& bucket : product_buckets) {
        bucket.points.reserve(average_bucket_size);
        bucket.values.reserve(average_bucket_size);
      }
    }
    auto bucket_product = [&](const ContactPotentialGradient& product) {
      if (product.points.empty()) {
        return;
      }
      const auto& hessian_points = product.points;
      const auto& hessian_values = product.gradient.cpu();
      for (int i = 0; i < hessian_points.size(); ++i) {
        const PointIdx point = hessian_points[static_cast<size_t>(i)];
        const SubsystemIdx subsystem_id =
            simulation.scene().geometry.pointOwner(point).subsystem;
        assert(subsystem_id >= 0);
        assert(subsystem_id <
               static_cast<SubsystemIdx>(product_buckets.size()));
        auto& bucket = product_buckets[static_cast<size_t>(subsystem_id)];
        bucket.points.push_back(point);
        bucket.values.push_back(hessian_values[static_cast<size_t>(i)]);
      }
    };
    {
      KSK_PROFILE_SCOPE(
          "GlobalSolver/ApplyGlobalContactHessian/BucketProducts");
      bucket_product(contact_hessian_product);
      bucket_product(friction_hessian_product);
    }
  }

  for (const auto& subsystem : simulation.subsystems()) {
    KSK_PROFILE_SCOPE(
        "GlobalSolver/ApplyGlobalContactHessian/ScatterSubsystem");
    const auto subsystem_id = static_cast<size_t>(subsystem->id());
    assert(subsystem_id < product_buckets.size());
    auto& bucket = product_buckets[subsystem_id];

    if (bucket.points.empty()) {
      continue;
    }
    subsystem->applyContactGeometryHessianProduct(
        bucket.points,
        GeometryBuffer::FromCPU(std::move(bucket.values)).view().asConst(),
        y.slice(subsystem->dofRange()));
  }
}

void GlobalGaussNewtonSolver::applyPreconditioner(SimulationContext& simulation,
                                                  const DofBuffer& residual,
                                                  DofBuffer& z) {
  KSK_PROFILE_SCOPE_COLOR("GlobalSolver/ApplyPreconditioner",
                          ksk::core::profiler_colors::kGreen);
  {
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyPreconditioner/SetZero");
    z.setZero();
  }
  for (const auto& subsystem : simulation.subsystems()) {
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyPreconditioner/SolveLocalSystem");
    const DofRange range = subsystem->dofRange();
    subsystem->solveLocalSystem(residual.slice(range), z.slice(range));
  }
}

SimulationRunner::SimulationRunner(SimulationContext simulation,
                                   double timeStep)
    : simulation_(std::move(simulation)), time_step_(timeStep) {}

RuntimeStepResult SimulationRunner::step(GlobalSolverStatsCollector* stats) {
  KSK_PROFILE_SCOPE_COLOR("SimulationRunner/Step",
                          ksk::core::profiler_colors::kBlue);
  {
    KSK_PROFILE_SCOPE("SimulationRunner/Step/SolverStep");
    last_step_ = solver_.step(simulation_, time_step_, time_, stats);
  }
  {
    KSK_PROFILE_SCOPE("SimulationRunner/Step/AdvanceCounters");
    time_ += time_step_;
    ++steps_completed_;
  }
  return last_step_;
}

RuntimeStepResult SimulationRunner::run(int steps,
                                        GlobalSolverStatsCollector* stats) {
  for (int step_index = 0; step_index < steps; ++step_index) {
    last_step_ = step(stats);
  }
  return last_step_;
}

SimulationRunner buildSimulationRunner(const RuntimeSceneDesc& scene) {
  return {buildSimulation(scene), scene.timeStep};
}

}  // namespace ksk::runtime
