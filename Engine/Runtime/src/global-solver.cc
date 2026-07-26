#include <Runtime/global-solver.h>

#include <Core/profiler.h>
#include <Runtime/contact-barrier-energy.h>

#include <algorithm>
#include <cmath>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ksk::runtime {
namespace {

#ifndef NDEBUG
void validateSimulationStateForStep(const SimulationContext& simulation)
{
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
        ", q.scalarCount=" +
        std::to_string(q_count) +
        ", qdot.scalarCount=" +
        std::to_string(qdot_count) + ".");
  }
}
#endif

GeometryBuffer createGeometryDirectionBuffer(const DofBuffer& direction,
                                           int point_count)
{
  if (direction.isCPU())
    return GeometryBuffer::CPU(point_count);
  return GeometryBuffer::GPU(*direction.device(), point_count);
}

DofBuffer createDofBufferLike(const DofBuffer& reference, int scalar_count)
{
  if (reference.isCPU()) {
    return DofBuffer::CPU(scalar_count);
  }
  return DofBuffer::GPU(*reference.device(), scalar_count);
}

double computeSceneScale(const GlobalGeometryManager& geometry)
{
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

double computeDofInfNorm(const DofBuffer& buffer)
{
  if (buffer.isCPU()) {
    return buffer.cpu().cwiseAbs().maxCoeff();
  }
  throw std::runtime_error("GPU DofBuffer infNorm is not implemented");
}

double computeDofViewDot(ConstDofView a, ConstDofView b)
{
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

double computeDofViewNorm(ConstDofView view)
{
  const double squared_norm = computeDofViewDot(view, view);
  if (!std::isfinite(squared_norm) || squared_norm < 0.0) {
    return squared_norm;
  }
  return std::sqrt(squared_norm);
}

double computeDofViewInfNorm(ConstDofView view)
{
  if (!view.isCPU()) {
    throw std::runtime_error("GPU DofView infNorm is not implemented");
  }
  double norm = 0.0;
  for (int i = 0; i < view.scalarCount(); ++i) {
    norm = std::max(norm, std::abs(view[i]));
  }
  return norm;
}

std::string subsystemTypeName(const RuntimeScene& scene, SubsystemIdx subsystem)
{
  for (const SubsystemBatch& batch : scene.subsystemBatches) {
    if (subsystem >= batch.firstSubsystem &&
        subsystem < batch.firstSubsystem + batch.subsystemCount) {
      return batch.type;
    }
  }
  return {};
}

int contactPointCount(ContactCase type)
{
  switch (type) {
  case ContactCase::PP:
    return 2;
  case ContactCase::PE:
    return 3;
  case ContactCase::PT:
  case ContactCase::EE:
    return 4;
  }
  return 0;
}

const char* contactCaseName(ContactCase type)
{
  switch (type) {
  case ContactCase::PP:
    return "PP";
  case ContactCase::PE:
    return "PE";
  case ContactCase::PT:
    return "PT";
  case ContactCase::EE:
    return "EE";
  }
  return "<unknown>";
}

void warnActiveContactBreakdown(const RuntimeScene& scene)
{
  constexpr int kMaxLoggedContacts = 24;
  const int logged_contacts =
      std::min<int>(kMaxLoggedContacts, scene.contacts.size());
  for (int contact_index = 0; contact_index < logged_contacts; ++contact_index) {
    const ContactStencil& contact =
        scene.contacts[static_cast<size_t>(contact_index)];
    const int point_count = contactPointCount(contact.type);
    std::array<GeometryOwner, 4> owners{};
    for (int i = 0; i < point_count; ++i) {
      owners[static_cast<size_t>(i)] =
          scene.geometry.pointOwner(PointIdx{contact.geometryIds[i]});
    }
    spdlog::warn(
        "active contact: index={} type={} ids=[{},{},{},{}] "
        "subsystems=[{},{},{},{}] bodies=[{},{},{},{}] dHat={} "
        "thickness={} stiffness={}",
        contact_index,
        contactCaseName(contact.type),
        contact.geometryIds[0],
        contact.geometryIds[1],
        contact.geometryIds[2],
        contact.geometryIds[3],
        owners[0].subsystem,
        owners[1].subsystem,
        owners[2].subsystem,
        owners[3].subsystem,
        owners[0].body,
        owners[1].body,
        owners[2].body,
        owners[3].body,
        contact.dHat,
        contact.thickness,
        contact.stiffness);
  }
  if (static_cast<int>(scene.contacts.size()) > logged_contacts) {
    spdlog::warn("active contact: omitted_count={}",
                 static_cast<int>(scene.contacts.size()) - logged_contacts);
  }
}

void warnInvalidPreconditionerDotBreakdown(
    SimulationContext& simulation,
    const DofBuffer& residual,
    const DofBuffer& z,
    const char* phase,
    int pcg_iteration,
    double rz,
    double residual_norm,
    double rhs_norm)
{
  spdlog::warn(
      "PCG preconditioner produced invalid rz: phase={} pcg_iteration={} "
      "rz={} residual_norm={} rhs_norm={} contacts={}",
      phase,
      pcg_iteration,
      rz,
      residual_norm,
      rhs_norm,
      simulation.scene().contacts.size());
  warnActiveContactBreakdown(simulation.scene());
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
        subsystem->id(),
        type_name.empty() ? "<unknown>" : type_name,
        range.scalarOffset,
        range.scalarCount,
        computeDofViewDot(local_residual, local_z),
        computeDofViewNorm(local_residual),
        computeDofViewNorm(local_z),
        computeDofViewInfNorm(local_residual),
        computeDofViewInfNorm(local_z));
  }
}

void synchronizeGeometryFromState(SimulationContext& simulation)
{
  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  for (const auto& subsystem : simulation.subsystems()) {
    const DofRange range = subsystem->dofRange();
    subsystem->readState(q.slice(range), qdot.slice(range));
    subsystem->updateGeometry(simulation.scene().geometry);
  }
}


#define KSK_RECORD_FRAME_STAT(stats, key, value) \
  do {                                           \
    if ((stats) != nullptr) {                    \
      (stats)->frameSet((key), (value));       \
    }                                            \
  } while (false)

#define KSK_ADD_FRAME_STAT(stats, key, value) \
  do {                                        \
    if ((stats) != nullptr) {                 \
      (stats)->frameAdd((key), (value));       \
    }                                         \
  } while (false)

#define KSK_AVERAGE_FRAME_STAT(stats, key, value) \
  do {                                            \
    if ((stats) != nullptr) {                     \
      (stats)->frameAverage((key), (value));       \
    }                                             \
  } while (false)

#define KSK_MAX_FRAME_STAT(stats, key, value) \
  do {                                        \
    if ((stats) != nullptr) {                 \
      (stats)->frameMax((key), (value));       \
    }                                         \
  } while (false)

}  // namespace

void GlobalSolverStatsCollector::beginFrame()
{
  frames_.emplace_back();
  frame_average_counts_.emplace_back();
}

void GlobalSolverStatsCollector::clear() noexcept
{
  global_.clear();
  global_average_counts_.clear();
  frames_.clear();
  frame_average_counts_.clear();
}

void GlobalSolverStatsCollector::frameSet(std::string key, double value)
{
  currentFrame()[std::move(key)] = value;
}

void GlobalSolverStatsCollector::frameAdd(std::string key, double value)
{
  add(currentFrame(), std::move(key), value);
}

void GlobalSolverStatsCollector::frameMax(std::string key, double value)
{
  max(currentFrame(), std::move(key), value);
}

void GlobalSolverStatsCollector::frameMin(std::string key, double value)
{
  min(currentFrame(), std::move(key), value);
}

void GlobalSolverStatsCollector::frameAverage(std::string key, double value)
{
  average(currentFrame(), frame_average_counts_.back(), std::move(key), value);
}

void GlobalSolverStatsCollector::globalSet(std::string key, double value)
{
  global_[std::move(key)] = value;
}

void GlobalSolverStatsCollector::globalAdd(std::string key, double value)
{
  add(global_, std::move(key), value);
}

void GlobalSolverStatsCollector::globalMax(std::string key, double value)
{
  max(global_, std::move(key), value);
}

void GlobalSolverStatsCollector::globalMin(std::string key, double value)
{
  min(global_, std::move(key), value);
}

void GlobalSolverStatsCollector::globalAverage(std::string key, double value)
{
  average(global_, global_average_counts_, std::move(key), value);
}

void GlobalSolverStatsCollector::add(StatMap& stats,
                                     std::string key,
                                     double value)
{
  stats[std::move(key)] += value;
}

void GlobalSolverStatsCollector::max(StatMap& stats,
                                     std::string key,
                                     double value)
{
  auto [it, inserted] = stats.try_emplace(std::move(key), value);
  if (!inserted) {
    it->second = std::max(it->second, value);
  }
}

void GlobalSolverStatsCollector::min(StatMap& stats,
                                     std::string key,
                                     double value)
{
  auto [it, inserted] = stats.try_emplace(std::move(key), value);
  if (!inserted) {
    it->second = std::min(it->second, value);
  }
}

void GlobalSolverStatsCollector::average(StatMap& stats,
                                         std::map<std::string, int>& counts,
                                         std::string key,
                                         double value)
{
  const std::string count_key = key;
  const int count = ++counts[count_key];
  auto [it, inserted] = stats.try_emplace(std::move(key), value);
  if (!inserted) {
    it->second += (value - it->second) / static_cast<double>(count);
  }
}

GlobalSolverStatsCollector::StatMap& GlobalSolverStatsCollector::currentFrame()
{
  if (frames_.empty()) {
    beginFrame();
  }
  return frames_.back();
}

void GlobalGaussNewtonSolver::prepare(const RuntimeScene& scene)
{
  scalar_count_ = scene.dofs.totalScalars;
  geometry_point_count_ = static_cast<int>(scene.geometry.points.size());
}

RuntimeStepResult GlobalGaussNewtonSolver::step(SimulationContext& simulation,
                                                double dt,
                                                double time,
                                                GlobalSolverStatsCollector* stats)
{
  SIM_PROFILE_SCOPE_COLOR("GlobalSolver/Step",
                          ksk::core::profiler_colors::kBlue);
#ifndef NDEBUG
  {
    KSK_PROFILE_SCOPE("GlobalSolver/Step/ValidateState");
    validateSimulationStateForStep(simulation);
  }
#endif
  prepare(simulation.scene());

  DofBuffer& q = simulation.q();
  DofBuffer& qdot = simulation.qdot();
  auto& subsystems = simulation.subsystems();

    for (const auto& subsystem : subsystems) {
      const DofRange range = subsystem->dofRange();
      subsystem->readState(q.slice(range), qdot.slice(range));
      subsystem->beginStep(q.slice(range), qdot.slice(range), dt);
    }

  RuntimeStepResult result;
  const GlobalSolverConfig& config = simulation.scene().solverConfig;
  const int scalar_count = simulation.scene().dofs.totalScalars;
  if (stats) {
    stats->beginFrame();
  }
  KSK_RECORD_FRAME_STAT(stats, "step.time", time);
  KSK_RECORD_FRAME_STAT(stats, "step.dt", dt);
  KSK_RECORD_FRAME_STAT(stats, "step.scalar_count", scalar_count);
  KSK_RECORD_FRAME_STAT(stats, "step.geometry_point_count", geometry_point_count_);
  KSK_RECORD_FRAME_STAT(stats, "step.subsystem_count", static_cast<int>(subsystems.size()));
  if (!config.enableContact) {
    simulation.scene().contacts.clear();
  }
  result.terminationReason = "max_newton_iterations";

  for (int iteration = 0; iteration < config.maxNewtonIterations; iteration++) {
    KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration");
    DofBuffer gradient;
    DofBuffer direction;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/CreateBuffers");
      gradient = createDofBufferLike(q, scalar_count);
      direction = createDofBufferLike(q, scalar_count);
    }

    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/UpdateSubsystemStates");
      for (const auto& subsystem : subsystems) {
        const DofRange range = subsystem->dofRange();
        subsystem->readState(q.slice(range), qdot.slice(range));
        subsystem->updateInternalConstraints(time + dt, dt);
        subsystem->updateGeometry(simulation.scene().geometry);
      }
    }
    if (config.enableContact) {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/RebuildActiveContacts");
      contact_detector_.rebuildActiveContacts(simulation);
    } else
      simulation.scene().contacts.clear();

    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/AssembleLocal"); // NOLINT(*-reserved-identifier)
      for (const auto& subsystem : subsystems) {
        const DofRange range = subsystem->dofRange();
        {
          KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/PrepareLocalOperator");
          subsystem->prepareLocalOperator(dt);
        }
        {
          KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/AssembleLocalGradient");
          subsystem->assembleLocalGradient(gradient.slice(range));
        }
      }
    }

    if (config.enableContact) {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/AssembleContactGradient");
      assembleContactGradient(simulation, gradient);
    }

    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/GradientNorm");
      result.finalGradientNorm = gradient.norm();
    }
    result.iterations = iteration + 1;
    KSK_AVERAGE_FRAME_STAT(stats, "newton.gradient_norm.average", result.finalGradientNorm);
    KSK_MAX_FRAME_STAT(stats, "newton.gradient_norm.max", result.finalGradientNorm);
    if (result.finalGradientNorm <= config.newtonGradientTolerance) {
      result.converged = true;
      result.terminationReason = "gradient_tolerance";
      KSK_ADD_FRAME_STAT(stats, "newton.converged_by_gradient.count", 1);
      KSK_ADD_FRAME_STAT(stats, "newton.termination.gradient_tolerance.count", 1);
      break;
    }

    DofBuffer q_before;
    q_before = q.clone();
    double objective_before = 0.0;
    objective_before = evaluateObjective(simulation, dt);

    bool solved_direction = false;
    std::string direction_failure_reason;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/SolveNewtonDirection");
      solved_direction =
          solveNewtonDirection(
              simulation,
              gradient,
              direction,
              stats,
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
    const double direction_norm = direction.norm();
    const double direction_inf_norm = computeDofInfNorm(direction);
    const double scene_scale = computeSceneScale(simulation.scene().geometry);
    const double step_tolerance =
        config.newtonStepTolerance * scene_scale * dt;
    KSK_AVERAGE_FRAME_STAT(stats, "newton.direction_norm.average", direction_norm);
    KSK_AVERAGE_FRAME_STAT(stats,
                           "newton.direction_inf_norm.average",
                           direction_inf_norm);
    KSK_AVERAGE_FRAME_STAT(stats,
                           "newton.scene_scale.average",
                           scene_scale);
    KSK_AVERAGE_FRAME_STAT(stats,
                           "newton.scaled_step_tolerance.average",
                           step_tolerance);
    if (direction_inf_norm < step_tolerance) {
      result.converged = true;
      result.finalStepNorm = direction_inf_norm;
      result.terminationReason = "step_tolerance";
      KSK_ADD_FRAME_STAT(stats, "newton.converged_by_step.count", 1);
      KSK_ADD_FRAME_STAT(stats, "newton.termination.step_tolerance.count", 1);
      break;
    }

    CCDStepBoundResult ccd_step_bound;
    if (config.enableContact) {
      {
        KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/ContactCcdDirectionSearch");
        ccd_step_bound =
            contact_detector_.computeCCDStepBound(simulation, direction);
      }
      KSK_AVERAGE_FRAME_STAT(stats,
                  "newton.ccd_step_upper_bound.average",
                  ccd_step_bound.stepSizeUpperBound);
      KSK_AVERAGE_FRAME_STAT(stats,
                  "newton.swept_candidate_count.average",
                  ccd_step_bound.sweptCandidates.size());
      KSK_MAX_FRAME_STAT(stats,
              "newton.swept_candidate_count.max",
              ccd_step_bound.sweptCandidates.size());
    }

    double directional_derivative = 0.0;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/DirectionalDerivative");
      directional_derivative = gradient.dot(direction);
    }
    KSK_AVERAGE_FRAME_STAT(stats,
                "newton.directional_derivative.average",
                directional_derivative);
    if (directional_derivative >= 0.0) {
      q.copyFrom(q_before);
      synchronizeGeometryFromState(simulation);
      if (config.enableContact) {
        contact_detector_.rebuildActiveContacts(simulation);
      } else {
        simulation.scene().contacts.clear();
      }
      result.finalStepNorm = 0.0;
      result.terminationReason = "non_descent_direction";
      KSK_ADD_FRAME_STAT(stats, "newton.termination.non_descent_direction.count", 1);
      break;
    }

    const double ccd_step_scale =
        std::clamp(config.ccdStepSizeScale, 0.0, 1.0);
    const double ccd_step_limit =
        std::clamp(ccd_step_bound.stepSizeUpperBound, 0.0, 1.0);
    double alpha = ccd_step_limit;
    if (ccd_step_limit < 1.0) {
      alpha *= ccd_step_scale;
    }
    if (!std::isfinite(alpha) || alpha <= 0.0) {
      q.copyFrom(q_before);
      synchronizeGeometryFromState(simulation);
      if (config.enableContact) {
        contact_detector_.rebuildActiveContacts(simulation);
      } else {
        simulation.scene().contacts.clear();
      }
      result.finalStepNorm = 0.0;
      result.terminationReason = "invalid_alpha";
      KSK_ADD_FRAME_STAT(stats, "newton.termination.invalid_alpha.count", 1);
      break;
    }

    bool accepted = false;
    int line_search_attempts = 0;
    for (int line_search = 0;
         line_search < config.maxLineSearchIterations;
         line_search++) {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/LineSearchIteration");
      {
        KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/LineSearch/ApplyTrialState");
        q.assignLinearCombination(q_before, alpha, direction);
      }
      {
        KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/LineSearch/SynchronizeGeometry");
        synchronizeGeometryFromState(simulation);
      }
      if (config.enableContact) {
        {
          KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/LineSearch/RebuildActiveContacts");
          contact_detector_.rebuildActiveContacts(simulation);
        }
      } else {
        simulation.scene().contacts.clear();
      }
      double objective_trial = 0.0;
      {
        KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/LineSearch/EvaluateObjective");
        objective_trial = evaluateObjective(simulation, dt);
      }
      const double armijo_rhs =
          objective_before +
          config.lineSearchArmijo * alpha * directional_derivative;
      line_search_attempts = line_search + 1;
      KSK_AVERAGE_FRAME_STAT(stats, "line_search.alpha.average", alpha);
      KSK_AVERAGE_FRAME_STAT(stats, "line_search.objective.average", objective_trial);
      KSK_AVERAGE_FRAME_STAT(stats, "line_search.armijo.average", armijo_rhs);
      if (std::isfinite(objective_trial) && objective_trial <= armijo_rhs) {
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    KSK_AVERAGE_FRAME_STAT(stats, "line_search.attempts.average", line_search_attempts);
    KSK_MAX_FRAME_STAT(stats, "line_search.attempts.max", line_search_attempts);
    KSK_ADD_FRAME_STAT(stats, "line_search.accepted.count", accepted);

    if (!accepted) {
      q.copyFrom(q_before);
      synchronizeGeometryFromState(simulation);
      if (config.enableContact) {
        contact_detector_.rebuildActiveContacts(simulation);
      } else {
        simulation.scene().contacts.clear();
      }
      result.finalStepNorm = 0.0;
      result.terminationReason = "line_search_failed";
      KSK_ADD_FRAME_STAT(stats, "newton.termination.line_search_failed.count", 1);
      break;
    }

    {
      KSK_PROFILE_SCOPE("GlobalSolver/Step/NewtonIteration/StepNorm");
      result.finalStepNorm = alpha * direction_norm;
    }
    KSK_AVERAGE_FRAME_STAT(stats, "newton.accepted_alpha.average", alpha);
    KSK_AVERAGE_FRAME_STAT(stats, "newton.step_norm.average", result.finalStepNorm);
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
  KSK_RECORD_FRAME_STAT(stats, "step.result.final_gradient_norm", result.finalGradientNorm);
  KSK_RECORD_FRAME_STAT(stats, "step.result.final_step_norm", result.finalStepNorm);
  KSK_RECORD_FRAME_STAT(stats, "step.result.converged", result.converged);
  if (!result.converged) {
    spdlog::warn(
        "Global solver did not converge: reason={} iterations={} grad={} "
        "step_norm={} dt={} contacts={} scalar_count={} geometry_points={}",
        result.terminationReason,
        result.iterations,
        result.finalGradientNorm,
        result.finalStepNorm,
        dt,
        simulation.scene().contacts.size(),
        scalar_count,
        geometry_point_count_);
  }

  return result;
}

double GlobalGaussNewtonSolver::evaluateObjective(
    SimulationContext& simulation,
    double dt)
{
  SIM_PROFILE_SCOPE_COLOR("GlobalSolver/EvaluateObjective",
                          ksk::core::profiler_colors::kPurple);
  double objective = 0.0;
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
      objective +=
          subsystem->evaluateObjective(q.slice(range), qdot.slice(range), dt);
    }
    {
      KSK_PROFILE_SCOPE("GlobalSolver/EvaluateObjective/UpdateGeometry");
      subsystem->updateGeometry(simulation.scene().geometry);
    }
  }
  if (simulation.scene().solverConfig.enableContact) {
    KSK_PROFILE_SCOPE("GlobalSolver/EvaluateObjective/ContactEnergy");
    objective += computeContactEnergy(simulation.scene().geometry,
                                      simulation.scene().contacts);
  }
  return objective;
}

void GlobalGaussNewtonSolver::assembleContactGradient(
    SimulationContext& simulation,
    DofBuffer& gradient)
{
  SIM_PROFILE_SCOPE_COLOR("GlobalSolver/AssembleContactGradient",
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
    KSK_PROFILE_SCOPE("GlobalSolver/AssembleContactGradient/ComputeGeometryGradient");
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
      if (simulation.scene().geometry.pointOwner(point).subsystem != subsystem_id) {
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

bool GlobalGaussNewtonSolver::solveNewtonDirection(
    SimulationContext& simulation,
    const DofBuffer& gradient,
    DofBuffer& direction,
    GlobalSolverStatsCollector* stats,
    std::string* failureReason)
{
  SIM_PROFILE_SCOPE_COLOR("GlobalSolver/SolveNewtonDirection",
                          ksk::core::profiler_colors::kBlue);
  const GlobalSolverConfig& config = simulation.scene().solverConfig;
  const int scalar_count = simulation.scene().dofs.totalScalars;
  if (scalar_count == 0) {
    direction.setZero();
    KSK_ADD_FRAME_STAT(stats, "pcg.mode.zero_scalar_system.count", 1);
    return true;
  }

  if (!config.enableContact || simulation.scene().contacts.empty()) {
    KSK_ADD_FRAME_STAT(stats, "pcg.mode.block_diagonal_subsystems.count", 1);
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/BlockDiagonalSubsystems");
    const bool solved =
        solveBlockDiagonalSubsystemDirection(simulation, gradient, direction);
    if (!solved && failureReason != nullptr) {
      *failureReason = "block_diagonal_direction_failed";
    }
    return solved;
  }
  KSK_ADD_FRAME_STAT(stats, "pcg.mode.pcg.count", 1);

  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/InitializePcg");
    direction.setZero();
  }

  DofBuffer residual;
  double rhs_norm = 0.0;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/BuildResidual");
    residual = createDofBufferLike(gradient, scalar_count);
    residual.assignScaled(-1.0, gradient);
    rhs_norm = residual.norm();
  }
  if (rhs_norm == 0.0) {
    return true;
  }

  DofBuffer z = createDofBufferLike(gradient, scalar_count);
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/InitialPreconditioner");
    applyPreconditioner(simulation, residual, z);
  }

  DofBuffer search;
  double rz_old = 0.0;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/InitialDotProducts");
    search = z.clone();
    rz_old = residual.dot(z);
  }
  if (!std::isfinite(rz_old) || rz_old <= 0.0) {
    warnInvalidPreconditionerDotBreakdown(simulation,
                                          residual,
                                          z,
                                          "initial",
                                          0,
                                          rz_old,
                                          rhs_norm,
                                          rhs_norm);

    if (failureReason != nullptr)
      *failureReason = "pcg_invalid_initial_rz";

    return false;
  }

  const double tolerance = config.pcgTolerance * std::max(1.0, rhs_norm);
  DofBuffer matrix_search = createDofBufferLike(gradient, scalar_count);
  auto record_verified_residual = [&](const char* label)
  {
    if (stats == nullptr) {
      return;
    }
    DofBuffer verification = createDofBufferLike(gradient, scalar_count);
    applyGlobalMatrix(simulation, direction, verification);
    verification.addScaled(1.0, gradient);
    const double verified_residual_norm = verification.norm();
    KSK_AVERAGE_FRAME_STAT(stats,
                           std::string("pcg.") + label +
                               ".verified_ax_plus_g_norm.average",
                           verified_residual_norm);
    KSK_AVERAGE_FRAME_STAT(stats,
                           std::string("pcg.") + label +
                               ".relative_verified_ax_plus_g_norm.average",
                           verified_residual_norm / rhs_norm);
  };

  for (int iteration = 0; iteration < config.maxPcgIterations; ++iteration) {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/PcgIteration");
    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/ApplyGlobalMatrix");
      applyGlobalMatrix(simulation, search, matrix_search);
    }

    double denominator = 0.0;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/DenominatorDot");
      denominator = search.dot(matrix_search);
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) {
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.iterations.average", iteration + 1);
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.invalid_denominator.average", denominator);
      KSK_ADD_FRAME_STAT(stats, "pcg.failure.invalid_denominator.count", 1);
      if (failureReason != nullptr) {
        *failureReason = "pcg_invalid_denominator";
      }
      return false;
    }

    const double alpha = rz_old / denominator;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/UpdateDirectionResidual");
      direction.addScaled(alpha, search);
      residual.addScaled(-alpha, matrix_search);
    }

    double residual_norm = 0.0;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/ResidualNorm");
      residual_norm = residual.norm();
    }
    KSK_AVERAGE_FRAME_STAT(stats, "pcg.residual_norm.average", residual_norm);
    KSK_AVERAGE_FRAME_STAT(stats,
                "pcg.relative_residual_norm.average",
                residual_norm / rhs_norm);
    KSK_AVERAGE_FRAME_STAT(stats, "pcg.denominator.average", denominator);
    KSK_AVERAGE_FRAME_STAT(stats, "pcg.alpha.average", alpha);
    if (residual_norm <= tolerance) {
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.iterations.average", iteration + 1);
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.final_residual_norm.average", residual_norm);
      KSK_ADD_FRAME_STAT(stats, "pcg.converged.count", 1);
      record_verified_residual("converged");
      return true;
    }

    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/ApplyPreconditioner");
      applyPreconditioner(simulation, residual, z);
    }
    double rz_new = 0.0;
    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/RzDot");
      rz_new = residual.dot(z);
    }
    if (!std::isfinite(rz_new) || rz_new <= 0.0) {
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.iterations.average", iteration + 1);
      KSK_AVERAGE_FRAME_STAT(stats, "pcg.invalid_rz.average", rz_new);
      KSK_ADD_FRAME_STAT(stats, "pcg.failure.invalid_rz.count", 1);
      warnInvalidPreconditionerDotBreakdown(simulation,
                                            residual,
                                            z,
                                            "iteration",
                                            iteration + 1,
                                            rz_new,
                                            residual_norm,
                                            rhs_norm);
      if (failureReason != nullptr) {
        *failureReason = "pcg_invalid_rz";
      }
      return false;
    }

    const double beta = rz_new / rz_old;
    KSK_AVERAGE_FRAME_STAT(stats, "pcg.rz.average", rz_new);
    KSK_AVERAGE_FRAME_STAT(stats, "pcg.beta.average", beta);
    {
      KSK_PROFILE_SCOPE("GlobalSolver/SolveNewtonDirection/Pcg/UpdateSearch");
      search.assignLinearCombination(z, beta, search);
    }
    rz_old = rz_new;
  }

  const double final_residual_norm = residual.norm();
  const bool converged = final_residual_norm <= tolerance;
  KSK_AVERAGE_FRAME_STAT(stats, "pcg.iterations.average", config.maxPcgIterations);
  KSK_AVERAGE_FRAME_STAT(stats, "pcg.final_residual_norm.average", final_residual_norm);
  KSK_ADD_FRAME_STAT(stats, "pcg.converged.count", converged);
  record_verified_residual("max_iterations");
  if (!converged && failureReason != nullptr) {
    *failureReason = "pcg_max_iterations";
  }
  return converged;
}

bool GlobalGaussNewtonSolver::solveSingleSubsystemDirection(
    SimulationContext& simulation,
    const DofBuffer& gradient,
    DofBuffer& direction)
{
  KSK_PROFILE_SCOPE("GlobalSolver/SolveSingleSubsystemDirection");
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveSingleSubsystemDirection/SetZero");
    direction.setZero();
  }

  DofBuffer residual;
  double residual_norm = 0.0;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveSingleSubsystemDirection/BuildResidual");
    residual = createDofBufferLike(gradient, simulation.scene().dofs.totalScalars);
    residual.assignScaled(-1.0, gradient);
    residual_norm = residual.norm();
  }
  if (residual_norm == 0.0) {
    return true;
  }

  const DofRange range = simulation.subsystems().front()->dofRange();
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveSingleSubsystemDirection/SolveLocalSystem");
    simulation.subsystems().front()->solveLocalSystem(residual.slice(range),
                                                      direction.slice(range));
  }
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveSingleSubsystemDirection/DirectionNorm");
    return std::isfinite(direction.norm());
  }
}

bool GlobalGaussNewtonSolver::solveBlockDiagonalSubsystemDirection(
    SimulationContext& simulation,
    const DofBuffer& gradient,
    DofBuffer& direction)
{
  KSK_PROFILE_SCOPE("GlobalSolver/SolveBlockDiagonalSubsystemDirection");
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveBlockDiagonalSubsystemDirection/SetZero");
    direction.setZero();
  }

  DofBuffer residual;
  double residual_norm = 0.0;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveBlockDiagonalSubsystemDirection/BuildResidual");
    residual = createDofBufferLike(gradient, simulation.scene().dofs.totalScalars);
    residual.assignScaled(-1.0, gradient);
    residual_norm = residual.norm();
  }
  if (residual_norm == 0.0) {
    return true;
  }

  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveBlockDiagonalSubsystemDirection/SolveLocalSystems");
    for (const auto& subsystem : simulation.subsystems()) {
      const DofRange range = subsystem->dofRange();
      subsystem->solveLocalSystem(residual.slice(range),
                                  direction.slice(range));
    }
  }

  {
    KSK_PROFILE_SCOPE("GlobalSolver/SolveBlockDiagonalSubsystemDirection/DirectionNorm");
    return std::isfinite(direction.norm());
  }
}

void GlobalGaussNewtonSolver::applyGlobalMatrix(SimulationContext& simulation,
                                                const DofBuffer& x,
                                                DofBuffer& y)
{
  SIM_PROFILE_SCOPE_COLOR("GlobalSolver/ApplyGlobalMatrix",
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
    applyGlobalContactHessian(simulation, x, y);
  }
}

void GlobalGaussNewtonSolver::applyGlobalContactHessian(
    SimulationContext& simulation,
    const DofBuffer& x,
    DofBuffer& y)
{
  SIM_PROFILE_SCOPE_COLOR("GlobalSolver/ApplyGlobalContactHessian",
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
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyGlobalContactHessian/CreateGeometryDirection");
    geometry_direction =
        createGeometryDirectionBuffer(x, simulation.scene().geometry.pointCount());
  }
  for (const auto& subsystem : simulation.subsystems()) {
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyGlobalContactHessian/MapDirectionToGeometry");
    subsystem->mapLocalDirectionToGeometry(
        x.slice(subsystem->dofRange()), geometry_direction.view());
  }

  ContactPotentialGradient contact_hessian_product;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyGlobalContactHessian/ComputeGeometryHessianProduct");
    contact_hessian_product =
        computeContactHessianProductWrtGeometry(
            simulation.scene().geometry,
            contacts,
            geometry_direction.view().asConst());
  }
  ContactPotentialGradient contact_gradient;
  {
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyGlobalContactHessian/ComputeGeometryGradient");
    contact_gradient =
        computeContactGradientWrtGeometry(simulation.scene().geometry, contacts);
  }
  if (contact_hessian_product.points.empty() &&
      contact_gradient.points.empty()) {
    return;
  }

  for (const auto& subsystem : simulation.subsystems()) {
    KSK_PROFILE_SCOPE("GlobalSolver/ApplyGlobalContactHessian/ScatterSubsystem");
    std::vector<PointIdx> gradient_points;
    std::vector<glm::dvec3> gradient_values;
    std::vector<PointIdx> product_points;
    std::vector<glm::dvec3> product_values;
    const SubsystemIdx subsystem_id = subsystem->id();

    for (int i = 0; i < contact_gradient.points.size(); ++i) {
      const PointIdx point = contact_gradient.points[i];
      if (simulation.scene().geometry.pointOwner(point).subsystem != subsystem_id)
        continue;
      gradient_points.push_back(point);
      gradient_values.push_back(contact_gradient.gradient.cpu()[i]);
    }

    for (int i = 0; i < contact_hessian_product.points.size(); ++i) {
      const PointIdx point = contact_hessian_product.points[i];
      if (simulation.scene().geometry.pointOwner(point).subsystem != subsystem_id)
        continue;
      product_points.push_back(point);
      product_values.push_back(contact_hessian_product.gradient.cpu()[i]);
    }

    if (gradient_points.empty() && product_points.empty()) {
      continue;
    }
    subsystem->applyContactGeometryHessianProduct(
        gradient_points,
        GeometryBuffer::FromCPU(std::move(gradient_values)).view().asConst(),
        product_points,
        GeometryBuffer::FromCPU(std::move(product_values)).view().asConst(),
        x.slice(subsystem->dofRange()),
        y.slice(subsystem->dofRange()));
  }
}

void GlobalGaussNewtonSolver::applyPreconditioner(
    SimulationContext& simulation,
    const DofBuffer& residual,
    DofBuffer& z)
{
  SIM_PROFILE_SCOPE_COLOR("GlobalSolver/ApplyPreconditioner",
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

SimulationRunner::SimulationRunner(SimulationContext simulation, double timeStep)
    : simulation_(std::move(simulation))
    , time_step_(timeStep)
{
}

RuntimeStepResult SimulationRunner::step(GlobalSolverStatsCollector* stats)
{
  SIM_PROFILE_SCOPE_COLOR("SimulationRunner/Step",
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

RuntimeStepResult SimulationRunner::run(int steps, GlobalSolverStatsCollector* stats)
{
  for (int step_index = 0; step_index < steps; ++step_index) {
    last_step_ = step(stats);
  }
  return last_step_;
}

SimulationRunner buildSimulationRunner(const RuntimeSceneDesc& scene)
{
  return {buildSimulation(scene), scene.timeStep};
}

}  // namespace ksk::runtime
