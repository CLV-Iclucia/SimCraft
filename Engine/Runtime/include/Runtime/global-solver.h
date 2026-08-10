#pragma once

#include <Runtime/contact-detector.h>
#include <Runtime/global-pd-solver.h>
#include <Runtime/newton-ipc-subsystem-api.h>
#include <Runtime/runtime-scene.h>
#include <Runtime/simulation-context.h>
#include <Runtime/solver-result.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ksk::runtime {

struct ContactGeometryHessian;

// Global Newton + IPC solver.
//
// Drives the nonlinear IPC optimization through the cached
// NewtonIPCSubsystemSolver adapters. The base Subsystem lifecycle
// (readState/beginStep/updateGeometry/acceptStep) stays solver-independent and
// is still called directly; energy/operator/contact work goes through the
// solver-specific APIs.
class GlobalNewtonIpcSolver {
 public:
  // Creates/caches the per-subsystem Newton IPC API adapters. Requires every
  // subsystem to support Newton IPC (throws otherwise).
  void prepare(SimulationContext& simulation);
  RuntimeStepResult step(SimulationContext& simulation, double dt,
                         double time = 0.0,
                         GlobalSolverStatsCollector* stats = nullptr);

  [[nodiscard]] int scalarCount() const noexcept { return scalar_count_; }
  [[nodiscard]] int geometryPointCount() const noexcept {
    return geometry_point_count_;
  }

 private:
  [[nodiscard]] NewtonIPCSubsystemSolver* apiFor(SubsystemIdx id) noexcept;
  [[nodiscard]] const NewtonIPCSubsystemSolver* apiFor(
      SubsystemIdx id) const noexcept;

  [[nodiscard]] double evaluateSubsystemObjectiveAndUpdateGeometry(
      SimulationContext& simulation, double dt);
  void assembleContactGradient(SimulationContext& simulation,
                               DofBuffer& gradient);
  void assembleContactFrictionGradient(SimulationContext& simulation,
                                       DofBuffer& gradient);
  [[nodiscard]] bool solveNewtonDirection(
      SimulationContext& simulation, const DofBuffer& gradient,
      DofBuffer& direction, GlobalSolverStatsCollector* stats,
      const ContactGeometryHessian* contactHessian = nullptr,
      std::string* failureReason = nullptr);
  [[nodiscard]] bool solveDecoupledDirection(
      SimulationContext& simulation, const DofBuffer& gradient,
      DofBuffer& direction, GlobalSolverStatsCollector* stats,
      const ContactGeometryHessian* contactHessian = nullptr,
      std::string* failureReason = nullptr);
  [[nodiscard]] bool solveCoupledDirection(
      SimulationContext& simulation, const DofBuffer& gradient,
      DofBuffer& direction, GlobalSolverStatsCollector* stats,
      const ContactGeometryHessian* contactHessian = nullptr,
      std::string* failureReason = nullptr);
  [[nodiscard]] bool solveSingleSubsystemDirection(
      SimulationContext& simulation, const DofBuffer& gradient,
      DofBuffer& direction);
  [[nodiscard]] bool solveBlockDiagonalSubsystemDirection(
      SimulationContext& simulation, const DofBuffer& gradient,
      DofBuffer& direction);
  void applyGlobalMatrix(
      SimulationContext& simulation, const DofBuffer& x, DofBuffer& y,
      const ContactGeometryHessian* contactHessian = nullptr);
  void applyGlobalContactHessian(
      SimulationContext& simulation, const DofBuffer& x, DofBuffer& y,
      const ContactGeometryHessian* contactHessian = nullptr);
  void applyPreconditioner(SimulationContext& simulation,
                           const DofBuffer& residual, DofBuffer& z);

  int scalar_count_ = 0;
  int geometry_point_count_ = 0;
  ContactDetector contact_detector_;
  std::vector<std::unique_ptr<NewtonIPCSubsystemSolver>> subsystem_apis_;
};

class SimulationRunner {
 public:
  SimulationRunner() = default;
  SimulationRunner(SimulationContext simulation, double timeStep);

  [[nodiscard]] RuntimeStepResult step(
      GlobalSolverStatsCollector* stats = nullptr);
  [[nodiscard]] RuntimeStepResult run(
      int steps, GlobalSolverStatsCollector* stats = nullptr);

  [[nodiscard]] SimulationContext& simulation() noexcept { return simulation_; }
  [[nodiscard]] const SimulationContext& simulation() const noexcept {
    return simulation_;
  }
  [[nodiscard]] const RuntimeStepResult& lastStep() const noexcept {
    return last_step_;
  }
  [[nodiscard]] double time() const noexcept { return time_; }
  [[nodiscard]] double timeStep() const noexcept { return time_step_; }
  [[nodiscard]] int stepsCompleted() const noexcept { return steps_completed_; }

 private:
  SimulationContext simulation_;
  GlobalNewtonIpcSolver newton_solver_;
  GlobalProjectiveDynamicsSolver pd_solver_;
  RuntimeStepResult last_step_;
  double time_step_ = 1.0 / 60.0;
  double time_ = 0.0;
  int steps_completed_ = 0;
};

[[nodiscard]] SimulationRunner buildSimulationRunner(
    const RuntimeSceneDesc& scene);

}  // namespace ksk::runtime

