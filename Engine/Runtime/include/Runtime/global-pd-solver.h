#pragma once

// Global Projective Dynamics solver.
//
// PD is an alternative time-step formulation to Newton + IPC. It runs pure
// elastic local/global (L-G) iterations with a constant matrix M + h^2 L
// (assembled once, Cholesky-factored once) and resolves collisions as a
// separate post-processing phase that directly corrects q.
//
// Initial PD support is scoped to the FEM subsystem: only FEM returns a
// non-null ProjectiveDynamicsSubsystemSolver. DER and ABD intentionally do not
// advertise PD support in the first phase.

#include <Runtime/pd-collision.h>
#include <Runtime/projective-dynamics-subsystem-api.h>
#include <Runtime/simulation-context.h>
#include <Runtime/solver-result.h>

#include <memory>
#include <vector>

namespace ksk::runtime {

class GlobalProjectiveDynamicsSolver {
 public:
  GlobalProjectiveDynamicsSolver();
  ~GlobalProjectiveDynamicsSolver();
  GlobalProjectiveDynamicsSolver(GlobalProjectiveDynamicsSolver&&) noexcept;
  GlobalProjectiveDynamicsSolver&
  operator=(GlobalProjectiveDynamicsSolver&&) noexcept;
  GlobalProjectiveDynamicsSolver(const GlobalProjectiveDynamicsSolver&) =
      delete;
  GlobalProjectiveDynamicsSolver&
  operator=(const GlobalProjectiveDynamicsSolver&) = delete;

  // Creates/caches per-subsystem PD API adapters. Requires every subsystem to
  // support ProjectiveDynamics (throws otherwise).
  void prepare(SimulationContext& simulation);
  RuntimeStepResult step(SimulationContext& simulation,
                         double dt,
                         double time = 0.0,
                         GlobalSolverStatsCollector* stats = nullptr);

  [[nodiscard]] int scalarCount() const noexcept;
  [[nodiscard]] int geometryPointCount() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ksk::runtime
