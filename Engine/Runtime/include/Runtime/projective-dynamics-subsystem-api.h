#pragma once

// Solver-facing contract for the Projective Dynamics global solver family.
//
// A global PD solver asks every subsystem for one of these adapters
// (Subsystem::createProjectiveDynamicsSolver). Once cached, the global solver
// drives the local/global (L-G) iterations through this interface.
//
// Note on contacts: this API deliberately has NO contact-related methods.
// PD collision handling is post-processing: the global PD solver runs pure
// elastic L-G iterations, then a separate resolve phase performs DCD +
// positional corrections directly on q. Subsystems only provide elastic
// operators. This keeps the PD subsystem API minimal and the collision
// strategy fully replaceable without touching subsystem code.

#include <Runtime/buffers.h>
#include <Runtime/types.h>

#include <vector>

namespace ksk::runtime {

class ProjectiveDynamicsSubsystemSolver {
 public:
  virtual ~ProjectiveDynamicsSubsystemSolver() = default;

  [[nodiscard]] virtual SubsystemIdx subsystemId() const noexcept = 0;
  [[nodiscard]] virtual DofRange dofRange() const noexcept = 0;

  // Per-step elastic operator setup (called once per time step, before the
  // L-G iterations begin).
  virtual void updateProjectiveConstraints(double time, double dt) = 0;
  virtual void prepareProjectiveOperator(double dt) = 0;

  // --- Local/global iterations ---
  // Local step: project each element to the closest constraint-compatible
  // configuration, storing the projected targets internally.
  virtual void projectLocalStep(ConstDofView q,
                                ConstDofView qdot,
                                double dt) = 0;
  // Global step right-hand side assembly: adds the per-subsystem rhs into the
  // provided buffer (assumed zeroed by the caller).
  virtual void assembleProjectiveRhs(DofView rhs) const = 0;

  // --- Global matrix (M + h^2 L) operator ---
  virtual void applyProjectiveMatrix(ConstDofView x, DofView y) const = 0;
  virtual void appendProjectiveMatrixEntries(
      std::vector<SparseMatrixEntry>& entries) const = 0;
  virtual void solveProjectiveLocalSystem(ConstDofView b,
                                          DofView x) const = 0;

  // Optional residual estimate for early L-G termination.
  [[nodiscard]] virtual double evaluateProjectiveResidual(
      ConstDofView q,
      double dt) const
  {
    return 0.0;
  }
};

}  // namespace ksk::runtime
