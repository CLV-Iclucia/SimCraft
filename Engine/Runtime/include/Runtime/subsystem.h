#pragma once

#include <Runtime/contact-table.h>
#include <Runtime/buffers.h>
#include <Runtime/dof-layout.h>
#include <Runtime/global-geometry-manager.h>

#include <memory>
#include <span>
#include <vector>

namespace ksk::runtime {

class NewtonIPCSubsystemSolver;
class ProjectiveDynamicsSubsystemSolver;

// Base interface for a simulation subsystem.
//
// `Subsystem` answers: "what state and geometry do you own?" It only exposes
// solver-independent lifecycle methods. Solver-specific operations are exposed
// through per-solver API objects created by createNewtonIPCSolver() /
// createProjectiveDynamicsSolver().
class Subsystem {
 public:
  virtual ~Subsystem() = default;

  [[nodiscard]] virtual SubsystemIdx id() const noexcept = 0;
  [[nodiscard]] virtual DofRange dofRange() const noexcept = 0;

  virtual void declareGeometry(GlobalGeometryManager& geometry) = 0;
  virtual void writeState(DofView q, DofView qdot) const = 0;
  virtual void readState(ConstDofView q, ConstDofView qdot) = 0;
  virtual void beginStep(ConstDofView q,
                         ConstDofView qdot,
                         double dt) = 0;
  virtual void acceptStep(ConstDofView q,
                          DofView qdot,
                          double dt) = 0;
  virtual void updateGeometry(GlobalGeometryManager& geometry) const = 0;

  // Solver API factories. Unsupported solvers return nullptr.
  // Defined out-of-line (Runtime/src/subsystem.cc) so the returned
  // std::unique_ptr has the complete solver API type at destruction time.
  [[nodiscard]] virtual std::unique_ptr<NewtonIPCSubsystemSolver>
  createNewtonIPCSolver();

  [[nodiscard]] virtual std::unique_ptr<ProjectiveDynamicsSubsystemSolver>
  createProjectiveDynamicsSolver();
};

}  // namespace ksk::runtime

