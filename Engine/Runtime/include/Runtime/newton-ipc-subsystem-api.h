#pragma once

// Solver-facing contract for the Newton + IPC global solver family.
//
// A global Newton solver asks every subsystem for one of these adapters
// (Subsystem::createNewtonIPCSolver). Once cached, the global solver calls the
// solver-specific operations through this interface and no longer calls
// Newton-specific methods on the base Subsystem interface.
//
// The concrete subsystem implements this interface directly or returns an
// adapter object that forwards into its internal backend.

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/global-geometry-manager.h>
#include <Runtime/types.h>

#include <span>
#include <vector>

namespace ksk::runtime {

class NewtonIPCSubsystemSolver {
 public:
  virtual ~NewtonIPCSubsystemSolver() = default;

  [[nodiscard]] virtual SubsystemIdx subsystemId() const noexcept = 0;
  [[nodiscard]] virtual DofRange dofRange() const noexcept = 0;

  // Full time-step objective (inertia + elasticity + internal constraints).
  [[nodiscard]] virtual double evaluateObjective(ConstDofView q,
                                                 ConstDofView qdot,
                                                 double dt) = 0;

  virtual void updateInternalConstraints(double time, double dt) = 0;
  virtual void prepareLocalOperator(double dt) = 0;
  virtual void assembleLocalGradient(DofView g) const = 0;
  virtual void applyLocalMatrix(ConstDofView x, DofView y) const = 0;
  virtual void appendLocalMatrixEntries(
      std::vector<SparseMatrixEntry>& entries) const = 0;
  virtual void solveLocalSystem(ConstDofView b, DofView x) const = 0;

  // IPC contact derivative mapping between geometry space and local DOF space.
  virtual void mapLocalDirectionToGeometry(ConstDofView localDq,
                                           GeometryView globalDx) const = 0;
  virtual void applyInternalContacts(ContactStencils contacts) = 0;
  virtual void scatterContactGradient(
      std::span<const PointIdx> points,
      ConstGeometryView pointGradient,
      DofView g) const = 0;
  virtual void applyContactGeometryHessianProduct(
      std::span<const PointIdx> productPoints,
      ConstGeometryView pointHessianProduct,
      DofView localY) const = 0;
  virtual void applyInternalContactHessian(ConstDofView localDq,
                                           DofView localY) const = 0;
};

}  // namespace ksk::runtime
