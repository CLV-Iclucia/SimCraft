#pragma once

// Newton + IPC adapter for the FEM subsystem.
//
// The adapter forwards every Newton IPC operation to the FEM subsystem's
// Newton methods (which in turn drive the CPU backend). It is a thin, cached
// adapter owned by the global Newton solver.

#include <Runtime/newton-ipc-subsystem-api.h>

#include <span>
#include <vector>

namespace ksk::engine::fem {

class FEMSubsystem;

class FEMNewtonIpcApi final : public runtime::NewtonIPCSubsystemSolver {
 public:
  explicit FEMNewtonIpcApi(FEMSubsystem& subsystem) : subsystem_(subsystem) {}

  [[nodiscard]] runtime::SubsystemIdx subsystemId() const noexcept override;
  [[nodiscard]] runtime::DofRange dofRange() const noexcept override;

  [[nodiscard]] double evaluateObjective(runtime::ConstDofView q,
                                         runtime::ConstDofView qdot,
                                         double dt) override;

  void updateInternalConstraints(double time, double dt) override;
  void prepareLocalOperator(double dt) override;
  void assembleLocalGradient(runtime::DofView g) const override;
  void applyLocalMatrix(runtime::ConstDofView x,
                        runtime::DofView y) const override;
  void appendLocalMatrixEntries(
      std::vector<runtime::SparseMatrixEntry>& entries) const override;
  void solveLocalSystem(runtime::ConstDofView b,
                        runtime::DofView x) const override;

  void mapLocalDirectionToGeometry(runtime::ConstDofView localDq,
                                   runtime::GeometryView globalDx) const override;
  void applyInternalContacts(runtime::ContactStencils contacts) override;
  void scatterContactGradient(
      std::span<const runtime::PointIdx> points,
      runtime::ConstGeometryView pointGradient,
      runtime::DofView g) const override;
  void applyContactGeometryHessianProduct(
      std::span<const runtime::PointIdx> productPoints,
      runtime::ConstGeometryView pointHessianProduct,
      runtime::DofView localY) const override;
  void applyInternalContactHessian(runtime::ConstDofView localDq,
                                   runtime::DofView localY) const override;

 private:
  FEMSubsystem& subsystem_;
};

}  // namespace ksk::engine::fem
