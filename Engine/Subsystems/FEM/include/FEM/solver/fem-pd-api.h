#pragma once

// Projective Dynamics adapter for the FEM subsystem.
//
// PD support is intentionally scoped to the FEM subsystem in the first phase.
// This adapter is the structural skeleton: the method signatures are final and
// match the PD protocol, but the local projection / global RHS / operator
// assembly are placeholders until the FEM PD backend is implemented. This
// keeps the adapter compilable and extensible without shipping half-broken
// physics.

#include <Runtime/projective-dynamics-subsystem-api.h>

#include <vector>

namespace ksk::engine::fem {

class FEMSubsystem;

class FEMProjectiveDynamicsApi final
    : public runtime::ProjectiveDynamicsSubsystemSolver {
 public:
  explicit FEMProjectiveDynamicsApi(FEMSubsystem& subsystem)
      : subsystem_(subsystem)
  {
  }

  [[nodiscard]] runtime::SubsystemIdx subsystemId() const noexcept override;
  [[nodiscard]] runtime::DofRange dofRange() const noexcept override;

  void updateProjectiveConstraints(double time, double dt) override;
  void prepareProjectiveOperator(double dt) override;

  void projectLocalStep(runtime::ConstDofView q,
                        runtime::ConstDofView qdot,
                        double dt) override;
  void assembleProjectiveRhs(runtime::DofView rhs) const override;

  void applyProjectiveMatrix(runtime::ConstDofView x,
                             runtime::DofView y) const override;
  void appendProjectiveMatrixEntries(
      std::vector<runtime::SparseMatrixEntry>& entries) const override;
  void solveProjectiveLocalSystem(runtime::ConstDofView b,
                                  runtime::DofView x) const override;

 private:
  FEMSubsystem& subsystem_;
};

}  // namespace ksk::engine::fem
