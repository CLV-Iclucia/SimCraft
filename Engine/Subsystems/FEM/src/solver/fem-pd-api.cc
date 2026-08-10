#include <FEM/solver/fem-pd-api.h>

#include <FEM/fem-subsystem.h>

namespace ksk::engine::fem {

runtime::SubsystemIdx FEMProjectiveDynamicsApi::subsystemId() const noexcept
{
  return subsystem_.id();
}

runtime::DofRange FEMProjectiveDynamicsApi::dofRange() const noexcept
{
  return subsystem_.dofRange();
}

void FEMProjectiveDynamicsApi::updateProjectiveConstraints(double time,
                                                           double dt)
{
  // TODO(PD): cache (time, dt) and refresh projective constraint targets from
  // the FEM scene when the FEM PD backend is implemented.
  (void)time;
  (void)dt;
}

void FEMProjectiveDynamicsApi::prepareProjectiveOperator(double dt)
{
  // TODO(PD): assemble the FEM contribution to the constant matrix
  // M + h^2 L (mass diagonal plus elastic Laplacian) here.
  (void)dt;
}

void FEMProjectiveDynamicsApi::projectLocalStep(runtime::ConstDofView q,
                                                runtime::ConstDofView qdot,
                                                double dt)
{
  // TODO(PD): project each FEM element to its closest constraint-compatible
  // configuration and store the projected targets for assembleProjectiveRhs.
  (void)q;
  (void)qdot;
  (void)dt;
}

void FEMProjectiveDynamicsApi::assembleProjectiveRhs(runtime::DofView rhs) const
{
  // TODO(PD): accumulate the FEM local RHS (inertia + projected targets).
  (void)rhs;
}

void FEMProjectiveDynamicsApi::applyProjectiveMatrix(runtime::ConstDofView x,
                                                     runtime::DofView y) const
{
  // TODO(PD): apply the FEM block of M + h^2 L to x and add into y.
  (void)x;
  (void)y;
}

void FEMProjectiveDynamicsApi::appendProjectiveMatrixEntries(
    std::vector<runtime::SparseMatrixEntry>& entries) const
{
  // TODO(PD): append the FEM block of M + h^2 L as sparse entries.
  (void)entries;
}

void FEMProjectiveDynamicsApi::solveProjectiveLocalSystem(
    runtime::ConstDofView b, runtime::DofView x) const
{
  // TODO(PD): optional per-subsystem local solve (used by a decoupled PD
  // variant). Leave x unchanged for the coupled path.
  (void)b;
  (void)x;
}

}  // namespace ksk::engine::fem
