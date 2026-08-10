#include <FEM/solver/fem-newton-ipc-api.h>

#include <FEM/fem-subsystem.h>

namespace ksk::engine::fem {

runtime::SubsystemIdx FEMNewtonIpcApi::subsystemId() const noexcept
{
  return subsystem_.id();
}

runtime::DofRange FEMNewtonIpcApi::dofRange() const noexcept
{
  return subsystem_.dofRange();
}

double FEMNewtonIpcApi::evaluateObjective(runtime::ConstDofView q,
                                          runtime::ConstDofView qdot,
                                          double dt)
{
  return subsystem_.evaluateObjective(q, qdot, dt);
}

void FEMNewtonIpcApi::updateInternalConstraints(double time, double dt)
{
  subsystem_.updateInternalConstraints(time, dt);
}

void FEMNewtonIpcApi::prepareLocalOperator(double dt)
{
  subsystem_.prepareLocalOperator(dt);
}

void FEMNewtonIpcApi::assembleLocalGradient(runtime::DofView g) const
{
  subsystem_.assembleLocalGradient(g);
}

void FEMNewtonIpcApi::applyLocalMatrix(runtime::ConstDofView x,
                                       runtime::DofView y) const
{
  subsystem_.applyLocalMatrix(x, y);
}

void FEMNewtonIpcApi::appendLocalMatrixEntries(
    std::vector<runtime::SparseMatrixEntry>& entries) const
{
  subsystem_.appendLocalMatrixEntries(entries);
}

void FEMNewtonIpcApi::solveLocalSystem(runtime::ConstDofView b,
                                       runtime::DofView x) const
{
  subsystem_.solveLocalSystem(b, x);
}

void FEMNewtonIpcApi::mapLocalDirectionToGeometry(
    runtime::ConstDofView localDq, runtime::GeometryView globalDx) const
{
  subsystem_.mapLocalDirectionToGeometry(localDq, globalDx);
}

void FEMNewtonIpcApi::applyInternalContacts(runtime::ContactStencils contacts)
{
  subsystem_.applyInternalContacts(std::move(contacts));
}

void FEMNewtonIpcApi::scatterContactGradient(
    std::span<const runtime::PointIdx> points,
    runtime::ConstGeometryView pointGradient, runtime::DofView g) const
{
  subsystem_.scatterContactGradient(points, pointGradient, g);
}

void FEMNewtonIpcApi::applyContactGeometryHessianProduct(
    std::span<const runtime::PointIdx> productPoints,
    runtime::ConstGeometryView pointHessianProduct,
    runtime::DofView localY) const
{
  subsystem_.applyContactGeometryHessianProduct(productPoints,
                                                pointHessianProduct, localY);
}

void FEMNewtonIpcApi::applyInternalContactHessian(
    runtime::ConstDofView localDq, runtime::DofView localY) const
{
  subsystem_.applyInternalContactHessian(localDq, localY);
}

}  // namespace ksk::engine::fem
