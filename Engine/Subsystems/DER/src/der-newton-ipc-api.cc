#include <DER/der-newton-ipc-api.h>

#include <DER/der-subsystem.h>

namespace ksk::der {

runtime::SubsystemIdx DERNewtonIpcApi::subsystemId() const noexcept
{
  return subsystem_.id();
}

runtime::DofRange DERNewtonIpcApi::dofRange() const noexcept
{
  return subsystem_.dofRange();
}

double DERNewtonIpcApi::evaluateObjective(runtime::ConstDofView q,
                                          runtime::ConstDofView qdot,
                                          double dt)
{
  return subsystem_.evaluateObjective(q, qdot, dt);
}

void DERNewtonIpcApi::updateInternalConstraints(double time, double dt)
{
  subsystem_.updateInternalConstraints(time, dt);
}

void DERNewtonIpcApi::prepareLocalOperator(double dt)
{
  subsystem_.prepareLocalOperator(dt);
}

void DERNewtonIpcApi::assembleLocalGradient(runtime::DofView g) const
{
  subsystem_.assembleLocalGradient(g);
}

void DERNewtonIpcApi::applyLocalMatrix(runtime::ConstDofView x,
                                       runtime::DofView y) const
{
  subsystem_.applyLocalMatrix(x, y);
}

void DERNewtonIpcApi::appendLocalMatrixEntries(
    std::vector<runtime::SparseMatrixEntry>& entries) const
{
  subsystem_.appendLocalMatrixEntries(entries);
}

void DERNewtonIpcApi::solveLocalSystem(runtime::ConstDofView b,
                                       runtime::DofView x) const
{
  subsystem_.solveLocalSystem(b, x);
}

void DERNewtonIpcApi::mapLocalDirectionToGeometry(
    runtime::ConstDofView localDq, runtime::GeometryView globalDx) const
{
  subsystem_.mapLocalDirectionToGeometry(localDq, globalDx);
}

void DERNewtonIpcApi::applyInternalContacts(runtime::ContactStencils contacts)
{
  subsystem_.applyInternalContacts(std::move(contacts));
}

void DERNewtonIpcApi::scatterContactGradient(
    std::span<const runtime::PointIdx> points,
    runtime::ConstGeometryView pointGradient, runtime::DofView g) const
{
  subsystem_.scatterContactGradient(points, pointGradient, g);
}

void DERNewtonIpcApi::applyContactGeometryHessianProduct(
    std::span<const runtime::PointIdx> productPoints,
    runtime::ConstGeometryView pointHessianProduct,
    runtime::DofView localY) const
{
  subsystem_.applyContactGeometryHessianProduct(productPoints,
                                                pointHessianProduct, localY);
}

void DERNewtonIpcApi::applyInternalContactHessian(
    runtime::ConstDofView localDq, runtime::DofView localY) const
{
  subsystem_.applyInternalContactHessian(localDq, localY);
}

}  // namespace ksk::der
