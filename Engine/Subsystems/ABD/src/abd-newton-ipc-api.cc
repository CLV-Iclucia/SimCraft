#include <ABD/abd-newton-ipc-api.h>

#include <ABD/abd-subsystem.h>

namespace ksk::abd {

runtime::SubsystemIdx ABDNewtonIpcApi::subsystemId() const noexcept
{
  return subsystem_.id();
}

runtime::DofRange ABDNewtonIpcApi::dofRange() const noexcept
{
  return subsystem_.dofRange();
}

double ABDNewtonIpcApi::evaluateObjective(runtime::ConstDofView q,
                                          runtime::ConstDofView qdot,
                                          double dt)
{
  return subsystem_.evaluateObjective(q, qdot, dt);
}

void ABDNewtonIpcApi::updateInternalConstraints(double time, double dt)
{
  subsystem_.updateInternalConstraints(time, dt);
}

void ABDNewtonIpcApi::prepareLocalOperator(double dt)
{
  subsystem_.prepareLocalOperator(dt);
}

void ABDNewtonIpcApi::assembleLocalGradient(runtime::DofView g) const
{
  subsystem_.assembleLocalGradient(g);
}

void ABDNewtonIpcApi::applyLocalMatrix(runtime::ConstDofView x,
                                       runtime::DofView y) const
{
  subsystem_.applyLocalMatrix(x, y);
}

void ABDNewtonIpcApi::appendLocalMatrixEntries(
    std::vector<runtime::SparseMatrixEntry>& entries) const
{
  subsystem_.appendLocalMatrixEntries(entries);
}

void ABDNewtonIpcApi::solveLocalSystem(runtime::ConstDofView b,
                                       runtime::DofView x) const
{
  subsystem_.solveLocalSystem(b, x);
}

void ABDNewtonIpcApi::mapLocalDirectionToGeometry(
    runtime::ConstDofView localDq, runtime::GeometryView globalDx) const
{
  subsystem_.mapLocalDirectionToGeometry(localDq, globalDx);
}

void ABDNewtonIpcApi::applyInternalContacts(runtime::ContactStencils contacts)
{
  subsystem_.applyInternalContacts(std::move(contacts));
}

void ABDNewtonIpcApi::scatterContactGradient(
    std::span<const runtime::PointIdx> points,
    runtime::ConstGeometryView pointGradient, runtime::DofView g) const
{
  subsystem_.scatterContactGradient(points, pointGradient, g);
}

void ABDNewtonIpcApi::applyContactGeometryHessianProduct(
    std::span<const runtime::PointIdx> productPoints,
    runtime::ConstGeometryView pointHessianProduct,
    runtime::DofView localY) const
{
  subsystem_.applyContactGeometryHessianProduct(productPoints,
                                                pointHessianProduct, localY);
}

void ABDNewtonIpcApi::applyInternalContactHessian(
    runtime::ConstDofView localDq, runtime::DofView localY) const
{
  subsystem_.applyInternalContactHessian(localDq, localY);
}

}  // namespace ksk::abd
