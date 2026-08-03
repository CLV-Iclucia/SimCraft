#include <ABD/abd-subsystem.h>
#include <ABD/cpu/abd-cpu-backend.h>
#include <Contact/contact-barrier.h>
#include <Core/profiler.h>
#include <Runtime/contact-barrier-energy.h>

#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <cmath>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksk::abd {
namespace {

void requireCPU(runtime::ConstDofView view, const char* operation) {
  if (!view.isCPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a CPU DofView");
  }
}

void requireCPU(runtime::DofView view, const char* operation) {
  requireCPU(view.asConst(), operation);
}

void requireCPU(runtime::ConstGeometryView view, const char* operation) {
  if (!view.isCPU()) {
    throw std::runtime_error(std::string(operation) +
                             " requires a CPU GeometryView");
  }
}

void requireCPU(runtime::GeometryView view, const char* operation) {
  requireCPU(view.asConst(), operation);
}

void requireDofCount(runtime::ConstDofView view, int expected,
                     const char* operation) {
  if (view.scalarCount() < expected) {
    throw std::invalid_argument(std::string(operation) +
                                " received a DOF view that is too small");
  }
}

void requireDofCount(runtime::DofView view, int expected,
                     const char* operation) {
  requireDofCount(view.asConst(), expected, operation);
}

void writeAffine(runtime::DofView view, int offset, const glm::dmat4x3& value) {
  for (int column = 0; column < 4; ++column) {
    view[offset + 3 * column + 0] = value[column].x;
    view[offset + 3 * column + 1] = value[column].y;
    view[offset + 3 * column + 2] = value[column].z;
  }
}

glm::dmat4x3 readAffine(runtime::ConstDofView view, int offset) {
  glm::dmat4x3 value{0.0};
  for (int column = 0; column < 4; ++column) {
    value[column] =
        glm::dvec3(view[offset + 3 * column + 0], view[offset + 3 * column + 1],
                   view[offset + 3 * column + 2]);
  }
  return value;
}

glm::dvec3 affineProduct(runtime::ConstDofView view, int offset,
                         const glm::dvec4& homogeneous) {
  glm::dvec3 result{0.0};
  for (int column = 0; column < 4; ++column) {
    result +=
        glm::dvec3(view[offset + 3 * column + 0], view[offset + 3 * column + 1],
                   view[offset + 3 * column + 2]) *
        homogeneous[column];
  }
  return result;
}

void scatterAffineGradient(runtime::DofView view, int offset,
                           const glm::dvec3& gx,
                           const glm::dvec4& homogeneous) {
  for (int column = 0; column < 4; ++column) {
    view[offset + 3 * column + 0] += gx.x * homogeneous[column];
    view[offset + 3 * column + 1] += gx.y * homogeneous[column];
    view[offset + 3 * column + 2] += gx.z * homogeneous[column];
  }
}

engine::contact::EBarrierStencil barrierStencilType(runtime::ContactCase type) {
  switch (type) {
    case runtime::ContactCase::PP:
      return engine::contact::EBarrierStencil::PP;
    case runtime::ContactCase::PE:
      return engine::contact::EBarrierStencil::PE;
    case runtime::ContactCase::PT:
      return engine::contact::EBarrierStencil::PT;
    case runtime::ContactCase::EE:
      return engine::contact::EBarrierStencil::EE;
  }
  return engine::contact::EBarrierStencil::PP;
}

struct ABDContactPoint {
  runtime::PointIdx point = -1;
  int body = -1;
  int vertex = -1;
  glm::dvec4 Xh{0.0};
};

std::optional<ABDContactPoint> gatherContactPoint(const ABDSubsystem& subsystem,
                                                  runtime::PointIdx point) {
  const int sample = subsystem.findGeometryPoint(point);
  if (sample < 0) {
    return std::nullopt;
  }
  const int body = subsystem.geometryPointBody(sample);
  const int vertex = subsystem.geometryPointVertex(sample);
  const glm::dvec3 X =
      subsystem.bodies()[static_cast<size_t>(body)].surface.vertices.at(
          static_cast<size_t>(vertex));
  return ABDContactPoint{
      .point = point,
      .body = body,
      .vertex = vertex,
      .Xh = glm::dvec4(X, 1.0),
  };
}

std::optional<std::array<ABDContactPoint, 4>> gatherContactPoints(
    const ABDSubsystem& subsystem, const runtime::ContactStencil& contact) {
  std::array<ABDContactPoint, 4> result;
  const int count = contactPointCount(contact.type);
  for (int i = 0; i < count; ++i) {
    std::optional<ABDContactPoint> point = gatherContactPoint(
        subsystem,
        runtime::PointIdx{contact.geometryIds[static_cast<size_t>(i)]});
    if (!point) {
      return std::nullopt;
    }
    result[static_cast<size_t>(i)] = *point;
  }
  return result;
}

glm::dvec3 transformPoint(const glm::dmat4x3& M, const glm::dvec4& Xh) {
  return M * Xh;
}

ksk::engine::contact::GIPCBarrierStencil makeBarrierStencil(
    const ABDSubsystem& subsystem, const runtime::ContactStencil& contact,
    const std::array<ABDContactPoint, 4>& points) {
  ksk::engine::contact::GIPCBarrierStencil barrier;
  barrier.type = barrierStencilType(contact.type);
  barrier.dHat = contact.dHat;
  barrier.reservedDist = contact.thickness;
  barrier.kappa = contact.stiffness;
  const int count = contactPointCount(contact.type);
  for (int i = 0; i < count; ++i) {
    const ABDContactPoint& point = points[static_cast<size_t>(i)];
    barrier.x[static_cast<size_t>(i)] =
        transformPoint(subsystem.bodyTransform(point.body), point.Xh);
    barrier.restX[static_cast<size_t>(i)] =
        transformPoint(subsystem.restTransform(point.body), point.Xh);
  }
  return barrier;
}

Eigen::VectorXd gatherCurrentState(const ABDSubsystem& subsystem) {
  const int scalar_count = subsystem.dofRange().scalarCount;
  Eigen::VectorXd values(scalar_count);
  for (const ABDBodyRuntimeRef& body : subsystem.runtimeBodies()) {
    const glm::dmat4x3 M = subsystem.bodyTransform(body.body);
    for (int column = 0; column < 4; ++column) {
      values[body.scalarOffset + 3 * column + 0] = M[column].x;
      values[body.scalarOffset + 3 * column + 1] = M[column].y;
      values[body.scalarOffset + 3 * column + 2] = M[column].z;
    }
  }
  return values;
}

Eigen::Vector4d computeMassFirstMoment(const ABDBodyDesc& desc) {
  Eigen::Vector4d moment = Eigen::Vector4d::Zero();
  double signed_volume = 0.0;
  for (const std::array<int, 3>& tri : desc.surface.triangles) {
    const glm::dvec3 a = desc.surface.vertices.at(static_cast<size_t>(tri[0]));
    const glm::dvec3 b = desc.surface.vertices.at(static_cast<size_t>(tri[1]));
    const glm::dvec3 c = desc.surface.vertices.at(static_cast<size_t>(tri[2]));
    const double volume = glm::dot(a, glm::cross(b, c)) / 6.0;
    signed_volume += volume;

    const Eigen::Vector4d sum = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0) +
                                Eigen::Vector4d(a.x, a.y, a.z, 1.0) +
                                Eigen::Vector4d(b.x, b.y, b.z, 1.0) +
                                Eigen::Vector4d(c.x, c.y, c.z, 1.0);
    moment += (volume / 4.0) * sum;
  }

  if (signed_volume < 0.0) {
    moment = -moment;
    signed_volume = -signed_volume;
  }

  const double density = std::max(desc.density, 1.0e-12);
  if (signed_volume <= 1.0e-12) {
    return Eigen::Vector4d(0.0, 0.0, 0.0, density);
  }
  return density * moment;
}

glm::dvec3 massWeightedPosition(const glm::dmat4x3& transform,
                                const Eigen::Vector4d& mass_first_moment) {
  glm::dvec3 position{0.0};
  for (int column = 0; column < 4; ++column) {
    position += transform[column] * mass_first_moment[column];
  }
  return position;
}

double enclosedSurfaceVolume(const ABDBodyDesc& desc) {
  double signed_volume = 0.0;
  for (const std::array<int, 3>& tri : desc.surface.triangles) {
    const glm::dvec3 a = desc.surface.vertices.at(static_cast<size_t>(tri[0]));
    const glm::dvec3 b = desc.surface.vertices.at(static_cast<size_t>(tri[1]));
    const glm::dvec3 c = desc.surface.vertices.at(static_cast<size_t>(tri[2]));
    signed_volume += glm::dot(a, glm::cross(b, c)) / 6.0;
  }
  return std::abs(signed_volume);
}

glm::dmat3 linearPart(const glm::dmat4x3& affine) {
  return glm::dmat3(affine[0], affine[1], affine[2]);
}

double orthogonalPotentialEnergy(const ABDBodyDesc& desc,
                                 const glm::dmat4x3& affine) {
  const double scale = desc.orthogonalStiffness * enclosedSurfaceVolume(desc);
  if (scale <= 0.0) {
    return 0.0;
  }

  const glm::dmat3 A = linearPart(affine);
  double value = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double diagonal = glm::dot(A[i], A[i]) - 1.0;
    value += diagonal * diagonal;
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        continue;
      }
      const double off_diagonal = glm::dot(A[i], A[j]);
      value += off_diagonal * off_diagonal;
    }
  }
  return scale * value;
}

glm::dmat3 orthogonalPotentialGradient(const ABDBodyDesc& desc,
                                       const glm::dmat4x3& affine) {
  const double scale =
      4.0 * desc.orthogonalStiffness * enclosedSurfaceVolume(desc);
  if (scale <= 0.0) {
    return glm::dmat3{0.0};
  }

  const glm::dmat3 A = linearPart(affine);
  const glm::dmat3 aat_residual = A * glm::transpose(A) - glm::dmat3(1.0);
  return scale * aat_residual * A;
}

glm::dmat3 orthogonalPotentialHessianProduct(const ABDBodyDesc& desc,
                                             const glm::dmat4x3& affine,
                                             const glm::dmat3& direction) {
  const double scale =
      4.0 * desc.orthogonalStiffness * enclosedSurfaceVolume(desc);
  if (scale <= 0.0) {
    return glm::dmat3{0.0};
  }

  const glm::dmat3 A = linearPart(affine);
  const glm::dmat3 aat_residual = A * glm::transpose(A) - glm::dmat3(1.0);
  const glm::dmat3 d_aat =
      direction * glm::transpose(A) + A * glm::transpose(direction);
  return scale * (d_aat * A + aat_residual * direction);
}

void addLinearBlock(runtime::DofView view, int offset,
                    const glm::dmat3& block) {
  for (int column = 0; column < 3; ++column) {
    view[offset + 3 * column + 0] += block[column].x;
    view[offset + 3 * column + 1] += block[column].y;
    view[offset + 3 * column + 2] += block[column].z;
  }
}

}  // namespace

ABDCPUBackend::ABDCPUBackend(ABDSubsystem& subsystem)
    : subsystem_(subsystem), mass_matrix_(buildMassMatrix()) {}

void ABDCPUBackend::writeState(runtime::DofView q,
                               runtime::DofView qdot) const {
  KSK_PROFILE_SCOPE("ABDCPUBackend/WriteState");
  requireCPU(q, "ABDCPUBackend::writeState");
  requireCPU(qdot, "ABDCPUBackend::writeState");
  requireDofCount(q, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::writeState");
  requireDofCount(qdot, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::writeState");

  for (const ABDBodyRuntimeRef& body : subsystem_.runtime_bodies_) {
    writeAffine(q, body.scalarOffset, subsystem_.bodyTransform(body.body));
    writeAffine(qdot, body.scalarOffset, subsystem_.bodyVelocity(body.body));
  }
}

void ABDCPUBackend::readState(runtime::ConstDofView q,
                              runtime::ConstDofView qdot) {
  KSK_PROFILE_SCOPE("ABDCPUBackend/ReadState");
  requireCPU(q, "ABDCPUBackend::readState");
  requireCPU(qdot, "ABDCPUBackend::readState");
  requireDofCount(q, subsystem_.range_.scalarCount, "ABDCPUBackend::readState");
  requireDofCount(qdot, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::readState");

  for (const ABDBodyRuntimeRef& body : subsystem_.runtime_bodies_) {
    subsystem_.setBodyTransform(body.body, readAffine(q, body.scalarOffset));
    subsystem_.setBodyVelocity(body.body, readAffine(qdot, body.scalarOffset));
  }
}

void ABDCPUBackend::beginStep(runtime::ConstDofView q,
                              runtime::ConstDofView qdot, double dt) {
  KSK_PROFILE_SCOPE("ABDCPUBackend/BeginStep");
  step_dt_ = dt;
  step_start_ = gatherLocalVector(q, "ABDCPUBackend::beginStep");
  inertial_target_ =
      step_start_ + dt * gatherLocalVector(qdot, "ABDCPUBackend::beginStep");
  mass_matrix_ = buildMassMatrix();
}

void ABDCPUBackend::acceptStep(runtime::ConstDofView q, runtime::DofView qdot,
                               double dt) {
  KSK_PROFILE_SCOPE("ABDCPUBackend/AcceptStep");
  if (step_start_.size() != subsystem_.range_.scalarCount) {
    throw std::runtime_error("ABD CPU backend step has not begun");
  }
  requireCPU(q, "ABDCPUBackend::acceptStep");
  requireCPU(qdot, "ABDCPUBackend::acceptStep");
  requireDofCount(q, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::acceptStep");
  requireDofCount(qdot, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::acceptStep");

  for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
    qdot[i] = (q[i] - step_start_[i]) / dt;
  }
}

double ABDCPUBackend::evaluateObjective(runtime::ConstDofView q,
                                        runtime::ConstDofView qdot, double dt) {
  KSK_PROFILE_SCOPE("ABDCPUBackend/EvaluateObjective");
  (void)qdot;
  if (inertial_target_.size() != subsystem_.range_.scalarCount) {
    return 0.0;
  }
  const Eigen::VectorXd local_q =
      gatherLocalVector(q, "ABDCPUBackend::evaluateObjective");
  const Eigen::VectorXd residual = local_q - inertial_target_;
  double objective = 0.0;
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/EvaluateObjective/Inertia");
    objective += 0.5 * residual.dot(mass_matrix_ * residual) / (dt * dt);
  }
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/EvaluateObjective/OrthogonalPotential");
    for (const ABDBodyRuntimeRef& body : subsystem_.runtime_bodies_) {
      const ABDBodyDesc& desc =
          subsystem_.bodies_[static_cast<size_t>(body.body)];
      const glm::dmat4x3 affine = readAffine(q, body.scalarOffset);
      objective += orthogonalPotentialEnergy(desc, affine);
    }
  }
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/EvaluateObjective/Gravity");
    for (const ABDBodyRuntimeRef& body : subsystem_.runtime_bodies_) {
      const ABDBodyDesc& desc =
          subsystem_.bodies_[static_cast<size_t>(body.body)];
      const glm::dmat4x3 affine = readAffine(q, body.scalarOffset);
      objective -=
          glm::dot(subsystem_.gravity_,
                   massWeightedPosition(affine, computeMassFirstMoment(desc)));
    }
  }
  KSK_PROFILE_VALUE("abd.internal_contacts",
                    static_cast<double>(subsystem_.internal_contacts_.size()));
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/EvaluateObjective/InternalContacts");
    for (const runtime::ContactStencil& contact :
         subsystem_.internal_contacts_) {
      const std::optional<std::array<ABDContactPoint, 4>> points =
          gatherContactPoints(subsystem_, contact);
      if (!points) {
        continue;
      }
      const ksk::engine::contact::GIPCBarrierStencil barrier =
          makeBarrierStencil(subsystem_, contact, *points);
      objective += ksk::engine::contact::computeGIPCBarrierEnergy(barrier);
    }
  }
  return objective;
}

void ABDCPUBackend::updateInternalConstraints(double time, double dt) {
  (void)time;
  (void)dt;
}

void ABDCPUBackend::prepareLocalOperator(double dt) {
  KSK_PROFILE_SCOPE("ABDCPUBackend/PrepareLocalOperator");
  const int scalar_count = subsystem_.range_.scalarCount;
  const double inv_dt2 = dt > 0.0 ? 1.0 / (dt * dt) : 1.0;
  local_solver_ready_ = false;
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/PrepareLocalOperator/BuildMassMatrix");
    mass_matrix_ = buildMassMatrix();
  }
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<size_t>(
      scalar_count * 16 + scalar_count * scalar_count + scalar_count));
  {
    KSK_PROFILE_SCOPE(
        "ABDCPUBackend/PrepareLocalOperator/AssembleMassTriplets");
    for (int col = 0; col < mass_matrix_.cols(); ++col) {
      for (int row = 0; row < mass_matrix_.rows(); ++row) {
        const double value = inv_dt2 * mass_matrix_(row, col);
        if (value != 0.0) {
          triplets.emplace_back(row, col, value);
        }
      }
    }
  }

  bool has_orthogonal_potential = false;
  for (const ABDBodyDesc& desc : subsystem_.bodies_) {
    has_orthogonal_potential =
        has_orthogonal_potential ||
        desc.orthogonalStiffness * enclosedSurfaceVolume(desc) > 0.0;
  }
  if (!subsystem_.internal_contacts_.empty() || has_orthogonal_potential) {
    KSK_PROFILE_SCOPE(
        "ABDCPUBackend/PrepareLocalOperator/AssemblePotentialTriplets");
    std::vector<double> basis_values(static_cast<size_t>(scalar_count), 0.0);
    std::vector<double> product_values(static_cast<size_t>(scalar_count), 0.0);
    for (int column = 0; column < scalar_count; ++column) {
      std::fill(basis_values.begin(), basis_values.end(), 0.0);
      std::fill(product_values.begin(), product_values.end(), 0.0);
      basis_values[static_cast<size_t>(column)] = 1.0;
      applyInternalContactHessian(
          runtime::ConstDofView::CPU(basis_values.data(), scalar_count),
          runtime::DofView::CPU(product_values.data(), scalar_count));
      for (const ABDBodyRuntimeRef& body : subsystem_.runtime_bodies_) {
        const ABDBodyDesc& desc =
            subsystem_.bodies_[static_cast<size_t>(body.body)];
        addLinearBlock(
            runtime::DofView::CPU(product_values.data(), scalar_count),
            body.scalarOffset,
            orthogonalPotentialHessianProduct(
                desc, subsystem_.bodyTransform(body.body),
                linearPart(readAffine(runtime::ConstDofView::CPU(
                                          basis_values.data(), scalar_count),
                                      body.scalarOffset))));
      }
      for (int row = 0; row < scalar_count; ++row) {
        const double value = product_values[static_cast<size_t>(row)];
        if (value != 0.0) {
          triplets.emplace_back(row, column, value);
        }
      }
    }
  }
  for (int i = 0; i < scalar_count; ++i) {
    triplets.emplace_back(i, i, 1.0e-12);
  }

  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/PrepareLocalOperator/CompressMatrix");
    local_matrix_.resize(scalar_count, scalar_count);
    local_matrix_.setFromTriplets(triplets.begin(), triplets.end());
  }
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/PrepareLocalOperator/Factorize");
    local_solver_.compute(local_matrix_);
  }
  if (local_solver_.info() != Eigen::Success) {
    throw std::runtime_error("failed to factor ABD CPU backend local matrix");
  }
  local_solver_ready_ = true;
}

void ABDCPUBackend::assembleLocalGradient(runtime::DofView g) const {
  requireCPU(g, "ABDCPUBackend::assembleLocalGradient");
  requireDofCount(g, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::assembleLocalGradient");
  if (step_dt_ <= 0.0 ||
      inertial_target_.size() != subsystem_.range_.scalarCount) {
    return;
  }

  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/AssembleLocalGradient/Inertia");
    const Eigen::VectorXd current = gatherCurrentState(subsystem_);
    const Eigen::VectorXd local_gradient =
        mass_matrix_ * (current - inertial_target_) / (step_dt_ * step_dt_);
    for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
      g[i] += local_gradient[i];
    }
  }
  {
    KSK_PROFILE_SCOPE(
        "ABDCPUBackend/AssembleLocalGradient/OrthogonalPotential");
    for (const ABDBodyRuntimeRef& body : subsystem_.runtime_bodies_) {
      const ABDBodyDesc& desc =
          subsystem_.bodies_[static_cast<size_t>(body.body)];
      addLinearBlock(g, body.scalarOffset,
                     orthogonalPotentialGradient(
                         desc, subsystem_.bodyTransform(body.body)));
    }
  }
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/AssembleLocalGradient/Gravity");
    for (const ABDBodyRuntimeRef& body : subsystem_.runtime_bodies_) {
      const ABDBodyDesc& desc =
          subsystem_.bodies_[static_cast<size_t>(body.body)];
      const Eigen::Vector4d mass_first_moment = computeMassFirstMoment(desc);
      for (int column = 0; column < 4; ++column) {
        const glm::dvec3 gravity_gradient =
            -subsystem_.gravity_ * mass_first_moment[column];
        g[body.scalarOffset + 3 * column + 0] += gravity_gradient.x;
        g[body.scalarOffset + 3 * column + 1] += gravity_gradient.y;
        g[body.scalarOffset + 3 * column + 2] += gravity_gradient.z;
      }
    }
  }

  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/AssembleLocalGradient/InternalContacts");
    for (const runtime::ContactStencil& contact :
         subsystem_.internal_contacts_) {
      const std::optional<std::array<ABDContactPoint, 4>> points =
          gatherContactPoints(subsystem_, contact);
      if (!points) {
        continue;
      }
      const ksk::engine::contact::GIPCBarrierStencil barrier =
          makeBarrierStencil(subsystem_, contact, *points);
      const ksk::engine::contact::LocalBarrierGradient contact_gradient =
          ksk::engine::contact::computeGIPCBarrierGradient(barrier);
      const int count = contactPointCount(contact.type);
      for (int i = 0; i < count; ++i) {
        const ABDContactPoint& point = (*points)[static_cast<size_t>(i)];
        const int offset =
            subsystem_.runtime_bodies_[static_cast<size_t>(point.body)]
                .scalarOffset;
        scatterAffineGradient(g, offset,
                              contact_gradient.gradient[static_cast<size_t>(i)],
                              point.Xh);
      }
    }
  }
}

void ABDCPUBackend::applyLocalMatrix(runtime::ConstDofView x,
                                     runtime::DofView y) const {
  requireCPU(x, "ABDCPUBackend::applyLocalMatrix");
  requireCPU(y, "ABDCPUBackend::applyLocalMatrix");
  requireDofCount(x, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::applyLocalMatrix");
  requireDofCount(y, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::applyLocalMatrix");

  const double inv_dt2 = step_dt_ > 0.0 ? 1.0 / (step_dt_ * step_dt_) : 1.0;
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/ApplyLocalMatrix/Inertia");
    const Eigen::VectorXd local_x =
        gatherLocalVector(x, "ABDCPUBackend::applyLocalMatrix");
    const Eigen::VectorXd local_y = inv_dt2 * (mass_matrix_ * local_x);
    for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
      y[i] += local_y[i];
    }
  }

  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/ApplyLocalMatrix/InternalContactHessian");
    applyInternalContactHessian(x, y);
  }
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/ApplyLocalMatrix/OrthogonalHessian");
    for (const ABDBodyRuntimeRef& body : subsystem_.runtime_bodies_) {
      const ABDBodyDesc& desc =
          subsystem_.bodies_[static_cast<size_t>(body.body)];
      addLinearBlock(y, body.scalarOffset,
                     orthogonalPotentialHessianProduct(
                         desc, subsystem_.bodyTransform(body.body),
                         linearPart(readAffine(x, body.scalarOffset))));
    }
  }
}

void ABDCPUBackend::appendLocalMatrixEntries(
    std::vector<runtime::SparseMatrixEntry>& entries) const {
  KSK_PROFILE_SCOPE("ABDCPUBackend/AppendLocalMatrixEntries");
  if (local_matrix_.rows() != subsystem_.range_.scalarCount) {
    throw std::runtime_error("ABD CPU backend local operator is not prepared");
  }

  const int offset = subsystem_.range_.scalarOffset;
  entries.reserve(entries.size() +
                  static_cast<size_t>(local_matrix_.nonZeros()));
  for (int outer = 0; outer < local_matrix_.outerSize(); ++outer) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(local_matrix_, outer);
         it; ++it) {
      entries.push_back(runtime::SparseMatrixEntry{
          .row = offset + static_cast<int>(it.row()),
          .col = offset + static_cast<int>(it.col()),
          .value = it.value(),
      });
    }
  }
}

void ABDCPUBackend::solveLocalSystem(runtime::ConstDofView b,
                                     runtime::DofView x) const {
  KSK_PROFILE_SCOPE("ABDCPUBackend/SolveLocalSystem");
  requireCPU(b, "ABDCPUBackend::solveLocalSystem");
  requireCPU(x, "ABDCPUBackend::solveLocalSystem");
  requireDofCount(b, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::solveLocalSystem");
  requireDofCount(x, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::solveLocalSystem");
  if (local_matrix_.rows() != subsystem_.range_.scalarCount) {
    throw std::runtime_error("ABD CPU backend local operator is not prepared");
  }
  if (!local_solver_ready_) {
    throw std::runtime_error("ABD CPU backend local solver is not prepared");
  }

  Eigen::VectorXd local_x;
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/SolveLocalSystem/Solve");
    const Eigen::VectorXd local_b =
        gatherLocalVector(b, "ABDCPUBackend::solveLocalSystem");
    local_x = local_solver_.solve(local_b);
  }
  if (local_solver_.info() != Eigen::Success) {
    throw std::runtime_error("failed to solve ABD CPU backend local system");
  }
  {
    KSK_PROFILE_SCOPE("ABDCPUBackend/SolveLocalSystem/ScatterSolution");
    for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
      x[i] = local_x[i];
    }
  }
}

void ABDCPUBackend::mapLocalDirectionToGeometry(
    runtime::ConstDofView localDq, runtime::GeometryView globalDx) const {
  KSK_PROFILE_SCOPE("ABDCPUBackend/MapLocalDirectionToGeometry");
  requireCPU(localDq, "ABDCPUBackend::mapLocalDirectionToGeometry");
  requireCPU(globalDx, "ABDCPUBackend::mapLocalDirectionToGeometry");
  requireDofCount(localDq, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::mapLocalDirectionToGeometry");

  for (int sample = 0;
       sample < static_cast<int>(subsystem_.geometry_points_.size());
       ++sample) {
    const runtime::PointIdx point = subsystem_.geometry_points_[sample];
    if (point < 0 || point >= globalDx.pointCount()) {
      throw std::invalid_argument(
          "ABD global geometry direction buffer is too small");
    }

    const int body = subsystem_.geometry_point_bodies_[sample];
    const int vertex = subsystem_.geometry_point_vertices_[sample];
    const ABDBodyDesc& body_desc = subsystem_.bodies_[body];
    const glm::dvec3 X = body_desc.surface.vertices.at(vertex);
    const glm::dvec4 X_h{X, 1.0};
    const int offset = subsystem_.runtime_bodies_[body].scalarOffset;
    globalDx[point] = affineProduct(localDq, offset, X_h);
  }
}

void ABDCPUBackend::scatterContactGradient(
    std::span<const runtime::PointIdx> points,
    runtime::ConstGeometryView pointGradient, runtime::DofView g) const {
  KSK_PROFILE_SCOPE("ABDCPUBackend/ScatterContactGradient");
  requireCPU(pointGradient, "ABDCPUBackend::scatterContactGradient");
  requireCPU(g, "ABDCPUBackend::scatterContactGradient");
  if (points.size() != static_cast<size_t>(pointGradient.pointCount())) {
    throw std::invalid_argument("contact point and gradient counts differ");
  }
  requireDofCount(g, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::scatterContactGradient");

  for (int i = 0; i < static_cast<int>(points.size()); ++i) {
    const int sample = subsystem_.findGeometryPoint(points[i]);
    if (sample < 0) {
      continue;
    }
    const int body = subsystem_.geometry_point_bodies_[sample];
    const int vertex = subsystem_.geometry_point_vertices_[sample];
    const ABDBodyDesc& body_desc = subsystem_.bodies_[body];
    const glm::dvec3 X = body_desc.surface.vertices.at(vertex);
    const glm::dvec4 X_h{X, 1.0};
    const int offset = subsystem_.runtime_bodies_[body].scalarOffset;
    scatterAffineGradient(g, offset, pointGradient[i], X_h);
  }
}

void ABDCPUBackend::applyContactGeometryHessianProduct(
    std::span<const runtime::PointIdx> productPoints,
    runtime::ConstGeometryView pointHessianProduct,
    runtime::DofView localY) const {
  scatterContactGradient(productPoints, pointHessianProduct, localY);
}

void ABDCPUBackend::applyInternalContactHessian(runtime::ConstDofView localDq,
                                                runtime::DofView localY) const {
  KSK_PROFILE_SCOPE("ABDCPUBackend/ApplyInternalContactHessian");
  KSK_PROFILE_VALUE("abd.internal_contact_hessian.contacts",
                    static_cast<double>(subsystem_.internal_contacts_.size()));
  requireCPU(localDq, "ABDCPUBackend::applyInternalContactHessian");
  requireCPU(localY, "ABDCPUBackend::applyInternalContactHessian");
  requireDofCount(localDq, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::applyInternalContactHessian");
  requireDofCount(localY, subsystem_.range_.scalarCount,
                  "ABDCPUBackend::applyInternalContactHessian");

  for (const runtime::ContactStencil& contact : subsystem_.internal_contacts_) {
    const std::optional<std::array<ABDContactPoint, 4>> points =
        gatherContactPoints(subsystem_, contact);
    if (!points) {
      continue;
    }
    std::array<glm::dvec3, 4> direction{
        glm::dvec3(0.0),
        glm::dvec3(0.0),
        glm::dvec3(0.0),
        glm::dvec3(0.0),
    };
    const int count = contactPointCount(contact.type);
    for (int i = 0; i < count; ++i) {
      const ABDContactPoint& point = (*points)[static_cast<size_t>(i)];
      const int offset =
          subsystem_.runtime_bodies_[static_cast<size_t>(point.body)]
              .scalarOffset;
      direction[static_cast<size_t>(i)] =
          affineProduct(localDq, offset, point.Xh);
    }

    const ksk::engine::contact::GIPCBarrierStencil barrier =
        makeBarrierStencil(subsystem_, contact, *points);
    const std::array<glm::dvec3, 4> product =
        ksk::engine::contact::computeGIPCBarrierHessianProduct(barrier,
                                                               direction);
    for (int i = 0; i < count; ++i) {
      const ABDContactPoint& point = (*points)[static_cast<size_t>(i)];
      const int offset =
          subsystem_.runtime_bodies_[static_cast<size_t>(point.body)]
              .scalarOffset;
      scatterAffineGradient(localY, offset, product[static_cast<size_t>(i)],
                            point.Xh);
    }
  }
}

Eigen::VectorXd ABDCPUBackend::gatherLocalVector(runtime::ConstDofView view,
                                                 const char* operation) const {
  requireCPU(view, operation);
  requireDofCount(view, subsystem_.range_.scalarCount, operation);
  Eigen::VectorXd local(subsystem_.range_.scalarCount);
  for (int i = 0; i < subsystem_.range_.scalarCount; ++i) {
    local[i] = view[i];
  }
  return local;
}

Eigen::MatrixXd ABDCPUBackend::buildMassMatrix() const {
  Eigen::MatrixXd mass = Eigen::MatrixXd::Zero(subsystem_.range_.scalarCount,
                                               subsystem_.range_.scalarCount);
  for (const ABDBodyRuntimeRef& body : subsystem_.runtime_bodies_) {
    Eigen::Matrix4d moment = Eigen::Matrix4d::Zero();
    double signed_volume = 0.0;
    const ABDBodyDesc& desc =
        subsystem_.bodies_[static_cast<size_t>(body.body)];
    for (const std::array<int, 3>& tri : desc.surface.triangles) {
      const glm::dvec3 a =
          desc.surface.vertices.at(static_cast<size_t>(tri[0]));
      const glm::dvec3 b =
          desc.surface.vertices.at(static_cast<size_t>(tri[1]));
      const glm::dvec3 c =
          desc.surface.vertices.at(static_cast<size_t>(tri[2]));
      const double volume = glm::dot(a, glm::cross(b, c)) / 6.0;
      signed_volume += volume;

      const std::array<Eigen::Vector4d, 4> X{
          Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
          Eigen::Vector4d(a.x, a.y, a.z, 1.0),
          Eigen::Vector4d(b.x, b.y, b.z, 1.0),
          Eigen::Vector4d(c.x, c.y, c.z, 1.0),
      };
      Eigen::Vector4d sum = Eigen::Vector4d::Zero();
      Eigen::Matrix4d diag_sum = Eigen::Matrix4d::Zero();
      for (const Eigen::Vector4d& value : X) {
        sum += value;
        diag_sum += value * value.transpose();
      }
      moment += (volume / 20.0) * (sum * sum.transpose() + diag_sum);
    }

    if (signed_volume < 0.0) {
      moment = -moment;
      signed_volume = -signed_volume;
    }
    if (signed_volume <= 1.0e-12) {
      moment.setIdentity();
    }

    const double density = std::max(desc.density, 1.0e-12);
    moment *= density;
    for (int row_column = 0; row_column < 4; ++row_column) {
      for (int col_column = 0; col_column < 4; ++col_column) {
        for (int lane = 0; lane < 3; ++lane) {
          const int row = body.scalarOffset + 3 * row_column + lane;
          const int col = body.scalarOffset + 3 * col_column + lane;
          mass(row, col) = moment(row_column, col_column);
        }
      }
    }
  }
  return mass;
}

}  // namespace ksk::abd
