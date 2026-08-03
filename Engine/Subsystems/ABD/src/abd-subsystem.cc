#include <ABD/abd-subsystem.h>

#include <ABD/cpu/abd-cpu-backend.h>

#include <stdexcept>
#include <utility>

namespace ksk::abd {
namespace {

constexpr int bodyScalarOffset(int body) noexcept
{
  return body * ABDSubsystem::kScalarsPerBody;
}

}  // namespace

glm::dmat4 affineToMat4(const glm::dmat4x3& affine) noexcept
{
  glm::dmat4 transform{1.0};
  transform[0] = glm::dvec4(affine[0], 0.0);
  transform[1] = glm::dvec4(affine[1], 0.0);
  transform[2] = glm::dvec4(affine[2], 0.0);
  transform[3] = glm::dvec4(affine[3], 1.0);
  return transform;
}

ABDSubsystem::ABDSubsystem(runtime::SubsystemIdx id,
                           std::vector<ABDBodyDesc> bodies,
                           int scalarOffset,
                           glm::dvec3 gravity)
    : id_(id),
      range_(runtime::DofRange{
          .subsystem = id,
          .scalarOffset = scalarOffset,
          .scalarCount = static_cast<int>(bodies.size()) * kScalarsPerBody,
          .blockSize = kScalarsPerBody,
      }),
      gravity_(gravity),
      bodies_(std::move(bodies))
{
  rest_transforms_.reserve(bodies_.size());
  runtime_bodies_.reserve(bodies_.size());
  for (int body = 0; body < static_cast<int>(bodies_.size()); ++body) {
    rest_transforms_.push_back(bodies_[static_cast<size_t>(body)].initialTransform);
    runtime_bodies_.push_back(ABDBodyRuntimeRef{
        .body = body,
        .scalarOffset = bodyScalarOffset(body),
    });
  }
  cpu_backend_ = std::make_unique<ABDCPUBackend>(*this);
}

ABDSubsystem::~ABDSubsystem() = default;

runtime::SubsystemIdx ABDSubsystem::id() const noexcept { return id_; }

runtime::DofRange ABDSubsystem::dofRange() const noexcept { return range_; }

void ABDSubsystem::declareGeometry(runtime::GlobalGeometryManager& geometry)
{
  geometry_points_.clear();
  geometry_point_bodies_.clear();
  geometry_point_vertices_.clear();

  for (int body = 0; body < bodies_.size(); body++) {
    ABDBodyDesc& body_desc = bodies_[body];
    ABDBodyRuntimeRef& runtime_body = runtime_bodies_[body];

    const runtime::GeometryBodyIdx geometry_body =
        geometry.addGeometryBody(
            runtime::GeometryBodyInfo{
                .type = runtime::EGeometryBody::Rigid,
                .motion = runtime::EBodyMotionMode::Dynamic,
                .subsystem = id_,
                .localBody = body,
            },
            runtime::GeometryBodyCollisionFilter{
                .group = runtime::kDefaultCollisionGroup,
                .mask = runtime::kAllCollisionGroups,
                .enabled = true,
                .physicalMaterial = body_desc.physicalMaterial});
    runtime_body.geometryBody = geometry_body;

    runtime::GeometryMeshDesc surface = body_desc.surface;
    runtime_body.surfacePrototype =
        geometry.addGeometryPrototype(std::move(surface));

    runtime_body.instance = geometry.addInstance(
        runtime::GeometryOwner{
            .body = geometry_body,
            .subsystem = id_,
        },
        runtime_body.surfacePrototype,
        body,
        affineToMat4(body_desc.initialTransform));

    const runtime::GeometryRange points =
        geometry.instancePointRange(runtime_body.instance);

    for (int vertex = 0; vertex < points.count; vertex++) {
      geometry_points_.push_back(runtime::PointIdx{points.first + vertex});
      geometry_point_bodies_.push_back(body);
      geometry_point_vertices_.push_back(vertex);
    }

  }
}

void ABDSubsystem::writeState(runtime::DofView q,
                              runtime::DofView qdot) const
{
  cpu_backend_->writeState(q, qdot);
}

void ABDSubsystem::readState(runtime::ConstDofView q,
                             runtime::ConstDofView qdot)
{
  cpu_backend_->readState(q, qdot);
}

void ABDSubsystem::beginStep(runtime::ConstDofView q,
                             runtime::ConstDofView qdot,
                             double dt)
{
  cpu_backend_->beginStep(q, qdot, dt);
}

void ABDSubsystem::acceptStep(runtime::ConstDofView q,
                              runtime::DofView qdot,
                              double dt)
{
  cpu_backend_->acceptStep(q, qdot, dt);
}

double ABDSubsystem::evaluateObjective(runtime::ConstDofView q,
                                       runtime::ConstDofView qdot,
                                       double dt)
{
  return cpu_backend_->evaluateObjective(q, qdot, dt);
}

void ABDSubsystem::updateInternalConstraints(double time, double dt)
{
  cpu_backend_->updateInternalConstraints(time, dt);
}

void ABDSubsystem::prepareLocalOperator(double dt)
{
  cpu_backend_->prepareLocalOperator(dt);
}

void ABDSubsystem::assembleLocalGradient(runtime::DofView g) const
{
  cpu_backend_->assembleLocalGradient(g);
}

void ABDSubsystem::applyLocalMatrix(runtime::ConstDofView x,
                                    runtime::DofView y) const
{
  cpu_backend_->applyLocalMatrix(x, y);
}

void ABDSubsystem::appendLocalMatrixEntries(
    std::vector<runtime::SparseMatrixEntry>& entries) const
{
  cpu_backend_->appendLocalMatrixEntries(entries);
}

void ABDSubsystem::solveLocalSystem(runtime::ConstDofView b,
                                    runtime::DofView x) const
{
  cpu_backend_->solveLocalSystem(b, x);
}

void ABDSubsystem::updateGeometry(
    runtime::GlobalGeometryManager& geometry) const
{
  for (const ABDBodyRuntimeRef& body : runtime_bodies_) {
    if (body.instance < 0) {
      throw std::runtime_error("ABD geometry instance has not been declared");
    }
    geometry.setInstanceTransform(body.instance,
                                  affineToMat4(bodyTransform(body.body)));
  }
}

void ABDSubsystem::mapLocalDirectionToGeometry(
    runtime::ConstDofView localDq,
    runtime::GeometryView globalDx) const
{
  cpu_backend_->mapLocalDirectionToGeometry(localDq, globalDx);
}

void ABDSubsystem::applyInternalContacts(runtime::ContactStencils contacts)
{
  internal_contacts_ = std::move(contacts);
}

void ABDSubsystem::scatterContactGradient(
    std::span<const runtime::PointIdx> points,
    runtime::ConstGeometryView pointGradient,
    runtime::DofView g) const
{
  cpu_backend_->scatterContactGradient(points, pointGradient, g);
}

void ABDSubsystem::applyContactGeometryHessianProduct(
    std::span<const runtime::PointIdx> productPoints,
    runtime::ConstGeometryView pointHessianProduct,
    runtime::DofView localY) const
{
  cpu_backend_->applyContactGeometryHessianProduct(
      productPoints, pointHessianProduct, localY);
}

void ABDSubsystem::applyInternalContactHessian(runtime::ConstDofView localDq,
                                               runtime::DofView localY) const
{
  cpu_backend_->applyInternalContactHessian(localDq, localY);
}

void ABDSubsystem::visit(runtime::CpuSubsystemBackend& backend)
{
  (void)backend;
}

void ABDSubsystem::visit(runtime::GpuSubsystemBackend& backend)
{
  (void)backend;
  throw std::runtime_error("ABDSubsystem does not implement a GPU backend yet");
}

const std::vector<ABDBodyDesc>& ABDSubsystem::bodies() const noexcept
{
  return bodies_;
}

const std::vector<ABDBodyRuntimeRef>& ABDSubsystem::runtimeBodies()
    const noexcept
{
  return runtime_bodies_;
}

const std::vector<runtime::PointIdx>& ABDSubsystem::geometryPointIds()
    const noexcept
{
  return geometry_points_;
}

int ABDSubsystem::localScalarCount() const noexcept
{
  return range_.scalarCount;
}

int ABDSubsystem::geometryPointBody(int sample) const
{
  return geometry_point_bodies_.at(static_cast<size_t>(sample));
}

int ABDSubsystem::geometryPointVertex(int sample) const
{
  return geometry_point_vertices_.at(static_cast<size_t>(sample));
}

const glm::dmat4x3& ABDSubsystem::restTransform(int body) const
{
  return rest_transforms_.at(static_cast<size_t>(body));
}

const glm::dvec3& ABDSubsystem::gravity() const noexcept
{
  return gravity_;
}

glm::dmat4x3 ABDSubsystem::bodyTransform(int body) const
{
  return bodies_.at(static_cast<size_t>(body)).initialTransform;
}

glm::dmat4x3 ABDSubsystem::bodyVelocity(int body) const
{
  return bodies_.at(static_cast<size_t>(body)).initialVelocity;
}

void ABDSubsystem::setBodyTransform(int body, const glm::dmat4x3& transform)
{
  bodies_.at(static_cast<size_t>(body)).initialTransform = transform;
}

void ABDSubsystem::setBodyVelocity(int body, const glm::dmat4x3& velocity)
{
  bodies_.at(static_cast<size_t>(body)).initialVelocity = velocity;
}

int ABDSubsystem::findGeometryPoint(runtime::PointIdx point) const noexcept
{
  for (int index = 0; index < static_cast<int>(geometry_points_.size());
       ++index) {
    if (geometry_points_[static_cast<size_t>(index)] == point) {
      return index;
    }
  }
  return -1;
}

}  // namespace ksk::abd
