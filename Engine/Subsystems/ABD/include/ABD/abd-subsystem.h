#pragma once

#include <ABD/abd-scene.h>

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/global-geometry-manager.h>
#include <Runtime/subsystem.h>

#include <glm/glm.hpp>

#include <memory>
#include <span>
#include <vector>

namespace ksk::abd {

class ABDCPUBackend;

struct ABDBodyRuntimeRef {
  int body = -1;
  int scalarOffset = 0;
  runtime::GeometryBodyIdx geometryBody = -1;
  runtime::GeometryPrototypeIdx surfacePrototype = -1;
  runtime::GeometryInstanceIdx instance = -1;
};

class ABDSubsystem final : public runtime::Subsystem {
 public:
  static constexpr int kScalarsPerBody = 12;

  ABDSubsystem(runtime::SubsystemIdx id,
               std::vector<ABDBodyDesc> bodies,
               int scalarOffset = 0,
               glm::dvec3 gravity = glm::dvec3(0.0));
  ~ABDSubsystem() override;

  [[nodiscard]] runtime::SubsystemIdx id() const noexcept override;
  [[nodiscard]] runtime::DofRange dofRange() const noexcept override;

  void declareGeometry(runtime::GlobalGeometryManager& geometry) override;
  void writeState(runtime::DofView q,
                  runtime::DofView qdot) const override;
  void readState(runtime::ConstDofView q,
                 runtime::ConstDofView qdot) override;
  void beginStep(runtime::ConstDofView q,
                 runtime::ConstDofView qdot,
                 double dt) override;
  void acceptStep(runtime::ConstDofView q,
                  runtime::DofView qdot,
                  double dt) override;
  [[nodiscard]] double evaluateObjective(runtime::ConstDofView q,
                                         runtime::ConstDofView qdot,
                                         double dt) override;

  void updateInternalConstraints(double time, double dt) override;
  void prepareLocalOperator(double dt) override;
  void assembleLocalGradient(runtime::DofView g) const override;
  void applyLocalMatrix(runtime::ConstDofView x,
                        runtime::DofView y) const override;
  void solveLocalSystem(runtime::ConstDofView b,
                        runtime::DofView x) const override;

  void updateGeometry(runtime::GlobalGeometryManager& geometry) const override;
  void mapLocalDirectionToGeometry(runtime::ConstDofView localDq,
                                   runtime::GeometryView globalDx) const override;
  void applyInternalContacts(runtime::ContactStencils contacts) override;
  void scatterContactGradient(
      std::span<const runtime::PointIdx> points,
      runtime::ConstGeometryView pointGradient,
      runtime::DofView g) const override;
  void applyContactGeometryHessianProduct(
      std::span<const runtime::PointIdx> gradientPoints,
      runtime::ConstGeometryView pointGradient,
      std::span<const runtime::PointIdx> productPoints,
      runtime::ConstGeometryView pointHessianProduct,
      runtime::ConstDofView localDq,
      runtime::DofView localY) const override;
  void applyInternalContactHessian(runtime::ConstDofView localDq,
                                   runtime::DofView localY) const override;
  void visit(runtime::CpuSubsystemBackend& backend) override;
  void visit(runtime::GpuSubsystemBackend& backend) override;

  [[nodiscard]] const std::vector<ABDBodyDesc>& bodies() const noexcept;
  [[nodiscard]] const std::vector<ABDBodyRuntimeRef>& runtimeBodies()
      const noexcept;
  [[nodiscard]] const std::vector<runtime::PointIdx>& geometryPointIds()
      const noexcept;
  [[nodiscard]] int localScalarCount() const noexcept;
  [[nodiscard]] int findGeometryPoint(runtime::PointIdx point) const noexcept;
  [[nodiscard]] int geometryPointBody(int sample) const;
  [[nodiscard]] int geometryPointVertex(int sample) const;
  [[nodiscard]] const glm::dmat4x3& restTransform(int body) const;
  [[nodiscard]] const glm::dvec3& gravity() const noexcept;
  [[nodiscard]] glm::dmat4x3 bodyTransform(int body) const;
  [[nodiscard]] glm::dmat4x3 bodyVelocity(int body) const;
  void setBodyTransform(int body, const glm::dmat4x3& transform);
  void setBodyVelocity(int body, const glm::dmat4x3& velocity);

 private:
  friend class ABDCPUBackend;

  runtime::SubsystemIdx id_;
  runtime::DofRange range_;
  glm::dvec3 gravity_;
  std::vector<ABDBodyDesc> bodies_;
  std::vector<ABDBodyRuntimeRef> runtime_bodies_;
  std::vector<runtime::PointIdx> geometry_points_;
  std::vector<int> geometry_point_bodies_;
  std::vector<int> geometry_point_vertices_;
  std::vector<glm::dmat4x3> rest_transforms_;
  runtime::ContactStencils internal_contacts_;
  std::unique_ptr<ABDCPUBackend> cpu_backend_;
};

[[nodiscard]] glm::dmat4 affineToMat4(const glm::dmat4x3& affine) noexcept;

}  // namespace ksk::abd
