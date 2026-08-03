#pragma once

#include <Runtime/types.h>
#include <Spatify/bbox.h>

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace ksk::runtime {

using GeometryInstanceIdx = int;
using GeometryPrototypeIdx = int;
using GeometryBodyIdx = int;

enum EGeometryBody {
  Deformable = 1,
  Rigid = 2,
};

enum EBodyMotionMode {
  None = 0,
  Dynamic = 1,
  Kinematic = 2,
  Static = 4,
};

inline constexpr std::uint32_t kDefaultCollisionGroup = 0u;
inline constexpr std::uint32_t kAllCollisionGroups = 0xffffffffu;

struct GeometryBodyInfo {
  EGeometryBody type = Deformable;
  EBodyMotionMode motion = Static;
  SubsystemIdx subsystem = -1;
  int localBody = -1;
};

struct GeometryBodyCollisionFilter {
  std::uint32_t group = kDefaultCollisionGroup;
  std::uint32_t mask = kAllCollisionGroups;
  bool enabled = true;
  Real frictionCoefficient = -1.0;
};

using GeometryBodyCollisionFilters = std::vector<GeometryBodyCollisionFilter>;

struct GeometryOwner {
  GeometryBodyIdx body = -1;
  SubsystemIdx subsystem = -1;
  int collider = -1;

  [[nodiscard]] bool hasBody() const noexcept { return body >= 0; }
  [[nodiscard]] bool isSubsystem() const noexcept {
    return subsystem >= 0 && collider < 0;
  }
  [[nodiscard]] bool isCollider() const noexcept {
    return subsystem < 0 && collider >= 0;
  }
};

struct GeometryReference {
  GeometryOwner owner;
  int localIndex = -1;
};

struct GeometryRange {
  int first = 0;
  int count = 0;

  [[nodiscard]] int end() const noexcept { return first + count; }
  [[nodiscard]] bool contains(int index) const noexcept {
    return index >= first && index < end();
  }
};

using GeometryBounds = spatify::BBox<Real, 3>;
struct GeometryPoint {
  GeometryOwner owner;
  int localSampleId = -1;
  GeometryInstanceIdx instance = -1;
  glm::dvec3 x{0.0};
  glm::dvec3 restX{0.0};
  Real radius = 0.0;
};

struct GeometryEdge {
  GeometryReference ref;
  PointIdx p0 = -1;
  PointIdx p1 = -1;
  Real radius = 0.0;
};

struct GeometryTriangle {
  GeometryReference ref;
  PointIdx p0 = -1;
  PointIdx p1 = -1;
  PointIdx p2 = -1;
  Real thickness = 0.0;
};

struct GeometryTet {
  GeometryReference ref;
  PointIdx p0 = -1;
  PointIdx p1 = -1;
  PointIdx p2 = -1;
  PointIdx p3 = -1;
};

struct GeometryMeshDesc {
  std::vector<glm::dvec3> vertices;
  std::vector<std::array<int, 2>> edges;
  std::vector<std::array<int, 3>> triangles;
  Real radius = 0.0;
};

struct GeometryPrototype {
  GeometryMeshDesc mesh;
  GeometryBounds bounds;
};

struct GeometryInstance {
  GeometryOwner owner;
  GeometryPrototypeIdx prototype = -1;
  GeometryRange points;
  GeometryRange edges;
  GeometryRange triangles;
  glm::dmat4 transform{1.0};
};

struct GeometryStencilInfo {
  bool valid = true;
  bool hasCollider = false;
  bool hasInstanceGeometry = false;
  bool crossesSubsystems = false;
  int subsystemCount = 0;
  std::array<SubsystemIdx, 2> subsystems{};
  int bodyCount = 0;
  std::array<GeometryBodyIdx, 2> bodies{-1, -1};
};

struct GlobalGeometryManager {
  std::vector<GeometryPoint> points;
  std::vector<GeometryEdge> edges;
  std::vector<GeometryTriangle> triangles;
  std::vector<GeometryTet> tets;
  std::vector<GeometryPrototype> prototypes;
  std::vector<GeometryInstance> instances;
  std::vector<GeometryBodyInfo> geometryBodies;
  GeometryBodyCollisionFilters bodyCollisionFilters;

  bool isInstanceBody(GeometryBodyIdx body_id) const noexcept {
    return geometryBodies[body_id].type & Rigid;
  }

  void clear() noexcept;

  [[nodiscard]] PointIdx addPoint(SubsystemIdx subsystem, int localSampleId,
                                  const glm::dvec3& position);

  [[nodiscard]] PointIdx addPoint(SubsystemIdx subsystem, int localSampleId,
                                  const glm::dvec3& position, Real radius);

  [[nodiscard]] PointIdx addPoint(SubsystemIdx subsystem, int localSampleId,
                                  const glm::dvec3& position,
                                  const glm::dvec3& restPosition);
  [[nodiscard]] PointIdx addPoint(SubsystemIdx subsystem, int localSampleId,
                                  const glm::dvec3& position,
                                  const glm::dvec3& restPosition, Real radius);
  [[nodiscard]] PointIdx addColliderPoint(int collider, int localSampleId,
                                          const glm::dvec3& position);
  [[nodiscard]] PointIdx addColliderPoint(int collider, int localSampleId,
                                          const glm::dvec3& position,
                                          Real radius);
  [[nodiscard]] PointIdx addColliderPoint(int collider, int localSampleId,
                                          const glm::dvec3& position,
                                          const glm::dvec3& restPosition);
  [[nodiscard]] PointIdx addColliderPoint(int collider, int localSampleId,
                                          const glm::dvec3& position,
                                          const glm::dvec3& restPosition,
                                          Real radius);
  [[nodiscard]] int addEdge(PointIdx p0, PointIdx p1, Real radius = 0.0);
  [[nodiscard]] int addTriangle(PointIdx p0, PointIdx p1, PointIdx p2,
                                Real thickness = 0.0);
  [[nodiscard]] int addTet(PointIdx p0, PointIdx p1, PointIdx p2, PointIdx p3);
  [[nodiscard]] GeometryBodyIdx addGeometryBody(
      GeometryBodyInfo info, GeometryBodyCollisionFilter collision_filter);
  [[nodiscard]] GeometryPrototypeIdx addGeometryPrototype(
      GeometryMeshDesc mesh);
  [[nodiscard]] GeometryInstanceIdx addInstance(
      SubsystemIdx subsystem, int localInstanceId, const GeometryMeshDesc& mesh,
      const glm::dmat4& transform = glm::dmat4(1.0));
  [[nodiscard]] GeometryInstanceIdx addColliderInstance(
      int collider, int localInstanceId, const GeometryMeshDesc& mesh,
      const glm::dmat4& transform = glm::dmat4(1.0));
  [[nodiscard]] GeometryInstanceIdx addInstance(
      GeometryOwner owner, GeometryPrototypeIdx prototype, int localInstanceId,
      const glm::dmat4& transform = glm::dmat4(1.0));
  [[nodiscard]] bool contains(PointIdx point) const noexcept;

  [[nodiscard]] int pointCount() const noexcept;
  [[nodiscard]] int edgeCount() const noexcept;
  [[nodiscard]] int triangleCount() const noexcept;
  [[nodiscard]] int tetCount() const noexcept;

  [[nodiscard]] GeometryRange pointRange(SubsystemIdx subsystem) const noexcept;
  [[nodiscard]] GeometryRange edgeRange(SubsystemIdx subsystem) const noexcept;
  [[nodiscard]] GeometryRange triangleRange(
      SubsystemIdx subsystem) const noexcept;
  [[nodiscard]] GeometryRange tetRange(SubsystemIdx subsystem) const noexcept;
  [[nodiscard]] GeometryRange colliderPointRange(int collider) const noexcept;
  [[nodiscard]] GeometryRange colliderEdgeRange(int collider) const noexcept;
  [[nodiscard]] GeometryRange colliderTriangleRange(
      int collider) const noexcept;
  [[nodiscard]] GeometryRange bodyPointRange(
      GeometryBodyIdx body) const noexcept;
  [[nodiscard]] GeometryRange bodyEdgeRange(
      GeometryBodyIdx body) const noexcept;
  [[nodiscard]] GeometryRange bodyTriangleRange(
      GeometryBodyIdx body) const noexcept;
  [[nodiscard]] GeometryRange bodyInstanceRange(
      GeometryBodyIdx body) const noexcept;
  [[nodiscard]] GeometryRange instancePointRange(
      GeometryInstanceIdx instance) const;
  [[nodiscard]] GeometryRange instanceEdgeRange(
      GeometryInstanceIdx instance) const;
  [[nodiscard]] GeometryRange instanceTriangleRange(
      GeometryInstanceIdx instance) const;

  [[nodiscard]] GeometryReference pointRef(PointIdx point) const;
  [[nodiscard]] GeometryReference edgeRef(int edge) const;
  [[nodiscard]] GeometryReference triangleRef(int triangle) const;
  [[nodiscard]] GeometryReference tetRef(int tet) const;

  [[nodiscard]] GeometryOwner pointOwner(PointIdx point) const;
  [[nodiscard]] GeometryOwner edgeOwner(int edge) const;
  [[nodiscard]] GeometryOwner triangleOwner(int triangle) const;
  [[nodiscard]] const GeometryBodyInfo& geometryBody(
      GeometryBodyIdx body) const;
  [[nodiscard]] bool isDynamicBody(GeometryBodyIdx body) const;
  [[nodiscard]] bool isKinematicBody(GeometryBodyIdx body) const;
  [[nodiscard]] bool isCollisionEnabled(GeometryBodyIdx body) const;
  [[nodiscard]] bool collisionPairEnabled(GeometryBodyIdx a,
                                          GeometryBodyIdx b) const;
  [[nodiscard]] bool sameSubsystem(std::span<const PointIdx> stencil) const;
  [[nodiscard]] bool hasCollider(std::span<const PointIdx> stencil) const;

  [[nodiscard]] GeometryStencilInfo classify(
      std::span<const PointIdx> stencil) const;

  [[nodiscard]] PointIdx instancePoint(GeometryInstanceIdx instance,
                                       int vertex) const;
  [[nodiscard]] int instanceEdge(GeometryInstanceIdx instance, int edge) const;
  [[nodiscard]] int instanceTriangle(GeometryInstanceIdx instance,
                                     int triangle) const;
  [[nodiscard]] std::array<PointIdx, 2> globalEdge(int edge) const;
  [[nodiscard]] std::array<PointIdx, 3> globalTriangle(int triangle) const;
  [[nodiscard]] std::array<PointIdx, 4> globalTet(int tet) const;

  [[nodiscard]] bool triangleContainsPoint(int triangle, PointIdx point) const;
  [[nodiscard]] bool edgesAdjacent(int edgeA, int edgeB) const;

  [[nodiscard]] GeometryBounds pointBounds(PointIdx point) const;
  [[nodiscard]] GeometryBounds edgeBounds(int edge) const;
  [[nodiscard]] GeometryBounds triangleBounds(int triangle) const;
  [[nodiscard]] GeometryBounds instanceBounds(
      GeometryInstanceIdx instance) const;
  [[nodiscard]] GeometryBounds bodyBounds(GeometryBodyIdx body) const;
  [[nodiscard]] GeometryBounds trajectoryPointBounds(
      PointIdx point, std::span<const glm::dvec3> directions, Real toi) const;
  [[nodiscard]] GeometryBounds trajectoryEdgeBounds(
      int edge, std::span<const glm::dvec3> directions, Real toi,
      Real radius = 0.0) const;
  [[nodiscard]] GeometryBounds trajectoryTriangleBounds(
      int triangle, std::span<const glm::dvec3> directions, Real toi,
      Real thickness = 0.0) const;

  [[nodiscard]] glm::dvec3 worldPosition(PointIdx point) const;
  [[nodiscard]] glm::dvec3 restPosition(PointIdx point) const;
  void setPointPosition(PointIdx point, const glm::dvec3& position);
  void setPointRestPosition(PointIdx point, const glm::dvec3& restPosition);
  void setInstanceTransform(GeometryInstanceIdx instance,
                            const glm::dmat4& transform);

 private:
  [[nodiscard]] GeometryRange rangeFor(const std::vector<GeometryRange>& ranges,
                                       int ownerIndex) const noexcept;
  [[nodiscard]] GeometryReference addReference(
      std::vector<GeometryRange>& ranges, const GeometryOwner& owner,
      int nextGlobalIndex);
  [[nodiscard]] GeometryReference addBodyReference(
      std::vector<GeometryRange>& ranges, const GeometryOwner& owner,
      int nextGlobalIndex);
  [[nodiscard]] PointIdx addPoint(const GeometryOwner& owner, int localSampleId,
                                  const glm::dvec3& position,
                                  const glm::dvec3& restPosition, Real radius,
                                  GeometryInstanceIdx instance,
                                  int instanceVertex);
  [[nodiscard]] GeometryInstanceIdx addInstance(const GeometryOwner& owner,
                                                int localInstanceId,
                                                const GeometryMeshDesc& mesh,
                                                const glm::dmat4& transform);
  [[nodiscard]] GeometryBodyIdx defaultSubsystemBody(SubsystemIdx subsystem);
  [[nodiscard]] GeometryBodyIdx defaultColliderBody(int collider);
  [[nodiscard]] GeometryOwner ownerWithDefaultBody(const GeometryOwner& owner);
  [[nodiscard]] const GeometryPrototype& checkedPrototype(
      GeometryPrototypeIdx prototype) const;
  [[nodiscard]] const GeometryPoint& checkedPoint(PointIdx point) const;
  [[nodiscard]] GeometryPoint& checkedPoint(PointIdx point);
  [[nodiscard]] const GeometryEdge& checkedEdge(int edge) const;
  [[nodiscard]] const GeometryTriangle& checkedTriangle(int triangle) const;
  [[nodiscard]] const GeometryTet& checkedTet(int tet) const;
  [[nodiscard]] const GeometryInstance& checkedInstance(
      GeometryInstanceIdx instance) const;
  [[nodiscard]] GeometryInstance& checkedInstance(GeometryInstanceIdx instance);

  std::vector<GeometryRange> point_ranges_;
  std::vector<GeometryRange> edge_ranges_;
  std::vector<GeometryRange> triangle_ranges_;
  std::vector<GeometryRange> tet_ranges_;
  std::vector<GeometryRange> collider_point_ranges_;
  std::vector<GeometryRange> collider_edge_ranges_;
  std::vector<GeometryRange> collider_triangle_ranges_;
  std::vector<GeometryRange> body_point_ranges_;
  std::vector<GeometryRange> body_edge_ranges_;
  std::vector<GeometryRange> body_triangle_ranges_;
  std::vector<GeometryRange> body_instance_ranges_;
  std::vector<GeometryBodyIdx> subsystem_default_bodies_;
  std::vector<GeometryBodyIdx> collider_default_bodies_;
};

}  // namespace ksk::runtime
