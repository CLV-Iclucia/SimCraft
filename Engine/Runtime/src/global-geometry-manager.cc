#include <Runtime/global-geometry-manager.h>

#include <stdexcept>
#include <utility>

namespace ksk::runtime {
namespace {

std::uint32_t collisionGroupBit(std::uint32_t group) noexcept {
  if (group >= 32u) {
    return 0u;
  }
  return 1u << group;
}

GeometryBounds transformBounds(const GeometryBounds& local_bounds,
                               const glm::dmat4& transform) {
  GeometryBounds world_bounds;
  for (int x = 0; x < 2; ++x) {
    for (int y = 0; y < 2; ++y) {
      for (int z = 0; z < 2; ++z) {
        const glm::dvec3 corner{
            x == 0 ? local_bounds.lo.x : local_bounds.hi.x,
            y == 0 ? local_bounds.lo.y : local_bounds.hi.y,
            z == 0 ? local_bounds.lo.z : local_bounds.hi.z,
        };
        world_bounds.expand(glm::dvec3(transform * glm::dvec4(corner, 1.0)));
      }
    }
  }
  return world_bounds;
}

}  // namespace

void GlobalGeometryManager::clear() noexcept {
  points.clear();
  edges.clear();
  triangles.clear();
  tets.clear();
  prototypes.clear();
  instances.clear();
  geometryBodies.clear();
  bodyCollisionFilters.clear();
  point_ranges_.clear();
  edge_ranges_.clear();
  triangle_ranges_.clear();
  tet_ranges_.clear();
  collider_point_ranges_.clear();
  collider_edge_ranges_.clear();
  collider_triangle_ranges_.clear();
  body_point_ranges_.clear();
  body_edge_ranges_.clear();
  body_triangle_ranges_.clear();
  body_instance_ranges_.clear();
  subsystem_default_bodies_.clear();
  collider_default_bodies_.clear();
}

PointIdx GlobalGeometryManager::addPoint(SubsystemIdx subsystem,
                                         int localSampleId,
                                         const glm::dvec3& position) {
  return addPoint(subsystem, localSampleId, position, position, 0.0);
}

PointIdx GlobalGeometryManager::addPoint(SubsystemIdx subsystem,
                                         int localSampleId,
                                         const glm::dvec3& position,
                                         Real radius) {
  return addPoint(subsystem, localSampleId, position, position, radius);
}

PointIdx GlobalGeometryManager::addPoint(SubsystemIdx subsystem,
                                         int localSampleId,
                                         const glm::dvec3& position,
                                         const glm::dvec3& restPosition) {
  return addPoint(subsystem, localSampleId, position, restPosition, 0.0);
}

PointIdx GlobalGeometryManager::addPoint(SubsystemIdx subsystem,
                                         int localSampleId,
                                         const glm::dvec3& position,
                                         const glm::dvec3& restPosition,
                                         Real radius) {
  return addPoint(
      GeometryOwner{
          .subsystem = subsystem,
      },
      localSampleId, position, restPosition, radius, -1, -1);
}

PointIdx GlobalGeometryManager::addColliderPoint(int collider,
                                                 int localSampleId,
                                                 const glm::dvec3& position) {
  return addColliderPoint(collider, localSampleId, position, position, 0.0);
}

PointIdx GlobalGeometryManager::addColliderPoint(int collider,
                                                 int localSampleId,
                                                 const glm::dvec3& position,
                                                 Real radius) {
  return addColliderPoint(collider, localSampleId, position, position, radius);
}

PointIdx GlobalGeometryManager::addColliderPoint(
    int collider, int localSampleId, const glm::dvec3& position,
    const glm::dvec3& restPosition) {
  return addColliderPoint(collider, localSampleId, position, restPosition, 0.0);
}

PointIdx GlobalGeometryManager::addColliderPoint(int collider,
                                                 int localSampleId,
                                                 const glm::dvec3& position,
                                                 const glm::dvec3& restPosition,
                                                 Real radius) {
  return addPoint(
      GeometryOwner{
          .collider = collider,
      },
      localSampleId, position, restPosition, radius, -1, -1);
}

PointIdx GlobalGeometryManager::addPoint(
    const GeometryOwner& owner, int localSampleId, const glm::dvec3& position,
    const glm::dvec3& restPosition, Real radius, GeometryInstanceIdx instance,
    int instanceVertex) {
  const GeometryOwner actual_owner = ownerWithDefaultBody(owner);
  PointIdx id = static_cast<int>(points.size());
  if (actual_owner.isSubsystem()) {
    addReference(point_ranges_, actual_owner, id);
  } else if (actual_owner.isCollider()) {
    addReference(collider_point_ranges_, actual_owner, id);
  } else {
    throw std::invalid_argument("geometry point owner is invalid");
  }
  addBodyReference(body_point_ranges_, actual_owner, id);

  points.push_back(GeometryPoint{
      .owner = actual_owner,
      .localSampleId = localSampleId,
      .instance = instance,
      .x = position,
      .restX = restPosition,
      .radius = radius,
  });
  return id;
}

int GlobalGeometryManager::addEdge(PointIdx p0, PointIdx p1, Real radius) {

  const GeometryPoint& first = checkedPoint(p0);
  const GeometryPoint& second = checkedPoint(p1);

  if (first.owner.body != second.owner.body ||
      first.owner.subsystem != second.owner.subsystem ||
      first.owner.collider != second.owner.collider) {
    throw std::invalid_argument(
        "geometry edge endpoints belong to different owners");
  }
  const int id = edges.size();
  const GeometryReference ref =
      first.owner.isSubsystem()
          ? addReference(edge_ranges_, first.owner, id)
          : addReference(collider_edge_ranges_, first.owner, id);
  addBodyReference(body_edge_ranges_, first.owner, id);
  edges.push_back(GeometryEdge{
      .ref = ref,
      .p0 = p0,
      .p1 = p1,
      .radius = radius,
  });
  return id;
}

int GlobalGeometryManager::addTriangle(PointIdx p0, PointIdx p1, PointIdx p2,
                                       Real radius) {
  const GeometryPoint& first = checkedPoint(p0);
  const GeometryPoint& second = checkedPoint(p1);
  const GeometryPoint& third = checkedPoint(p2);
  if (first.owner.body != second.owner.body ||
      first.owner.body != third.owner.body ||
      first.owner.subsystem != second.owner.subsystem ||
      first.owner.subsystem != third.owner.subsystem ||
      first.owner.collider != second.owner.collider ||
      first.owner.collider != third.owner.collider) {
    throw std::invalid_argument(
        "geometry triangle vertices belong to different owners");
  }
  const int id = static_cast<int>(triangles.size());
  const GeometryReference ref =
      first.owner.isSubsystem()
          ? addReference(triangle_ranges_, first.owner, id)
          : addReference(collider_triangle_ranges_, first.owner, id);
  addBodyReference(body_triangle_ranges_, first.owner, id);
  triangles.push_back(GeometryTriangle{
      .ref = ref,
      .p0 = p0,
      .p1 = p1,
      .p2 = p2,
      .thickness = radius,
  });
  return id;
}

int GlobalGeometryManager::addTet(PointIdx p0, PointIdx p1, PointIdx p2,
                                  PointIdx p3) {
  const GeometryPoint& first = checkedPoint(p0);
  const GeometryPoint& second = checkedPoint(p1);
  const GeometryPoint& third = checkedPoint(p2);
  const GeometryPoint& fourth = checkedPoint(p3);
  if (first.owner.body != second.owner.body ||
      first.owner.body != third.owner.body ||
      first.owner.body != fourth.owner.body ||
      first.owner.subsystem != second.owner.subsystem ||
      first.owner.subsystem != third.owner.subsystem ||
      first.owner.subsystem != fourth.owner.subsystem ||
      first.owner.collider != second.owner.collider ||
      first.owner.collider != third.owner.collider ||
      first.owner.collider != fourth.owner.collider) {
    throw std::invalid_argument(
        "geometry tet vertices belong to different owners");
  }
  if (!first.owner.isSubsystem()) {
    throw std::invalid_argument("geometry tets must belong to a subsystem");
  }

  const int id = static_cast<int>(tets.size());
  const GeometryReference ref = addReference(tet_ranges_, first.owner, id);
  tets.push_back(GeometryTet{
      .ref = ref,
      .p0 = p0,
      .p1 = p1,
      .p2 = p2,
      .p3 = p3,
  });
  return id;
}

GeometryBodyIdx GlobalGeometryManager::addGeometryBody(
    GeometryBodyInfo info, GeometryBodyCollisionFilter collision_filter) {
  const GeometryBodyIdx id = geometryBodies.size();
  geometryBodies.push_back(info);
  bodyCollisionFilters.emplace_back(collision_filter);
  return id;
}

GeometryPrototypeIdx GlobalGeometryManager::addGeometryPrototype(
    GeometryMeshDesc mesh) {
  if (mesh.vertices.empty()) {
    throw std::invalid_argument("geometry prototype mesh has no vertices");
  }
  const int vertex_count = static_cast<int>(mesh.vertices.size());
  for (const auto& edge : mesh.edges) {
    if (edge[0] < 0 || edge[0] >= vertex_count || edge[1] < 0 ||
        edge[1] >= vertex_count) {
      throw std::out_of_range("geometry prototype edge vertex is out of range");
    }
  }
  for (const auto& triangle : mesh.triangles) {
    if (triangle[0] < 0 || triangle[0] >= vertex_count || triangle[1] < 0 ||
        triangle[1] >= vertex_count || triangle[2] < 0 ||
        triangle[2] >= vertex_count) {
      throw std::out_of_range(
          "geometry prototype triangle vertex is out of range");
    }
  }

  GeometryBounds local_bounds;
  for (const glm::dvec3& vertex : mesh.vertices) {
    local_bounds.expand(vertex);
  }

  const GeometryPrototypeIdx id = static_cast<int>(prototypes.size());
  prototypes.push_back(GeometryPrototype{
      .mesh = std::move(mesh),
      .bounds = local_bounds,
  });
  return id;
}

GeometryInstanceIdx GlobalGeometryManager::addInstance(
    SubsystemIdx subsystem, int localInstanceId, const GeometryMeshDesc& mesh,
    const glm::dmat4& transform) {
  return addInstance(
      GeometryOwner{
          .subsystem = subsystem,
      },
      localInstanceId, mesh, transform);
}

GeometryInstanceIdx GlobalGeometryManager::addColliderInstance(
    int collider, int localInstanceId, const GeometryMeshDesc& mesh,
    const glm::dmat4& transform) {
  return addInstance(
      GeometryOwner{
          .collider = collider,
      },
      localInstanceId, mesh, transform);
}

GeometryInstanceIdx GlobalGeometryManager::addInstance(
    const GeometryOwner& owner, int localInstanceId,
    const GeometryMeshDesc& mesh, const glm::dmat4& transform) {
  const GeometryPrototypeIdx prototype = addGeometryPrototype(mesh);
  return addInstance(owner, prototype, localInstanceId, transform);
}

GeometryInstanceIdx GlobalGeometryManager::addInstance(
    GeometryOwner owner, GeometryPrototypeIdx prototype, int localInstanceId,
    const glm::dmat4& transform) {
  owner = ownerWithDefaultBody(owner);
  const GeometryPrototype& prototype_entry = checkedPrototype(prototype);
  const GeometryMeshDesc& mesh = prototype_entry.mesh;
  const GeometryInstanceIdx instance_id = static_cast<int>(instances.size());
  const GeometryRange point_range{
      .first = static_cast<int>(points.size()),
      .count = static_cast<int>(mesh.vertices.size()),
  };
  for (int vertex = 0; vertex < mesh.vertices.size(); vertex++) {
    addPoint(owner, vertex, mesh.vertices[vertex],
             mesh.vertices[vertex], mesh.radius,
             instance_id, vertex);
  }

  const GeometryRange edge_range{
      .first = static_cast<int>(edges.size()),
      .count = static_cast<int>(mesh.edges.size()),
  };
  for (const auto& edge : mesh.edges) {
    addEdge(PointIdx{point_range.first + edge[0]},
            PointIdx{point_range.first + edge[1]}, mesh.radius);
  }

  const GeometryRange triangle_range{
      .first = static_cast<int>(triangles.size()),
      .count = static_cast<int>(mesh.triangles.size()),
  };
  for (const auto& triangle : mesh.triangles) {
    addTriangle(PointIdx{point_range.first + triangle[0]},
                PointIdx{point_range.first + triangle[1]},
                PointIdx{point_range.first + triangle[2]},
                2.0 * mesh.radius);
  }

  instances.push_back(GeometryInstance{
      .owner = owner,
      .prototype = prototype,
      .points = point_range,
      .edges = edge_range,
      .triangles = triangle_range,
      .transform = transform,
  });
  addBodyReference(body_instance_ranges_, owner, instance_id);
  return instance_id;
}

bool GlobalGeometryManager::contains(PointIdx point) const noexcept {
  return point >= 0 && point < static_cast<int>(points.size());
}

int GlobalGeometryManager::pointCount() const noexcept {
  return static_cast<int>(points.size());
}

int GlobalGeometryManager::edgeCount() const noexcept {
  return static_cast<int>(edges.size());
}

int GlobalGeometryManager::triangleCount() const noexcept {
  return static_cast<int>(triangles.size());
}

int GlobalGeometryManager::tetCount() const noexcept {
  return static_cast<int>(tets.size());
}

GeometryRange GlobalGeometryManager::pointRange(
    SubsystemIdx subsystem) const noexcept {
  return rangeFor(point_ranges_, subsystem);
}

GeometryRange GlobalGeometryManager::edgeRange(
    SubsystemIdx subsystem) const noexcept {
  return rangeFor(edge_ranges_, subsystem);
}

GeometryRange GlobalGeometryManager::triangleRange(
    SubsystemIdx subsystem) const noexcept {
  return rangeFor(triangle_ranges_, subsystem);
}

GeometryRange GlobalGeometryManager::tetRange(
    SubsystemIdx subsystem) const noexcept {
  return rangeFor(tet_ranges_, subsystem);
}

GeometryRange GlobalGeometryManager::colliderPointRange(
    int collider) const noexcept {
  return rangeFor(collider_point_ranges_, collider);
}

GeometryRange GlobalGeometryManager::colliderEdgeRange(
    int collider) const noexcept {
  return rangeFor(collider_edge_ranges_, collider);
}

GeometryRange GlobalGeometryManager::colliderTriangleRange(
    int collider) const noexcept {
  return rangeFor(collider_triangle_ranges_, collider);
}

GeometryRange GlobalGeometryManager::bodyPointRange(
    GeometryBodyIdx body) const noexcept {
  return rangeFor(body_point_ranges_, body);
}

GeometryRange GlobalGeometryManager::bodyEdgeRange(
    GeometryBodyIdx body) const noexcept {
  return rangeFor(body_edge_ranges_, body);
}

GeometryRange GlobalGeometryManager::bodyTriangleRange(
    GeometryBodyIdx body) const noexcept {
  return rangeFor(body_triangle_ranges_, body);
}

GeometryRange GlobalGeometryManager::bodyInstanceRange(
    GeometryBodyIdx body) const noexcept {
  return rangeFor(body_instance_ranges_, body);
}

GeometryRange GlobalGeometryManager::instancePointRange(
    GeometryInstanceIdx instance) const {
  return checkedInstance(instance).points;
}

GeometryRange GlobalGeometryManager::instanceEdgeRange(
    GeometryInstanceIdx instance) const {
  return checkedInstance(instance).edges;
}

GeometryRange GlobalGeometryManager::instanceTriangleRange(
    GeometryInstanceIdx instance) const {
  return checkedInstance(instance).triangles;
}

GeometryReference GlobalGeometryManager::pointRef(PointIdx point) const {
  const GeometryPoint& entry = checkedPoint(point);
  return {
      .owner = entry.owner,
      .localIndex = entry.localSampleId,
  };
}

GeometryReference GlobalGeometryManager::edgeRef(int edge) const {
  return checkedEdge(edge).ref;
}

GeometryReference GlobalGeometryManager::triangleRef(int triangle) const {
  return checkedTriangle(triangle).ref;
}

GeometryReference GlobalGeometryManager::tetRef(int tet) const {
  return checkedTet(tet).ref;
}

GeometryOwner GlobalGeometryManager::pointOwner(PointIdx point) const {
  return checkedPoint(point).owner;
}

GeometryOwner GlobalGeometryManager::edgeOwner(int edge) const {
  return checkedEdge(edge).ref.owner;
}

GeometryOwner GlobalGeometryManager::triangleOwner(int triangle) const {
  return checkedTriangle(triangle).ref.owner;
}

const GeometryBodyInfo& GlobalGeometryManager::geometryBody(
    GeometryBodyIdx body) const {
  assert(body >= 0 && body < static_cast<int>(geometryBodies.size()));
  return geometryBodies[body];
}

bool GlobalGeometryManager::isDynamicBody(GeometryBodyIdx body) const {
  if (body < 0 || body >= static_cast<int>(geometryBodies.size())) {
    return false;
  }
  return geometryBodies[body].motion & Dynamic;
}

bool GlobalGeometryManager::isKinematicBody(GeometryBodyIdx body) const {
  if (body < 0 || body >= static_cast<int>(geometryBodies.size())) {
    return false;
  }
  return geometryBodies[body].motion & Kinematic;
}

bool GlobalGeometryManager::isCollisionEnabled(GeometryBodyIdx body) const {
  if (body < 0 || body >= static_cast<int>(bodyCollisionFilters.size())) {
    return false;
  }
  return bodyCollisionFilters[body].enabled;
}

bool GlobalGeometryManager::collisionPairEnabled(GeometryBodyIdx a,
                                                 GeometryBodyIdx b) const {

  if (!isCollisionEnabled(a) || !isCollisionEnabled(b))
    return false;

  const GeometryBodyCollisionFilter& first =
      bodyCollisionFilters[a];
  const GeometryBodyCollisionFilter& second =
      bodyCollisionFilters[b];
  const std::uint32_t first_group = collisionGroupBit(first.group);
  const std::uint32_t second_group = collisionGroupBit(second.group);
  return first_group != 0u && second_group != 0u &&
         (first.mask & second_group) != 0u &&
         (second.mask & first_group) != 0u;
}

bool GlobalGeometryManager::sameSubsystem(
    std::span<const PointIdx> stencil) const {
  const GeometryStencilInfo info = classify(stencil);
  return info.valid && !info.hasCollider && info.subsystemCount == 1;
}

bool GlobalGeometryManager::hasCollider(
    std::span<const PointIdx> stencil) const {
  return classify(stencil).hasCollider;
}

GeometryStencilInfo GlobalGeometryManager::classify(
    std::span<const PointIdx> stencil) const {
  GeometryStencilInfo info;
  for (const PointIdx point : stencil) {
    const GeometryPoint& entry = checkedPoint(point);
    info.hasInstanceGeometry = info.hasInstanceGeometry || entry.instance >= 0;
    const GeometryBodyIdx body = entry.owner.body;
    if (body < 0) {
      info.valid = false;
    } else if (info.bodyCount == 0) {
      info.bodies[0] = body;
      info.bodyCount = 1;
    } else if (info.bodies[0] != body) {
      if (info.bodyCount == 1) {
        info.bodies[1] = body;
        info.bodyCount = 2;
      } else if (info.bodies[1] != body) {
        info.valid = false;
      }
    }
    if (entry.owner.isCollider()) {
      info.hasCollider = true;
      continue;
    }
    if (!entry.owner.isSubsystem()) {
      info.valid = false;
      continue;
    }

    const SubsystemIdx subsystem = entry.owner.subsystem;
    if (info.subsystemCount == 0) {
      info.subsystems[0] = subsystem;
      info.subsystemCount = 1;
      continue;
    }
    if (info.subsystems[0] == subsystem) {
      continue;
    }
    if (info.subsystemCount == 1) {
      info.subsystems[1] = subsystem;
      info.subsystemCount = 2;
      info.crossesSubsystems = true;
      continue;
    }
    if (info.subsystems[1] != subsystem) {
      info.valid = false;
      info.crossesSubsystems = true;
    }
  }
  return info;
}

PointIdx GlobalGeometryManager::instancePoint(GeometryInstanceIdx instance,
                                              int vertex) const {
  const GeometryRange range = checkedInstance(instance).points;
  if (vertex < 0 || vertex >= range.count) {
    return -1;
  }
  return PointIdx{range.first + vertex};
}

int GlobalGeometryManager::instanceEdge(GeometryInstanceIdx instance,
                                        int edge) const {
  const GeometryRange range = checkedInstance(instance).edges;
  if (edge < 0 || edge >= range.count) {
    return -1;
  }
  return range.first + edge;
}

int GlobalGeometryManager::instanceTriangle(GeometryInstanceIdx instance,
                                            int triangle) const {
  const GeometryRange range = checkedInstance(instance).triangles;
  if (triangle < 0 || triangle >= range.count) {
    return -1;
  }
  return range.first + triangle;
}

std::array<PointIdx, 2> GlobalGeometryManager::globalEdge(int edge) const {
  const GeometryEdge& entry = checkedEdge(edge);
  return {entry.p0, entry.p1};
}

std::array<PointIdx, 3> GlobalGeometryManager::globalTriangle(
    int triangle) const {
  const GeometryTriangle& entry = checkedTriangle(triangle);
  return {entry.p0, entry.p1, entry.p2};
}

std::array<PointIdx, 4> GlobalGeometryManager::globalTet(int tet) const {
  const GeometryTet& entry = checkedTet(tet);
  return {entry.p0, entry.p1, entry.p2, entry.p3};
}

bool GlobalGeometryManager::triangleContainsPoint(int triangle,
                                                  PointIdx point) const {
  const auto vertices = globalTriangle(triangle);
  return vertices[0] == point || vertices[1] == point || vertices[2] == point;
}

bool GlobalGeometryManager::edgesAdjacent(int edgeA, int edgeB) const {
  const auto a = globalEdge(edgeA);
  const auto b = globalEdge(edgeB);
  return a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1];
}

GeometryBounds GlobalGeometryManager::pointBounds(PointIdx point) const {
  GeometryBounds bounds;
  bounds.expand(worldPosition(point));
  return bounds;
}

GeometryBounds GlobalGeometryManager::edgeBounds(int edge) const {
  GeometryBounds bounds;
  const auto vertices = globalEdge(edge);
  bounds.expand(worldPosition(vertices[0]));
  bounds.expand(worldPosition(vertices[1]));
  return bounds;
}

GeometryBounds GlobalGeometryManager::triangleBounds(int triangle) const {
  GeometryBounds bounds;
  const auto vertices = globalTriangle(triangle);
  for (const PointIdx point : vertices) {
    bounds.expand(worldPosition(point));
  }
  return bounds;
}

GeometryBounds GlobalGeometryManager::instanceBounds(
    GeometryInstanceIdx instance) const {
  const GeometryInstance& entry = checkedInstance(instance);
  const GeometryPrototype& prototype = checkedPrototype(entry.prototype);
  return transformBounds(prototype.bounds, entry.transform);
}

GeometryBounds GlobalGeometryManager::bodyBounds(GeometryBodyIdx body) const {
  GeometryBounds bounds;
  const GeometryRange instances = bodyInstanceRange(body);
  for (int instance = instances.first; instance < instances.end(); ++instance) {
    bounds.expand(instanceBounds(instance));
  }
  if (instances.count > 0) {
    return bounds;
  }

  const GeometryRange points = bodyPointRange(body);
  for (int point = points.first; point < points.end(); ++point) {
    bounds.expand(worldPosition(PointIdx{point}));
  }
  return bounds;
}

GeometryBounds GlobalGeometryManager::trajectoryPointBounds(
    PointIdx point, std::span<const glm::dvec3> directions, Real toi) const {
  checkedPoint(point);
  const glm::dvec3 position = worldPosition(point);
  GeometryBounds bounds;
  bounds.expand(position);
  bounds.expand(position + directions[point] * toi);
  return bounds;
}

GeometryBounds GlobalGeometryManager::trajectoryEdgeBounds(
    int edge, std::span<const glm::dvec3> directions, Real toi,
    Real radius) const {
  GeometryBounds bounds;
  const auto vertices = globalEdge(edge);
  for (const PointIdx point : vertices) {
    const GeometryBounds point_bounds =
        trajectoryPointBounds(point, directions, toi);
    bounds.expand(point_bounds.lo);
    bounds.expand(point_bounds.hi);
  }
  return bounds;
}

glm::dvec3 GlobalGeometryManager::worldPosition(PointIdx point) const {
  const GeometryPoint& entry = checkedPoint(point);
  if (entry.instance < 0) {
    return entry.x;
  }
  const GeometryInstance& instance = checkedInstance(entry.instance);
  return glm::dvec3(instance.transform * glm::dvec4(entry.x, 1.0));
}

glm::dvec3 GlobalGeometryManager::restPosition(PointIdx point) const {
  const GeometryPoint& entry = checkedPoint(point);
  if (entry.instance < 0) {
    return entry.restX;
  }
  const GeometryInstance& instance = checkedInstance(entry.instance);
  return glm::dvec3(instance.transform * glm::dvec4(entry.restX, 1.0));
}

void GlobalGeometryManager::setPointPosition(PointIdx point,
                                             const glm::dvec3& position) {
  GeometryPoint& entry = checkedPoint(point);
  if (entry.instance >= 0) {
    throw std::invalid_argument(
        "cannot directly set an instance vertex position");
  }
  entry.x = position;
}

void GlobalGeometryManager::setPointRestPosition(
    PointIdx point, const glm::dvec3& restPosition) {
  GeometryPoint& entry = checkedPoint(point);
  if (entry.instance >= 0) {
    throw std::invalid_argument(
        "cannot directly set an instance vertex rest position");
  }
  entry.restX = restPosition;
}

void GlobalGeometryManager::setInstanceTransform(GeometryInstanceIdx instance,
                                                 const glm::dmat4& transform) {
  checkedInstance(instance).transform = transform;
}

spatify::BBox<Real, 3> GlobalGeometryManager::trajectoryTriangleBounds(
    int triangle, std::span<const glm::dvec3> directions, Real toi,
    Real thickness) const {
  GeometryBounds bounds;
  const auto vertices = globalTriangle(triangle);
  for (const PointIdx point : vertices) {
    const GeometryBounds point_bounds =
        trajectoryPointBounds(point, directions, toi);
    bounds.expand(point_bounds.lo);
    bounds.expand(point_bounds.hi);
  }
  return bounds;
}

GeometryRange GlobalGeometryManager::rangeFor(
    const std::vector<GeometryRange>& ranges, int ownerIndex) const noexcept {
  if (ownerIndex < 0 || ownerIndex >= static_cast<int>(ranges.size())) {
    return {};
  }
  return ranges[static_cast<size_t>(ownerIndex)];
}

GeometryReference GlobalGeometryManager::addReference(
    std::vector<GeometryRange>& ranges, const GeometryOwner& owner,
    int nextGlobalIndex) {
  int owner_index = -1;
  if (owner.isSubsystem()) {
    owner_index = owner.subsystem;
  } else if (owner.isCollider()) {
    owner_index = owner.collider;
  } else {
    throw std::invalid_argument("geometry owner is invalid");
  }
  if (ranges.size() <= static_cast<size_t>(owner_index)) {
    ranges.resize(static_cast<size_t>(owner_index + 1));
  }

  GeometryRange& range = ranges[static_cast<size_t>(owner_index)];
  if (range.count == 0) {
    range.first = nextGlobalIndex;
  } else if (nextGlobalIndex != range.end()) {
    throw std::invalid_argument(
        "geometry for a subsystem must be appended contiguously");
  }

  const GeometryReference ref{
      .owner = owner,
      .localIndex = range.count,
  };
  ++range.count;
  return ref;
}

GeometryReference GlobalGeometryManager::addBodyReference(
    std::vector<GeometryRange>& ranges, const GeometryOwner& owner,
    int nextGlobalIndex) {

  if (!owner.hasBody()) {
    throw std::invalid_argument("geometry owner has no body");
  }

  if (ranges.size() <= static_cast<size_t>(owner.body)) {
    ranges.resize(static_cast<size_t>(owner.body + 1));
  }

  GeometryRange& range = ranges[owner.body];
  if (range.count == 0) {
    range.first = nextGlobalIndex;
  } else if (nextGlobalIndex != range.end()) {
    throw std::invalid_argument(
        "geometry for a body must be appended contiguously");
  }

  const GeometryReference ref{
      .owner = owner,
      .localIndex = range.count,
  };
  ++range.count;
  return ref;
}

GeometryBodyIdx GlobalGeometryManager::defaultSubsystemBody(
    SubsystemIdx subsystem) {

  if (subsystem < 0)
    throw std::invalid_argument("subsystem id is invalid");

  if (subsystem_default_bodies_.size() <= static_cast<size_t>(subsystem))
    subsystem_default_bodies_.resize(static_cast<size_t>(subsystem + 1), -1);

  GeometryBodyIdx& body =
      subsystem_default_bodies_[static_cast<size_t>(subsystem)];
  if (body < 0) {
    body = addGeometryBody(
        GeometryBodyInfo{
            .type = Deformable,
            .motion = Dynamic,
            .subsystem = subsystem,
            .localBody = 0,
        },
        GeometryBodyCollisionFilter{
            .group = kDefaultCollisionGroup,
            .mask = kAllCollisionGroups,
            .enabled = true});
  }
  return body;
}

GeometryBodyIdx GlobalGeometryManager::defaultColliderBody(int collider) {
  if (collider < 0) {
    throw std::invalid_argument("collider id is invalid");
  }
  if (collider_default_bodies_.size() <= collider) {
    collider_default_bodies_.resize(collider + 1, -1);
  }
  GeometryBodyIdx& body = collider_default_bodies_[collider];
  if (body < 0) {
    body = addGeometryBody(
        GeometryBodyInfo{
            .type = Rigid,
            .motion = Static,
            .subsystem = -1,
            .localBody = collider,
        },
        GeometryBodyCollisionFilter{
            .group = kDefaultCollisionGroup,
            .mask = kAllCollisionGroups,
            .enabled = true});
  }
  return body;
}

GeometryOwner GlobalGeometryManager::ownerWithDefaultBody(
    const GeometryOwner& owner) {
  GeometryOwner result = owner;
  if (result.hasBody()) {
    const GeometryBodyInfo& body = geometryBody(result.body);
    if (result.subsystem < 0) {
      result.subsystem = body.subsystem;
    }
    if (result.collider < 0 && (body.type & Rigid) &&
        (body.motion & Static) && body.subsystem < 0) {
      result.collider = body.localBody;
    }
    return result;
  }

  if (result.isSubsystem()) {
    result.body = defaultSubsystemBody(result.subsystem);
    return result;
  }

  if (result.isCollider()) {
    result.body = defaultColliderBody(result.collider);
    return result;
  }

  throw std::invalid_argument("geometry owner is invalid");
}

const GeometryPrototype& GlobalGeometryManager::checkedPrototype(
    GeometryPrototypeIdx prototype) const {
  if (prototype < 0 || prototype >= static_cast<int>(prototypes.size())) {
    throw std::out_of_range("geometry prototype id is out of range");
  }
  return prototypes[static_cast<size_t>(prototype)];
}

const GeometryPoint& GlobalGeometryManager::checkedPoint(PointIdx point) const {
  assert(contains(point));
  return points[point];
}

GeometryPoint& GlobalGeometryManager::checkedPoint(PointIdx point) {
  assert(contains(point));
  return points[point];
}

const GeometryEdge& GlobalGeometryManager::checkedEdge(int edge) const {
  assert(edge >= 0 && edge < static_cast<int>(edges.size()));
  return edges[edge];
}

const GeometryTriangle& GlobalGeometryManager::checkedTriangle(
    int triangle) const {
  assert(triangle >= 0 && triangle < triangles.size());
  return triangles[triangle];
}

const GeometryTet& GlobalGeometryManager::checkedTet(int tet) const {
  assert(tet >= 0 && tet < static_cast<int>(tets.size()));
  return tets[static_cast<size_t>(tet)];
}

const GeometryInstance& GlobalGeometryManager::checkedInstance(
    GeometryInstanceIdx instance) const {
  if (instance < 0 || instance >= static_cast<int>(instances.size())) {
    throw std::out_of_range("geometry instance id is out of range");
  }
  return instances[static_cast<size_t>(instance)];
}

GeometryInstance& GlobalGeometryManager::checkedInstance(
    GeometryInstanceIdx instance) {
  if (instance < 0 || instance >= static_cast<int>(instances.size())) {
    throw std::out_of_range("geometry instance id is out of range");
  }
  return instances[static_cast<size_t>(instance)];
}

}  // namespace ksk::runtime
