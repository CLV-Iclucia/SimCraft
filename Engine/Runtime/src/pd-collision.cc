#include <Runtime/pd-collision.h>

#include <Contact/shared/ipc-distance.h>
#include <Core/profiler.h>

#include <algorithm>
#include <cmath>
#include <glm/geometric.hpp>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ksk::runtime {
namespace {

namespace contact_shared = ksk::engine::contact::shared;

bool isDeformableOwner(const GeometryOwner& owner) noexcept
{
  return owner.isSubsystem();
}

bool isColliderOwner(const GeometryOwner& owner) noexcept
{
  return owner.isCollider();
}

std::pair<int, int> edgeKey(int u, int v) noexcept
{
  return u < v ? std::make_pair(u, v) : std::make_pair(v, u);
}

// Average push displacement (normal * depth) over every constraint that
// involves the given point. This is the "push from DCD" used to move boundary
// vertices out of penetration.
glm::dvec3 pushDirectionForPoint(const RawPenetrations& raw, PointIdx point)
{
  glm::dvec3 push{0.0};
  int count = 0;
  for (const PdPenetrationConstraint& constraint : raw.constraints) {
    const int n = std::clamp(constraint.pointCount, 0, 4);
    const bool involved =
        std::find(constraint.points.begin(),
                  constraint.points.begin() + n, point) !=
        (constraint.points.begin() + n);
    if (!involved) {
      continue;
    }
    push += constraint.normal * constraint.penetrationDepth;
    ++count;
  }
  if (count > 0) {
    push /= static_cast<double>(count);
  }
  return push;
}

PdPenetrationConstraint makePointTriangleConstraint(
    PointIdx point,
    const glm::dvec3& p,
    PointIdx t0,
    PointIdx t1,
    PointIdx t2,
    const glm::dvec3& a,
    const glm::dvec3& b,
    const glm::dvec3& c,
    double geometricDistance,
    double reserved)
{
  PdPenetrationConstraint constraint;
  constraint.kind = EContactCandidate::PointTriangle;
  constraint.points = {point, t0, t1, t2};
  constraint.pointCount = 4;

  glm::dvec3 normal = glm::cross(b - a, c - a);
  const double normalLength = glm::length(normal);
  if (normalLength > 1.0e-12) {
    normal /= normalLength;
    // Orient A -> B (triangle -> point).
    if (glm::dot(normal, p - a) < 0.0) {
      normal = -normal;
    }
  } else {
    normal = glm::dvec3(0.0, 1.0, 0.0);
  }

  constraint.normal = normal;
  constraint.penetrationDepth = std::max(0.0, reserved - geometricDistance);
  // Approximate closest points. TODO: full closest-point + barycentric
  // computation for the exact push direction.
  constraint.closestA = p - normal * geometricDistance;
  constraint.closestB = p;
  return constraint;
}

PdPenetrationConstraint makeEdgeEdgeConstraint(
    PointIdx e0p0,
    PointIdx e0p1,
    const glm::dvec3& a,
    const glm::dvec3& b,
    PointIdx e1p0,
    PointIdx e1p1,
    const glm::dvec3& c,
    const glm::dvec3& d,
    double geometricDistance,
    double reserved)
{
  PdPenetrationConstraint constraint;
  constraint.kind = EContactCandidate::EdgeEdge;
  constraint.points = {e0p0, e0p1, e1p0, e1p1};
  constraint.pointCount = 4;

  glm::dvec3 normal = glm::cross(b - a, d - c);
  const double normalLength = glm::length(normal);
  if (normalLength > 1.0e-12) {
    normal /= normalLength;
  } else {
    // Parallel or degenerate edges: fall back to the segment start-to-start
    // direction, then to a fixed axis. TODO: robust closest-point handling.
    normal = c - a;
    if (glm::length(normal) > 1.0e-12) {
      normal = glm::normalize(normal);
    } else {
      normal = glm::dvec3(0.0, 1.0, 0.0);
    }
  }

  constraint.normal = normal;
  constraint.penetrationDepth = std::max(0.0, reserved - geometricDistance);
  constraint.closestA = a;
  constraint.closestB = c;
  return constraint;
}

}  // namespace

RawPenetrations detectPenetrations(const GlobalGeometryManager& geometry)
{
  KSK_PROFILE_SCOPE("PdCollision/Detect");
  RawPenetrations raw;
  const auto& points = geometry.points;
  const auto& triangles = geometry.triangles;
  const auto& edges = geometry.edges;

  // --- Self: deformable points vs deformable triangles (PT) ---
  for (int pointIndex = 0; pointIndex < static_cast<int>(points.size());
       ++pointIndex) {
    const GeometryPoint& point = points[static_cast<size_t>(pointIndex)];
    if (!isDeformableOwner(point.owner)) {
      continue;
    }
    const PointIdx pidx = PointIdx{pointIndex};
    const glm::dvec3 p = geometry.worldPosition(pidx);
    for (int triIndex = 0; triIndex < static_cast<int>(triangles.size());
         ++triIndex) {
      const GeometryTriangle& tri = triangles[static_cast<size_t>(triIndex)];
      if (!isDeformableOwner(tri.ref.owner)) {
        continue;
      }
      // Skip triangles that share the point (adjacent primitives).
      if (tri.p0 == pidx || tri.p1 == pidx || tri.p2 == pidx) {
        continue;
      }
      const glm::dvec3 a = geometry.worldPosition(tri.p0);
      const glm::dvec3 b = geometry.worldPosition(tri.p1);
      const glm::dvec3 c = geometry.worldPosition(tri.p2);
      const double distanceSqr = contact_shared::shDistanceSqrPointTriangle(p, a, b, c);
      const double reserved = point.radius + tri.thickness;
      const double distance = std::sqrt(distanceSqr);
      if (distance >= reserved) {
        continue;
      }
      raw.constraints.push_back(makePointTriangleConstraint(
          pidx, p, tri.p0, tri.p1, tri.p2, a, b, c, distance, reserved));
      raw.penetratingVertices.insert(pidx);
      raw.penetratingTriangles.insert(TriangleIdx{triIndex});
    }
  }

  // --- Self: deformable edges vs deformable edges (EE) ---
  for (int edgeA = 0; edgeA < static_cast<int>(edges.size()); ++edgeA) {
    const GeometryEdge& e0 = edges[static_cast<size_t>(edgeA)];
    if (!isDeformableOwner(e0.ref.owner)) {
      continue;
    }
    const glm::dvec3 a = geometry.worldPosition(e0.p0);
    const glm::dvec3 b = geometry.worldPosition(e0.p1);
    for (int edgeB = edgeA + 1; edgeB < static_cast<int>(edges.size());
         ++edgeB) {
      const GeometryEdge& e1 = edges[static_cast<size_t>(edgeB)];
      if (!isDeformableOwner(e1.ref.owner)) {
        continue;
      }
      // Skip adjacent edges (share a vertex).
      if (e0.p0 == e1.p0 || e0.p0 == e1.p1 || e0.p1 == e1.p0 ||
          e0.p1 == e1.p1) {
        continue;
      }
      const glm::dvec3 c = geometry.worldPosition(e1.p0);
      const glm::dvec3 d = geometry.worldPosition(e1.p1);
      const double distanceSqr =
          contact_shared::shDistanceSqrEdgeEdge(a, b, c, d);
      const double reserved = e0.radius + e1.radius;
      const double distance = std::sqrt(distanceSqr);
      if (distance >= reserved) {
        continue;
      }
      raw.constraints.push_back(makeEdgeEdgeConstraint(
          e0.p0, e0.p1, a, b, e1.p0, e1.p1, c, d, distance, reserved));
      raw.penetratingVertices.insert(e0.p0);
      raw.penetratingVertices.insert(e0.p1);
      raw.penetratingVertices.insert(e1.p0);
      raw.penetratingVertices.insert(e1.p1);
    }
  }

  // --- Inter: deformable points vs collider triangles (PT) ---
  for (int pointIndex = 0; pointIndex < static_cast<int>(points.size());
       ++pointIndex) {
    const GeometryPoint& point = points[static_cast<size_t>(pointIndex)];
    if (!isDeformableOwner(point.owner)) {
      continue;
    }
    const PointIdx pidx = PointIdx{pointIndex};
    const glm::dvec3 p = geometry.worldPosition(pidx);
    for (int triIndex = 0; triIndex < static_cast<int>(triangles.size());
         ++triIndex) {
      const GeometryTriangle& tri = triangles[static_cast<size_t>(triIndex)];
      if (!isColliderOwner(tri.ref.owner)) {
        continue;
      }
      const glm::dvec3 a = geometry.worldPosition(tri.p0);
      const glm::dvec3 b = geometry.worldPosition(tri.p1);
      const glm::dvec3 c = geometry.worldPosition(tri.p2);
      const double distanceSqr = contact_shared::shDistanceSqrPointTriangle(p, a, b, c);
      const double reserved = point.radius + tri.thickness;
      const double distance = std::sqrt(distanceSqr);
      if (distance >= reserved) {
        continue;
      }
      raw.constraints.push_back(makePointTriangleConstraint(
          pidx, p, tri.p0, tri.p1, tri.p2, a, b, c, distance, reserved));
      raw.penetratingVertices.insert(pidx);
      raw.penetratingTriangles.insert(TriangleIdx{triIndex});
    }
  }

  // --- Inter: deformable edges vs collider edges (EE) ---
  for (const GeometryEdge& e0 : edges) {
    if (!isDeformableOwner(e0.ref.owner)) {
      continue;
    }
    const glm::dvec3 a = geometry.worldPosition(e0.p0);
    const glm::dvec3 b = geometry.worldPosition(e0.p1);
    for (const GeometryEdge& e1 : edges) {
      if (!isColliderOwner(e1.ref.owner)) {
        continue;
      }
      const glm::dvec3 c = geometry.worldPosition(e1.p0);
      const glm::dvec3 d = geometry.worldPosition(e1.p1);
      const double distanceSqr =
          contact_shared::shDistanceSqrEdgeEdge(a, b, c, d);
      const double reserved = e0.radius + e1.radius;
      const double distance = std::sqrt(distanceSqr);
      if (distance >= reserved) {
        continue;
      }
      raw.constraints.push_back(makeEdgeEdgeConstraint(
          e0.p0, e0.p1, a, b, e1.p0, e1.p1, c, d, distance, reserved));
      raw.penetratingVertices.insert(e0.p0);
      raw.penetratingVertices.insert(e0.p1);
      raw.penetratingVertices.insert(e1.p0);
      raw.penetratingVertices.insert(e1.p1);
    }
  }

  return raw;
}

std::vector<PenetrationDomain> identifyPenetrationDomains(
    const RawPenetrations& raw,
    std::span<const GeometryEdge> meshEdges,
    const std::unordered_set<PointIdx>& pinnedVertices,
    int totalVertexCount)
{
  KSK_PROFILE_SCOPE("PdCollision/Interpret");
  if (raw.empty() || totalVertexCount <= 0) {
    return {};
  }

  // Build the mesh graph G = (V, E) and the penetration boundary E_bound:
  // edges with exactly one endpoint in V_pen.
  std::vector<std::vector<int>> adjacency(
      static_cast<size_t>(totalVertexCount));
  std::set<std::pair<int, int>> e_bound;
  for (const GeometryEdge& edge : meshEdges) {
    const int u = edge.p0;
    const int v = edge.p1;
    if (u < 0 || v < 0 || u >= totalVertexCount || v >= totalVertexCount) {
      continue;
    }
    adjacency[static_cast<size_t>(u)].push_back(v);
    adjacency[static_cast<size_t>(v)].push_back(u);
    const bool uIn = raw.penetratingVertices.contains(PointIdx{u});
    const bool vIn = raw.penetratingVertices.contains(PointIdx{v});
    if (uIn != vIn) {
      e_bound.insert(edgeKey(u, v));
    }
  }

  // Connected components of G after removing E_bound.
  std::vector<int> component(static_cast<size_t>(totalVertexCount), -1);
  int componentCount = 0;
  for (int start = 0; start < totalVertexCount; ++start) {
    if (component[static_cast<size_t>(start)] != -1) {
      continue;
    }
    std::vector<int> stack{start};
    component[static_cast<size_t>(start)] = componentCount;
    while (!stack.empty()) {
      const int current = stack.back();
      stack.pop_back();
      for (const int neighbor : adjacency[static_cast<size_t>(current)]) {
        if (component[static_cast<size_t>(neighbor)] != -1) {
          continue;
        }
        if (e_bound.contains(edgeKey(current, neighbor))) {
          continue;  // cut edge
        }
        component[static_cast<size_t>(neighbor)] = componentCount;
        stack.push_back(neighbor);
      }
    }
    ++componentCount;
  }

  std::vector<std::vector<PointIdx>> members(
      static_cast<size_t>(componentCount));
  for (int v = 0; v < totalVertexCount; ++v) {
    if (component[static_cast<size_t>(v)] >= 0) {
      members[static_cast<size_t>(component[static_cast<size_t>(v)])]
          .push_back(PointIdx{v});
    }
  }

  std::vector<PenetrationDomain> domains;
  for (int ci = 0; ci < componentCount; ++ci) {
    std::vector<PointIdx>& componentVertices = members[static_cast<size_t>(ci)];
    if (componentVertices.empty()) {
      continue;
    }

    // Rule 1: pin exclusion.
    bool hasPin = false;
    for (const PointIdx v : componentVertices) {
      if (pinnedVertices.contains(v)) {
        hasPin = true;
        break;
      }
    }
    if (hasPin) {
      continue;
    }

    // Boundary vertices (incident to an E_bound edge) and penetrating flags.
    std::unordered_set<PointIdx> boundary;
    bool hasPenetration = false;
    bool touchesMeshBoundary = false;
    for (const PointIdx v : componentVertices) {
      if (raw.penetratingVertices.contains(v)) {
        hasPenetration = true;
      }
      bool onBoundary = false;
      for (const int neighbor : adjacency[static_cast<size_t>(v)]) {
        if (e_bound.contains(edgeKey(v, neighbor))) {
          onBoundary = true;
          break;
        }
      }
      if (onBoundary) {
        boundary.insert(v);
      }
      // Approximate open mesh boundary: a vertex with an incomplete one-ring
      // (degree < 3) is treated as a cloth boundary vertex. TODO: use the
      // per-mesh topology for exact boundary detection.
      if (adjacency[static_cast<size_t>(v)].size() < 3) {
        touchesMeshBoundary = true;
      }
    }

    // A component is "enclosed by E_bound" when it has external connections
    // through the penetration boundary and does not touch the open mesh
    // boundary.
    const bool enclosed = !boundary.empty() && !touchesMeshBoundary;

    PenetrationDomain domain;
    domain.allVertices = componentVertices;

    if (!enclosed && !touchesMeshBoundary) {
      // Isolated clean island: shallow resolution, push only the directly
      // penetrating vertices.
      domain.type = PenetrationDomain::Type::Shallow;
      for (const PointIdx v : componentVertices) {
        if (raw.penetratingVertices.contains(v)) {
          domain.boundaryDisplacements.emplace_back(
              v, pushDirectionForPoint(raw, v));
        }
      }
      if (!domain.boundaryDisplacements.empty()) {
        domains.push_back(std::move(domain));
      }
      continue;
    }

    bool isPenetrationDomain = false;
    if (enclosed && !touchesMeshBoundary) {
      isPenetrationDomain = true;
    } else if (touchesMeshBoundary) {
      // The smaller of the two sides separated by the penetration boundary is
      // treated as the domain to correct.
      const int complementSize =
          totalVertexCount - static_cast<int>(componentVertices.size());
      isPenetrationDomain =
          static_cast<int>(componentVertices.size()) <= complementSize;
    }
    if (!isPenetrationDomain) {
      continue;
    }

    domain.type = PenetrationDomain::Type::PenetrationDomain;
    for (const PointIdx v : componentVertices) {
      if (boundary.contains(v)) {
        domain.boundaryDisplacements.emplace_back(
            v, pushDirectionForPoint(raw, v));
      } else {
        domain.interiorVertices.push_back(v);
      }
    }
    domains.push_back(std::move(domain));
  }

  return domains;
}

void resolvePenetrationDomains(DofBuffer& q,
                               const std::vector<PenetrationDomain>& domains,
                               std::span<const GeometryEdge> meshEdges)
{
  KSK_PROFILE_SCOPE("PdCollision/Solve");
  if (!q.isCPU()) {
    // TODO: GPU position-correction path.
    return;
  }

  std::unordered_map<PointIdx, std::vector<PointIdx>> adjacency;
  for (const GeometryEdge& edge : meshEdges) {
    adjacency[edge.p0].push_back(edge.p1);
    adjacency[edge.p1].push_back(edge.p0);
  }

  for (const PenetrationDomain& domain : domains) {
    std::unordered_map<PointIdx, glm::dvec3> displacement;
    for (const auto& [v, d] : domain.boundaryDisplacements) {
      q.cpu()[3 * v + 0] += d.x;
      q.cpu()[3 * v + 1] += d.y;
      q.cpu()[3 * v + 2] += d.z;
      displacement[v] = d;
    }

    if (domain.type != PenetrationDomain::Type::PenetrationDomain ||
        domain.interiorVertices.empty()) {
      continue;
    }

    // BFS layer propagation from the boundary: each interior vertex receives
    // the average displacement of its already-processed neighbors.
    std::unordered_set<PointIdx> inDomain(domain.allVertices.begin(),
                                          domain.allVertices.end());
    std::unordered_set<PointIdx> visited;
    std::vector<PointIdx> frontier;
    for (const auto& [v, d] : displacement) {
      (void)d;
      visited.insert(v);
      frontier.push_back(v);
    }

    while (!frontier.empty()) {
      std::vector<PointIdx> nextFrontier;
      for (const PointIdx v : frontier) {
        const auto neighbors = adjacency.find(v);
        if (neighbors == adjacency.end()) {
          continue;
        }
        for (const PointIdx nb : neighbors->second) {
          if (!inDomain.contains(nb) || visited.contains(nb)) {
            continue;
          }
          glm::dvec3 accumulated{0.0};
          int count = 0;
          const auto nbNeighbors = adjacency.find(nb);
          if (nbNeighbors != adjacency.end()) {
            for (const PointIdx nn : nbNeighbors->second) {
              const auto found = displacement.find(nn);
              if (found != displacement.end()) {
                accumulated += found->second;
                ++count;
              }
            }
          }
          if (count > 0) {
            const glm::dvec3 d = accumulated / static_cast<double>(count);
            q.cpu()[3 * nb + 0] += d.x;
            q.cpu()[3 * nb + 1] += d.y;
            q.cpu()[3 * nb + 2] += d.z;
            displacement[nb] = d;
          }
          visited.insert(nb);
          nextFrontier.push_back(nb);
        }
      }
      frontier = std::move(nextFrontier);
    }
  }
}

void resolvePdCollisions(SimulationContext& simulation,
                         const std::unordered_set<PointIdx>& pinnedVertices,
                         int maxResolveIterations)
{
  KSK_PROFILE_SCOPE("PdCollision/Resolve");
  const GlobalGeometryManager& geometry = simulation.scene().geometry;
  for (int iteration = 0; iteration < maxResolveIterations; ++iteration) {
    RawPenetrations raw = detectPenetrations(geometry);
    if (raw.empty()) {
      break;
    }
    std::vector<PenetrationDomain> domains = identifyPenetrationDomains(
        raw, geometry.edges, pinnedVertices, geometry.pointCount());
    if (domains.empty()) {
      break;
    }
    resolvePenetrationDomains(simulation.q(), domains, geometry.edges);

    // Resync geometry from the corrected state.
    for (const auto& subsystem : simulation.subsystems()) {
      const DofRange range = subsystem->dofRange();
      subsystem->readState(simulation.q().slice(range),
                           simulation.qdot().slice(range));
      subsystem->updateGeometry(simulation.scene().geometry);
    }
  }
}

}  // namespace ksk::runtime
