#include <Runtime/contact-detection.h>
#include <Core/dynamic-bitset.h>
#include <Spatify/lbvh.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/geometric.hpp>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../contact-detection-impl.h"

namespace ksk::runtime {
namespace {
constexpr int kMaxACCDIterations = 2000;
constexpr Real kACCDGapFraction = 0.1;
constexpr Real kBVHMinimumExtent = 1.0e-12;

enum class PointTriangleDistanceType {
  P_A,
  P_B,
  P_C,
  P_AB,
  P_BC,
  P_CA,
  P_ABC,
};

enum class EdgeEdgeDistanceType {
  A_C,
  A_D,
  B_C,
  B_D,
  AB_C,
  AB_D,
  A_CD,
  B_CD,
  AB_CD,
};

struct PrimitiveRef {
  int id = -1;
  Real reserved_dist = 0.0;
  spatify::BBox<Real, 3> bounds;
};

struct PrimitiveAccessor {
  using CoordType = Real;

  const std::vector<PrimitiveRef>* primitives = nullptr;

  [[nodiscard]] spatify::BBox<Real, 3> bbox(int index) const {
    return primitives->at(static_cast<size_t>(index)).bounds;
  }

  [[nodiscard]] int size() const {
    return static_cast<int>(primitives->size());
  }
};

struct BodyPairPrimitiveRef {
  int pair = -1;
  int primitive = -1;
  std::uint8_t side = 0;
};

struct RigidInvolvedPairRefs {
  std::vector<BodyPairPrimitiveRef> points;
  std::vector<BodyPairPrimitiveRef> edges;
};

spatify::BBox<Real, 3> ensureFiniteBVHBounds(spatify::BBox<Real, 3> bounds) {
  for (int axis = 0; axis < 3; ++axis) {
    const Real extent = bounds.hi[axis] - bounds.lo[axis];
    if (extent >= kBVHMinimumExtent) {
      continue;
    }
    const Real centre = (bounds.lo[axis] + bounds.hi[axis]) * 0.5;
    bounds.lo[axis] = centre - kBVHMinimumExtent;
    bounds.hi[axis] = centre + kBVHMinimumExtent;
  }
  return bounds;
}

void makeBoundsSafeForLBVH(std::vector<PrimitiveRef>& primitives) {
  for (PrimitiveRef& primitive : primitives) {
    primitive.bounds = ensureFiniteBVHBounds(primitive.bounds);
  }
}

[[nodiscard]] spatify::LBVH<Real> buildLBVH(
    std::vector<PrimitiveRef>& primitives) {
  makeBoundsSafeForLBVH(primitives);
  spatify::LBVH<Real> tree;
  if (!primitives.empty())
    tree.update(PrimitiveAccessor{.primitives = &primitives});
  return tree;
}

void sortUniquePairs(std::vector<std::array<int, 2>>& pairs) {
  std::ranges::sort(pairs);
  pairs.erase(std::ranges::unique(pairs).begin(), pairs.end());
}

void finalizeCollisionWorkList(ContactWorkList& work) {
  sortUniquePairs(work.pointTriangles);
  sortUniquePairs(work.edgeEdges);
}

[[nodiscard]] bool isFiniteBounds(const GeometryBounds& bounds) {
  return bounds.lo.x <= bounds.hi.x && bounds.lo.y <= bounds.hi.y &&
         bounds.lo.z <= bounds.hi.z;
}

[[nodiscard]] GeometryBounds intersectBounds(const GeometryBounds& first,
                                             const GeometryBounds& second) {
  GeometryBounds result;
  result.lo = glm::max(first.lo, second.lo);
  result.hi = glm::min(first.hi, second.hi);
  return result;
}

bool geometryPairCollisionEnabled(const GlobalGeometryManager& geometry,
                                  PointIdx first, PointIdx second) {
  const GeometryBodyIdx first_body = geometry.pointOwner(first).body;
  const GeometryBodyIdx second_body = geometry.pointOwner(second).body;
  return geometry.collisionPairEnabled(first_body, second_body);
}

bool isDeformableBody(const GlobalGeometryManager& geometry,
                      GeometryBodyIdx body) {
  return geometry.geometryBody(body).type & Deformable;
}

bool isRigidBody(const GlobalGeometryManager& geometry, GeometryBodyIdx body) {
  return geometry.geometryBody(body).type & Rigid;
}

GeometryBodyIdx pointBody(const GlobalGeometryManager& geometry,
                          int point) {
  return geometry.pointOwner(point).body;
}

GeometryBodyIdx edgeBody(const GlobalGeometryManager& geometry, int edge) {
  return geometry.edgeOwner(edge).body;
}

GeometryBodyIdx triangleBody(const GlobalGeometryManager& geometry,
                             int triangle) {
  return geometry.triangleOwner(triangle).body;
}

bool pointTriangleCollisionEnabled(const GlobalGeometryManager& geometry,
                                   PointIdx point, int triangle) {
  const auto tri = geometry.globalTriangle(triangle);
  return geometryPairCollisionEnabled(geometry, point, tri[0]);
}

bool edgeEdgeCollisionEnabled(const GlobalGeometryManager& geometry,
                              int first_edge, int second_edge) {
  const auto first = geometry.globalEdge(first_edge);
  const auto second = geometry.globalEdge(second_edge);
  return geometryPairCollisionEnabled(geometry, first[0], second[0]);
}

[[nodiscard]] std::vector<size_t> exclusiveOffsets(
    const std::vector<size_t>& counts) {
  std::vector<size_t> offsets(counts.size() + 1, size_t{0});
  if (counts.empty()) {
    return offsets;
  }
  std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), size_t{0});
  offsets.back() = offsets[counts.size() - 1] + counts.back();
  return offsets;
}

ContactStencil makeStencil(ContactCase type, std::array<int, 4> ids,
                           const ContactDetectionConfig& config,
                           Real reserved_dist = 0.0) {
  return ContactStencil{
      .type = type,
      .geometryIds = ids,
      .dHat = config.dHat,
      .stiffness = config.stiffness,
      .thickness = reserved_dist,
  };
}

ContactCandidate makeCandidate(EContactCandidate kind, std::array<int, 4> ids,
                               const ContactDetectionConfig& config,
                               Real reserved_distance = 0.0) {
  return ContactCandidate{
      .type = kind,
      .geometryIds = ids,
      .dHat = config.dHat,
      .stiffness = config.stiffness,
      .reservedDistance = reserved_distance,
  };
}

Real pointRadius(const GlobalGeometryManager& geometry, PointIdx point) {
  return geometry.points.at(static_cast<size_t>(point)).radius;
}

Real triangleSurfaceOffset(const GlobalGeometryManager& geometry,
                           const std::array<PointIdx, 3>& triangle) {
  Real offset = 0.0;
  for (const GeometryTriangle& candidate : geometry.triangles) {
    const std::array points{candidate.p0, candidate.p1, candidate.p2};
    const bool same = (points[0] == triangle[0] || points[0] == triangle[1] ||
                       points[0] == triangle[2]) &&
                      (points[1] == triangle[0] || points[1] == triangle[1] ||
                       points[1] == triangle[2]) &&
                      (points[2] == triangle[0] || points[2] == triangle[1] ||
                       points[2] == triangle[2]);
    if (same) {
      offset = std::max(offset, candidate.thickness);
    }
  }
  return offset;
}

Real edgeSurfaceOffset(const GlobalGeometryManager& geometry, PointIdx p0,
                       PointIdx p1, Real fallback = 0.0) {
  Real offset = std::max(
      {pointRadius(geometry, p0), pointRadius(geometry, p1), fallback});
  for (const GeometryEdge& edge : geometry.edges) {
    if ((edge.p0 == p0 && edge.p1 == p1) || (edge.p0 == p1 && edge.p1 == p0)) {
      offset = std::max(offset, edge.radius);
    }
  }
  return offset;
}

Real pointEdgeSurfaceOffset(const GlobalGeometryManager& geometry,
                            PointIdx point, PointIdx edge0, PointIdx edge1,
                            Real fallback = 0.0) {
  return pointRadius(geometry, point) +
         edgeSurfaceOffset(geometry, edge0, edge1, fallback);
}

Real distanceSqrPointPoint(const glm::dvec3& a, const glm::dvec3& b) {
  const glm::dvec3 d = a - b;
  return glm::dot(d, d);
}

Real distanceSqrPointLine(const glm::dvec3& p, const glm::dvec3& a,
                          const glm::dvec3& b) {
  const glm::dvec3 e = b - a;
  const Real e_len2 = glm::dot(e, e);
  if (e_len2 == 0.0) {
    return distanceSqrPointPoint(p, a);
  }
  const glm::dvec3 c = glm::cross(e, p - a);
  return glm::dot(c, c) / e_len2;
}

Real distanceSqrPointPlane(const glm::dvec3& p, const glm::dvec3& a,
                           const glm::dvec3& b, const glm::dvec3& c) {
  const glm::dvec3 normal = glm::cross(b - a, c - a);
  const Real n_len2 = glm::dot(normal, normal);
  if (n_len2 == 0.0) {
    return std::min({distanceSqrPointLine(p, a, b),
                     distanceSqrPointLine(p, b, c),
                     distanceSqrPointLine(p, c, a)});
  }
  const Real d = glm::dot(p - a, normal);
  return d * d / n_len2;
}

Real distanceSqrLineLine(const glm::dvec3& a0, const glm::dvec3& a1,
                         const glm::dvec3& b0, const glm::dvec3& b1) {
  const glm::dvec3 normal = glm::cross(a1 - a0, b1 - b0);
  const Real n_len2 = glm::dot(normal, normal);
  if (n_len2 == 0.0) {
    return distanceSqrPointLine(a0, b0, b1);
  }
  const Real d = glm::dot(a0 - b0, normal);
  return d * d / n_len2;
}

void pointTriangleEdgeTest(const glm::dvec3& from, const glm::dvec3& to,
                           const glm::dvec3& point, const glm::dvec3& normal,
                           Real& s, Real& t) {
  const glm::dvec3 e = to - from;
  const glm::dvec3 n = glm::cross(e, normal);
  const Real a00 = glm::dot(e, e);
  const Real a01 = glm::dot(e, n);
  const Real a11 = glm::dot(n, n);
  const Real b0 = glm::dot(e, point - from);
  const Real b1 = glm::dot(n, point - from);
  const Real det = a00 * a11 - a01 * a01;
  if (det == 0.0) {
    s = 0.0;
    t = -1.0;
    return;
  }
  s = (a11 * b0 - a01 * b1) / det;
  t = (a00 * b1 - a01 * b0) / det;
}

PointTriangleDistanceType decidePointTriangleDistanceType(const glm::dvec3& p,
                                                          const glm::dvec3& a,
                                                          const glm::dvec3& b,
                                                          const glm::dvec3& c) {
  const glm::dvec3 normal = glm::cross(b - a, c - a);
  if (glm::dot(normal, normal) == 0.0) {
    const Real da = distanceSqrPointLine(p, a, b);
    const Real db = distanceSqrPointLine(p, b, c);
    const Real dc = distanceSqrPointLine(p, c, a);
    if (da <= db && da <= dc) {
      return PointTriangleDistanceType::P_AB;
    }
    if (db <= dc) {
      return PointTriangleDistanceType::P_BC;
    }
    return PointTriangleDistanceType::P_CA;
  }

  Real s0 = 0.0;
  Real t0 = 0.0;
  pointTriangleEdgeTest(a, b, p, normal, s0, t0);
  if (s0 > 0.0 && s0 < 1.0 && t0 >= 0.0) {
    return PointTriangleDistanceType::P_AB;
  }

  Real s1 = 0.0;
  Real t1 = 0.0;
  pointTriangleEdgeTest(b, c, p, normal, s1, t1);
  if (s1 > 0.0 && s1 < 1.0 && t1 >= 0.0) {
    return PointTriangleDistanceType::P_BC;
  }

  Real s2 = 0.0;
  Real t2 = 0.0;
  pointTriangleEdgeTest(c, a, p, normal, s2, t2);
  if (s2 > 0.0 && s2 < 1.0 && t2 >= 0.0) {
    return PointTriangleDistanceType::P_CA;
  }

  if (s0 <= 0.0 && s2 >= 1.0) {
    return PointTriangleDistanceType::P_A;
  }
  if (s1 <= 0.0 && s0 >= 1.0) {
    return PointTriangleDistanceType::P_B;
  }
  if (s2 <= 0.0 && s1 >= 1.0) {
    return PointTriangleDistanceType::P_C;
  }
  return PointTriangleDistanceType::P_ABC;
}

EdgeEdgeDistanceType decideEdgeEdgeParallelDistanceType(const glm::dvec3& ea0,
                                                        const glm::dvec3& ea1,
                                                        const glm::dvec3& eb0,
                                                        const glm::dvec3& eb1) {
  const glm::dvec3 ea = ea1 - ea0;
  const Real ea_sqr = glm::dot(ea, ea);
  if (ea_sqr == 0.0) {
    return EdgeEdgeDistanceType::A_C;
  }
  const Real alpha = glm::dot(eb0 - ea0, ea) / ea_sqr;
  const Real beta = glm::dot(eb1 - ea0, ea) / ea_sqr;

  int eac = 0;
  int ebc = 0;
  if (alpha < 0.0) {
    eac = (0.0 <= beta && beta <= 1.0) ? 2 : 0;
    ebc = (beta <= alpha) ? 0 : (beta <= 1.0 ? 1 : 2);
  } else if (alpha > 1.0) {
    eac = (0.0 <= beta && beta <= 1.0) ? 2 : 1;
    ebc = (beta >= alpha) ? 0 : (0.0 <= beta ? 1 : 2);
  } else {
    eac = 2;
    ebc = 0;
  }

  if (ebc < 2) {
    const int code = (eac << 1) | ebc;
    switch (code) {
      case 0:
        return EdgeEdgeDistanceType::A_C;
      case 1:
        return EdgeEdgeDistanceType::A_D;
      case 2:
        return EdgeEdgeDistanceType::B_C;
      case 3:
        return EdgeEdgeDistanceType::B_D;
      case 4:
        return EdgeEdgeDistanceType::AB_C;
      default:
        return EdgeEdgeDistanceType::AB_D;
    }
  }

  return eac == 0 ? EdgeEdgeDistanceType::A_CD : EdgeEdgeDistanceType::B_CD;
}

EdgeEdgeDistanceType decideEdgeEdgeDistanceType(const glm::dvec3& ea0,
                                                const glm::dvec3& ea1,
                                                const glm::dvec3& eb0,
                                                const glm::dvec3& eb1) {
  constexpr Real parallel_threshold = 1.0e-20;
  const glm::dvec3 u = ea1 - ea0;
  const glm::dvec3 v = eb1 - eb0;
  const glm::dvec3 w = ea0 - eb0;

  const Real a = glm::dot(u, u);
  const Real b = glm::dot(u, v);
  const Real c = glm::dot(v, v);
  const Real d = glm::dot(u, w);
  const Real e = glm::dot(v, w);
  const Real det = a * c - b * b;

  if (a == 0.0 && c == 0.0) {
    return EdgeEdgeDistanceType::A_C;
  }
  if (a == 0.0) {
    return EdgeEdgeDistanceType::A_CD;
  }
  if (c == 0.0) {
    return EdgeEdgeDistanceType::AB_C;
  }

  const Real parallel_tolerance = parallel_threshold * std::max(1.0, a * c);
  const glm::dvec3 uxv = glm::cross(u, v);
  if (glm::dot(uxv, uxv) < parallel_tolerance) {
    return decideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);
  }

  EdgeEdgeDistanceType default_case = EdgeEdgeDistanceType::AB_CD;
  const Real sN = b * e - c * d;
  Real tN = 0.0;
  Real tD = 0.0;
  if (sN <= 0.0) {
    tN = e;
    tD = c;
    default_case = EdgeEdgeDistanceType::A_CD;
  } else if (sN >= det) {
    tN = e + b;
    tD = c;
    default_case = EdgeEdgeDistanceType::B_CD;
  } else {
    tN = a * e - b * d;
    tD = det;
  }

  if (tN <= 0.0) {
    if (-d <= 0.0) {
      return EdgeEdgeDistanceType::A_C;
    }
    return -d >= a ? EdgeEdgeDistanceType::B_C : EdgeEdgeDistanceType::AB_C;
  }
  if (tN >= tD) {
    if ((-d + b) <= 0.0) {
      return EdgeEdgeDistanceType::A_D;
    }
    return (-d + b) >= a ? EdgeEdgeDistanceType::B_D
                         : EdgeEdgeDistanceType::AB_D;
  }
  return default_case;
}

Real pointTriangleDistanceSqrByType(PointTriangleDistanceType type,
                                    const glm::dvec3& p, const glm::dvec3& a,
                                    const glm::dvec3& b, const glm::dvec3& c) {
  switch (type) {
    case PointTriangleDistanceType::P_A:
      return distanceSqrPointPoint(p, a);
    case PointTriangleDistanceType::P_B:
      return distanceSqrPointPoint(p, b);
    case PointTriangleDistanceType::P_C:
      return distanceSqrPointPoint(p, c);
    case PointTriangleDistanceType::P_AB:
      return distanceSqrPointLine(p, a, b);
    case PointTriangleDistanceType::P_BC:
      return distanceSqrPointLine(p, b, c);
    case PointTriangleDistanceType::P_CA:
      return distanceSqrPointLine(p, c, a);
    case PointTriangleDistanceType::P_ABC:
      return distanceSqrPointPlane(p, a, b, c);
  }
  return 0.0;
}

Real edgeEdgeDistanceSqrByType(EdgeEdgeDistanceType type, const glm::dvec3& a0,
                               const glm::dvec3& a1, const glm::dvec3& b0,
                               const glm::dvec3& b1) {
  switch (type) {
    case EdgeEdgeDistanceType::A_C:
      return distanceSqrPointPoint(a0, b0);
    case EdgeEdgeDistanceType::A_D:
      return distanceSqrPointPoint(a0, b1);
    case EdgeEdgeDistanceType::B_C:
      return distanceSqrPointPoint(a1, b0);
    case EdgeEdgeDistanceType::B_D:
      return distanceSqrPointPoint(a1, b1);
    case EdgeEdgeDistanceType::AB_C:
      return distanceSqrPointLine(b0, a0, a1);
    case EdgeEdgeDistanceType::AB_D:
      return distanceSqrPointLine(b1, a0, a1);
    case EdgeEdgeDistanceType::A_CD:
      return distanceSqrPointLine(a0, b0, b1);
    case EdgeEdgeDistanceType::B_CD:
      return distanceSqrPointLine(a1, b0, b1);
    case EdgeEdgeDistanceType::AB_CD:
      return distanceSqrLineLine(a0, a1, b0, b1);
  }
  return 0.0;
}

Real pointEdgeProjectionParameter(const glm::dvec3& p, const glm::dvec3& a,
                                  const glm::dvec3& b) {
  const glm::dvec3 e = b - a;
  const Real e_len2 = glm::dot(e, e);
  if (e_len2 == 0.0) {
    return 0.0;
  }
  return glm::dot(p - a, e) / e_len2;
}

std::optional<Real> runACCD(bool edgeEdge, std::array<glm::dvec3, 4> x,
                            std::array<glm::dvec3, 4> u, Real toi,
                            Real reserved_distance) {
  const glm::dvec3 mean_step = (u[0] + u[1] + u[2] + u[3]) * 0.25;
  for (glm::dvec3& step : u) {
    step -= mean_step;
  }

  const Real lp = edgeEdge ? std::max(glm::length(u[0]), glm::length(u[1])) +
                                 std::max(glm::length(u[2]), glm::length(u[3]))
                           : glm::length(u[0]) +
                                 std::max({glm::length(u[1]), glm::length(u[2]),
                                           glm::length(u[3])});
  if (lp == 0.0) {
    return std::nullopt;
  }

  auto distance = [&]() -> Real {
    if (edgeEdge) {
      const EdgeEdgeDistanceType type =
          decideEdgeEdgeDistanceType(x[0], x[1], x[2], x[3]);
      return std::sqrt(edgeEdgeDistanceSqrByType(type, x[0], x[1], x[2], x[3]));
    }
    const PointTriangleDistanceType type =
        decidePointTriangleDistanceType(x[0], x[1], x[2], x[3]);
    return std::sqrt(
        pointTriangleDistanceSqrByType(type, x[0], x[1], x[2], x[3]));
  };

  Real dis = distance();
  if (dis <= reserved_distance) {
    return 0.0;
  }
  const Real initial_gap = dis - reserved_distance;
  const Real gap = kACCDGapFraction * initial_gap;
  Real t = 0.0;
  Real dt = (1.0 - kACCDGapFraction) * (initial_gap / lp);

  for (int iteration = 0; iteration < kMaxACCDIterations; ++iteration) {
    for (int i = 0; i < 4; ++i) x[i] += u[i] * dt;

    dis = distance();

    if (dis - reserved_distance < gap + 1.0e-10) return t == 0.0 ? dt : t;

    t += dt;

    if (t > toi) return std::nullopt;

    dt = 0.9 * (dis - reserved_distance) / lp;
  }
  return t > 0.0 ? std::optional(t) : std::nullopt;
}

bool shouldKeepByDistanceOrCCD(Real distance_sqr, std::optional<Real> ccd_toi,
                               const ContactDetectionConfig& config,
                               Real primitive_thickness) {
  const Real threshold = config.dHat + primitive_thickness;
  return distance_sqr <= threshold * threshold ||
         (ccd_toi && *ccd_toi <= config.toi);
}

ContactStencil createPointPointStencil(PointIdx first, PointIdx second,
                                       const GlobalGeometryManager& geometry,
                                       const ContactDetectionConfig& config) {
  return makeStencil(
      ContactCase::PP, {first, second, -1, -1}, config,
      pointRadius(geometry, first) + pointRadius(geometry, second));
}

ContactStencil createPointEdgeStencil(PointIdx point, PointIdx edge0,
                                      PointIdx edge1,
                                      const GlobalGeometryManager& geometry,
                                      const ContactDetectionConfig& config,
                                      Real fallback_edge_offset = 0.0) {
  return makeStencil(ContactCase::PE, {point, edge0, edge1, -1}, config,
                     pointEdgeSurfaceOffset(geometry, point, edge0, edge1,
                                            fallback_edge_offset));
}

std::optional<ContactStencil> createPointTriangleStencil(
    PointIdx point, const std::array<PointIdx, 3>& triangle,
    const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction, const ContactDetectionConfig& config,
    Real primitive_thickness) {
  const glm::dvec3 p = geometry.worldPosition(point);
  const glm::dvec3 a = geometry.worldPosition(triangle[0]);
  const glm::dvec3 b = geometry.worldPosition(triangle[1]);
  const glm::dvec3 c = geometry.worldPosition(triangle[2]);
  const Real point_offset = pointRadius(geometry, point);
  const Real triangle_offset =
      std::max(primitive_thickness, triangleSurfaceOffset(geometry, triangle));
  const PointTriangleDistanceType type =
      decidePointTriangleDistanceType(p, a, b, c);
  const Real distance_sqr = pointTriangleDistanceSqrByType(type, p, a, b, c);
  const std::optional<Real> toi =
      runACCD(false, {p, a, b, c},
              {direction[point], direction[triangle[0]], direction[triangle[1]],
               direction[triangle[2]]},
              config.toi, point_offset + triangle_offset);
  if (!shouldKeepByDistanceOrCCD(distance_sqr, toi, config,
                                 point_offset + triangle_offset)) {
    return std::nullopt;
  }

  switch (type) {
    case PointTriangleDistanceType::P_A:
      return createPointPointStencil(point, triangle[0], geometry, config);
    case PointTriangleDistanceType::P_B:
      return createPointPointStencil(point, triangle[1], geometry, config);
    case PointTriangleDistanceType::P_C:
      return createPointPointStencil(point, triangle[2], geometry, config);
    case PointTriangleDistanceType::P_AB:
      return createPointEdgeStencil(point, triangle[0], triangle[1], geometry,
                                    config, triangle_offset);
    case PointTriangleDistanceType::P_BC:
      return createPointEdgeStencil(point, triangle[1], triangle[2], geometry,
                                    config, triangle_offset);
    case PointTriangleDistanceType::P_CA:
      return createPointEdgeStencil(point, triangle[2], triangle[0], geometry,
                                    config, triangle_offset);
    case PointTriangleDistanceType::P_ABC:
      return makeStencil(ContactCase::PT,
                         {point, triangle[0], triangle[1], triangle[2]}, config,
                         point_offset + triangle_offset);
  }
  return std::nullopt;
}

ContactStencil createEdgeEdgeFeatureStencil(
    EdgeEdgeDistanceType type, const std::array<PointIdx, 2>& first,
    const std::array<PointIdx, 2>& second,
    const GlobalGeometryManager& geometry, const ContactDetectionConfig& config,
    Real first_offset, Real second_offset) {
  switch (type) {
    case EdgeEdgeDistanceType::A_C:
      return createPointPointStencil(first[0], second[0], geometry, config);
    case EdgeEdgeDistanceType::A_D:
      return createPointPointStencil(first[0], second[1], geometry, config);
    case EdgeEdgeDistanceType::B_C:
      return createPointPointStencil(first[1], second[0], geometry, config);
    case EdgeEdgeDistanceType::B_D:
      return createPointPointStencil(first[1], second[1], geometry, config);
    case EdgeEdgeDistanceType::AB_C:
      return createPointEdgeStencil(second[0], first[0], first[1], geometry,
                                    config, first_offset);
    case EdgeEdgeDistanceType::AB_D:
      return createPointEdgeStencil(second[1], first[0], first[1], geometry,
                                    config, first_offset);
    case EdgeEdgeDistanceType::A_CD:
      return createPointEdgeStencil(first[0], second[0], second[1], geometry,
                                    config, second_offset);
    case EdgeEdgeDistanceType::B_CD:
      return createPointEdgeStencil(first[1], second[0], second[1], geometry,
                                    config, second_offset);
    case EdgeEdgeDistanceType::AB_CD:
      return makeStencil(ContactCase::EE,
                         {first[0], first[1], second[0], second[1]}, config,
                         first_offset + second_offset);
  }
  return makeStencil(ContactCase::EE, {-1, -1, -1, -1}, config);
}

std::optional<ContactStencil> createEdgeEdgeStencil(
    const std::array<PointIdx, 2>& first, const std::array<PointIdx, 2>& second,
    const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction, const ContactDetectionConfig& config,
    Real primitive_thickness) {
  const glm::dvec3 a0 = geometry.worldPosition(first[0]);
  const glm::dvec3 a1 = geometry.worldPosition(first[1]);
  const glm::dvec3 b0 = geometry.worldPosition(second[0]);
  const glm::dvec3 b1 = geometry.worldPosition(second[1]);
  const Real first_offset =
      edgeSurfaceOffset(geometry, first[0], first[1], primitive_thickness);
  const Real second_offset =
      edgeSurfaceOffset(geometry, second[0], second[1], primitive_thickness);
  const Real reserved_dist = first_offset + second_offset;
  const EdgeEdgeDistanceType type = decideEdgeEdgeDistanceType(a0, a1, b0, b1);
  const Real distance_sqr = edgeEdgeDistanceSqrByType(type, a0, a1, b0, b1);
  const std::optional<Real> toi =
      runACCD(true, {a0, a1, b0, b1},
              {direction[first[0]], direction[first[1]], direction[second[0]],
               direction[second[1]]},
              config.toi, reserved_dist);
  if (!shouldKeepByDistanceOrCCD(distance_sqr, toi, config, reserved_dist)) {
    return std::nullopt;
  }

  return createEdgeEdgeFeatureStencil(type, first, second, geometry, config,
                                      first_offset, second_offset);
}

#ifndef NDEBUG
void validateCPUCCDInput(const GlobalGeometryManager& geometry,
                         const GeometryBuffer& direction) {
  if (!direction.isCPU()) {
    throw std::runtime_error(
        "CPU contact detection requires a CPU geometry direction buffer");
  }
  if (direction.cpu().size() < static_cast<size_t>(geometry.pointCount())) {
    throw std::invalid_argument(
        "geometry direction buffer is smaller than geometry");
  }
}
#endif

[[nodiscard]] std::vector<PrimitiveRef> buildPointPrimitives(
    const GlobalGeometryManager& geometry, const GeometryBuffer& direction,
    Real toi) {
  core::dynamic_bitset surface_points(geometry.pointCount());
  const auto mark_surface_point = [&](PointIdx point) {
    assert(geometry.contains(point));
    surface_points.set(point);
  };
  for (int edge_id = 0; edge_id < geometry.edgeCount(); edge_id++) {
    const GeometryEdge& edge = geometry.edges[edge_id];
    mark_surface_point(edge.p0);
    mark_surface_point(edge.p1);
  }
  for (int triangle_id = 0; triangle_id < geometry.triangleCount();
       triangle_id++) {
    const GeometryTriangle& triangle = geometry.triangles[triangle_id];
    mark_surface_point(triangle.p0);
    mark_surface_point(triangle.p1);
    mark_surface_point(triangle.p2);
  }

  std::vector<PrimitiveRef> primitives;
  primitives.reserve(geometry.pointCount());
  for (int point = 0; point < geometry.pointCount(); point++) {
    const GeometryBodyIdx body = pointBody(geometry, point);
    const bool point_only_body =
        geometry.bodyEdgeRange(body).count == 0 &&
        geometry.bodyTriangleRange(body).count == 0;
    if (!surface_points.test(point) && !point_only_body) {
      continue;
    }
    const Real radius = geometry.points[point].radius;
    primitives.push_back(PrimitiveRef{
        .id = point,
        .reserved_dist = radius,
        .bounds = geometry.trajectoryPointBounds(point, direction.cpu(), toi)
                      .dilate(radius),
    });
  }
  return primitives;
}

[[nodiscard]] std::vector<PrimitiveRef> buildEdgePrimitives(
    const GlobalGeometryManager& geometry, const GeometryBuffer& direction,
    Real toi) {
  std::vector<PrimitiveRef> primitives;
  primitives.reserve(geometry.edgeCount());
  for (int edge = 0; edge < geometry.edgeCount(); ++edge) {
    const Real radius = geometry.edges[edge].radius;
    primitives.push_back(PrimitiveRef{
        .id = edge,
        .reserved_dist = radius,
        .bounds =
            geometry.trajectoryEdgeBounds(edge, direction.cpu(), toi, radius)
                .dilate(radius),
    });
  }
  return primitives;
}

[[nodiscard]] std::vector<PrimitiveRef> buildTrianglePrimitives(
    const GlobalGeometryManager& geometry, const GeometryBuffer& direction,
    Real toi) {
  std::vector<PrimitiveRef> primitives;
  primitives.reserve(static_cast<size_t>(geometry.triangleCount()));
  for (int triangle = 0; triangle < geometry.triangleCount(); ++triangle) {
    const Real thickness = geometry.triangles[triangle].thickness;
    primitives.push_back(PrimitiveRef{
        .id = triangle,
        .reserved_dist = thickness,
        .bounds = geometry
                      .trajectoryTriangleBounds(triangle, direction.cpu(), toi,
                                                thickness)
                      .dilate(thickness),
    });
  }
  return primitives;
}

[[nodiscard]] std::vector<PrimitiveRef> collectDeformablePointPrimitives(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& points) {
  std::vector<PrimitiveRef> result;
  result.reserve(points.size());
  for (const PrimitiveRef& point : points) {
    if (isDeformableBody(geometry, pointBody(geometry, point.id))) {
      result.push_back(point);
    }
  }
  return result;
}

[[nodiscard]] std::vector<PrimitiveRef> collectDeformableEdgePrimitives(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& edges) {
  std::vector<PrimitiveRef> result;
  result.reserve(edges.size());
  for (const PrimitiveRef& edge : edges) {
    if (isDeformableBody(geometry, edgeBody(geometry, edge.id))) {
      result.push_back(edge);
    }
  }
  return result;
}

[[nodiscard]] std::vector<PrimitiveRef> collectDeformableTrianglePrimitives(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& triangles) {
  std::vector<PrimitiveRef> result;
  result.reserve(triangles.size());
  for (const PrimitiveRef& triangle : triangles) {
    if (isDeformableBody(geometry, triangleBody(geometry, triangle.id))) {
      result.push_back(triangle);
    }
  }
  return result;
}

[[nodiscard]] std::vector<PrimitiveRef> collectRigidEdgePrimitives(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& edges) {
  std::vector<PrimitiveRef> result;
  result.reserve(edges.size());
  for (const PrimitiveRef& edge : edges) {
    if (isRigidBody(geometry, edgeBody(geometry, edge.id))) {
      result.push_back(edge);
    }
  }
  return result;
}

[[nodiscard]] std::vector<PrimitiveRef> collectRigidTrianglePrimitives(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& triangles) {
  std::vector<PrimitiveRef> result;
  result.reserve(triangles.size());
  for (const PrimitiveRef& triangle : triangles) {
    if (isRigidBody(geometry, triangleBody(geometry, triangle.id))) {
      result.push_back(triangle);
    }
  }
  return result;
}

GeometryBodyIdx oppositeBody(const ContactWorkList::BodyPairWork& pair,
                             std::uint8_t side) {
  return side == 0 ? pair.second : pair.first;
}

int bodyPairSideForPrimitiveBody(const ContactWorkList::BodyPairWork& pair,
                                 GeometryBodyIdx body) {
  if (body == pair.first) {
    return 0;
  }
  if (body == pair.second) {
    return 1;
  }
  return -1;
}

[[nodiscard]] std::vector<PrimitiveRef> buildBodyPairPrimitives(
    const ContactWorkList& work) {
  std::vector<PrimitiveRef> pairs;
  pairs.reserve(work.bodyPairs.size());
  for (int pair_index = 0; pair_index < static_cast<int>(work.bodyPairs.size());
       ++pair_index) {
    pairs.push_back(PrimitiveRef{
        .id = pair_index,
        .bounds = work.bodyPairs.at(static_cast<size_t>(pair_index)).iAABB,
    });
  }
  return pairs;
}

void gatherPointRefsThroughBodyPairBVH(
    const GlobalGeometryManager& geometry, const std::vector<PrimitiveRef>& points,
    const std::vector<PrimitiveRef>& pair_primitives,
    const spatify::LBVH<Real>& pair_tree, const ContactWorkList& work,
    std::vector<BodyPairPrimitiveRef>& refs) {
  std::vector<size_t> counts(points.size(), size_t{0});

  tbb::parallel_for(size_t{0}, points.size(), [&](size_t index) {
    const PrimitiveRef& point = points[index];
    const GeometryBodyIdx owner = pointBody(geometry, point.id);
    const GeometryBounds query_bounds = ensureFiniteBVHBounds(point.bounds);
    pair_tree.runSpatialQuery(
        [&](int primitive_index) {
          const int pair_index =
              pair_primitives.at(static_cast<size_t>(primitive_index)).id;
          const ContactWorkList::BodyPairWork& pair =
              work.bodyPairs.at(static_cast<size_t>(pair_index));
          if (bodyPairSideForPrimitiveBody(pair, owner) < 0) {
            return false;
          }
          ++counts[index];
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });

  const std::vector<size_t> offsets = exclusiveOffsets(counts);
  refs.resize(offsets.back());

  tbb::parallel_for(size_t{0}, points.size(), [&](size_t index) {
    const PrimitiveRef& point = points[index];
    const GeometryBodyIdx owner = pointBody(geometry, point.id);
    const GeometryBounds query_bounds = ensureFiniteBVHBounds(point.bounds);
    size_t write = offsets[index];
    pair_tree.runSpatialQuery(
        [&](int primitive_index) {
          const int pair_index =
              pair_primitives.at(static_cast<size_t>(primitive_index)).id;
          const ContactWorkList::BodyPairWork& pair =
              work.bodyPairs.at(static_cast<size_t>(pair_index));
          const int side = bodyPairSideForPrimitiveBody(pair, owner);
          if (side < 0) {
            return false;
          }
          refs.at(write++) = BodyPairPrimitiveRef{
              .pair = pair_index,
              .primitive = static_cast<int>(index),
              .side = static_cast<std::uint8_t>(side),
          };
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });
}

void gatherEdgeRefsThroughBodyPairBVH(
    const GlobalGeometryManager& geometry, const std::vector<PrimitiveRef>& edges,
    const std::vector<PrimitiveRef>& pair_primitives,
    const spatify::LBVH<Real>& pair_tree, const ContactWorkList& work,
    std::vector<BodyPairPrimitiveRef>& refs) {
  std::vector<size_t> counts(edges.size(), size_t{0});

  tbb::parallel_for(size_t{0}, edges.size(), [&](size_t index) {
    const PrimitiveRef& edge = edges[index];
    const GeometryBodyIdx owner = edgeBody(geometry, edge.id);
    const GeometryBounds query_bounds = ensureFiniteBVHBounds(edge.bounds);
    pair_tree.runSpatialQuery(
        [&](int primitive_index) {
          const int pair_index =
              pair_primitives.at(static_cast<size_t>(primitive_index)).id;
          const ContactWorkList::BodyPairWork& pair =
              work.bodyPairs.at(static_cast<size_t>(pair_index));
          if (bodyPairSideForPrimitiveBody(pair, owner) < 0) {
            return false;
          }
          ++counts[index];
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });

  const std::vector<size_t> offsets = exclusiveOffsets(counts);
  refs.resize(offsets.back());

  tbb::parallel_for(size_t{0}, edges.size(), [&](size_t index) {
    const PrimitiveRef& edge = edges[index];
    const GeometryBodyIdx owner = edgeBody(geometry, edge.id);
    const GeometryBounds query_bounds = ensureFiniteBVHBounds(edge.bounds);
    size_t write = offsets[index];
    pair_tree.runSpatialQuery(
        [&](int primitive_index) {
          const int pair_index =
              pair_primitives.at(static_cast<size_t>(primitive_index)).id;
          const ContactWorkList::BodyPairWork& pair =
              work.bodyPairs.at(static_cast<size_t>(pair_index));
          const int side = bodyPairSideForPrimitiveBody(pair, owner);
          if (side < 0) {
            return false;
          }
          refs.at(write++) = BodyPairPrimitiveRef{
              .pair = pair_index,
              .primitive = static_cast<int>(index),
              .side = static_cast<std::uint8_t>(side),
          };
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });
}

[[nodiscard]] RigidInvolvedPairRefs buildRigidInvolvedPairRefs(
    const GlobalGeometryManager& geometry, const std::vector<PrimitiveRef>& points,
    const std::vector<PrimitiveRef>& edges, const ContactWorkList& work) {
  RigidInvolvedPairRefs refs;
  std::vector<PrimitiveRef> pair_primitives = buildBodyPairPrimitives(work);
  if (pair_primitives.empty())
    return refs;

  const spatify::LBVH<Real> pair_tree = buildLBVH(pair_primitives);
  gatherPointRefsThroughBodyPairBVH(geometry, points, pair_primitives,
                                    pair_tree, work, refs.points);
  gatherEdgeRefsThroughBodyPairBVH(geometry, edges, pair_primitives, pair_tree,
                                   work, refs.edges);
  return refs;
}

void queryPointRefsAgainstRigidTriangles(
    const GlobalGeometryManager& geometry, const std::vector<PrimitiveRef>& points,
    const std::vector<PrimitiveRef>& rigid_triangles,
    const spatify::LBVH<Real>& rigid_triangle_tree,
    const RigidInvolvedPairRefs& refs, const ContactWorkList& work,
    Real query_dilation, std::vector<std::array<int, 2>>& point_triangles) {
  if (rigid_triangles.empty()) {
    return;
  }
  std::vector<size_t> counts(refs.points.size(), size_t{0});

  tbb::parallel_for(size_t{0}, refs.points.size(), [&](size_t index) {
    const BodyPairPrimitiveRef& ref = refs.points[index];
    const ContactWorkList::BodyPairWork& pair =
        work.bodyPairs.at(static_cast<size_t>(ref.pair));
    const GeometryBodyIdx target_body = oppositeBody(pair, ref.side);
    if (!isRigidBody(geometry, target_body)) {
      return;
    }
    const PrimitiveRef& point = points.at(static_cast<size_t>(ref.primitive));
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(point.bounds.dilate(query_dilation));
    rigid_triangle_tree.runSpatialQuery(
        [&](int primitive_index) {
          const PrimitiveRef& triangle =
              rigid_triangles[static_cast<size_t>(primitive_index)];
          if (triangleBody(geometry, triangle.id) != target_body)
            return false;

          if (!triangle.bounds.overlap(pair.iAABB))
            return false;

          if (geometry.triangleContainsPoint(triangle.id,
                                             PointIdx{point.id}))
            return false;

          if (!pointTriangleCollisionEnabled(geometry, PointIdx{point.id},
                                             triangle.id))
            return false;

          ++counts[index];
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
        });
  });

  const std::vector<size_t> offsets = exclusiveOffsets(counts);
  const size_t old_size = point_triangles.size();
  point_triangles.resize(old_size + offsets.back());

  tbb::parallel_for(size_t{0}, refs.points.size(), [&](size_t index) {
    const BodyPairPrimitiveRef& ref = refs.points[index];
    const ContactWorkList::BodyPairWork& pair =
        work.bodyPairs.at(static_cast<size_t>(ref.pair));
    const GeometryBodyIdx target_body = oppositeBody(pair, ref.side);
    if (!isRigidBody(geometry, target_body)) {
      return;
    }
    const PrimitiveRef& point = points.at(static_cast<size_t>(ref.primitive));
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(point.bounds.dilate(query_dilation));
    size_t write = old_size + offsets[index];
    rigid_triangle_tree.runSpatialQuery(
        [&](int primitive_index) {
          const PrimitiveRef& triangle =
              rigid_triangles[static_cast<size_t>(primitive_index)];
          if (triangleBody(geometry, triangle.id) != target_body)
            return false;

          if (!triangle.bounds.overlap(pair.iAABB))
            return false;

          if (geometry.triangleContainsPoint(triangle.id,
                                             PointIdx{point.id}))
            return false;

          if (!pointTriangleCollisionEnabled(geometry, PointIdx{point.id},
                                             triangle.id))
            return false;

          point_triangles.at(write++) = {point.id, triangle.id};
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
        });
  });
}

void queryEdgeRefsAgainstRigidEdges(
    const GlobalGeometryManager& geometry, const std::vector<PrimitiveRef>& edges,
    const std::vector<PrimitiveRef>& rigid_edges,
    const spatify::LBVH<Real>& rigid_edge_tree,
    const RigidInvolvedPairRefs& refs, const ContactWorkList& work,
    Real query_dilation, std::vector<std::array<int, 2>>& edge_edges) {
  if (rigid_edges.empty()) {
    return;
  }
  std::vector<size_t> counts(refs.edges.size(), size_t{0});

  tbb::parallel_for(size_t{0}, refs.edges.size(), [&](size_t index) {
    const BodyPairPrimitiveRef& ref = refs.edges[index];
    const ContactWorkList::BodyPairWork& pair =
        work.bodyPairs.at(static_cast<size_t>(ref.pair));
    const GeometryBodyIdx target_body = oppositeBody(pair, ref.side);
    if (!isRigidBody(geometry, target_body)) {
      return;
    }
    const PrimitiveRef& query_edge = edges.at(static_cast<size_t>(ref.primitive));
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(query_edge.bounds.dilate(query_dilation));
    rigid_edge_tree.runSpatialQuery(
        [&](int primitive_index) {
          const PrimitiveRef& candidate =
              rigid_edges.at(static_cast<size_t>(primitive_index));
          if (edgeBody(geometry, candidate.id) != target_body)
            return false;

          if (!candidate.bounds.overlap(pair.iAABB))
            return false;

          if (query_edge.id == candidate.id ||
              geometry.edgesAdjacent(query_edge.id, candidate.id))
            return false;

          if (!edgeEdgeCollisionEnabled(geometry, query_edge.id,
                                        candidate.id))
            return false;

          ++counts[index];
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
        });
  });

  const std::vector<size_t> offsets = exclusiveOffsets(counts);
  const size_t old_size = edge_edges.size();
  edge_edges.resize(old_size + offsets.back());

  tbb::parallel_for(size_t{0}, refs.edges.size(), [&](size_t index) {
    const BodyPairPrimitiveRef& ref = refs.edges[index];
    const ContactWorkList::BodyPairWork& pair =
        work.bodyPairs.at(static_cast<size_t>(ref.pair));
    const GeometryBodyIdx target_body = oppositeBody(pair, ref.side);
    if (!isRigidBody(geometry, target_body)) {
      return;
    }
    const PrimitiveRef& query_edge = edges.at(static_cast<size_t>(ref.primitive));
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(query_edge.bounds.dilate(query_dilation));
    size_t write = old_size + offsets[index];
    rigid_edge_tree.runSpatialQuery(
        [&](int primitive_index) {
          const PrimitiveRef& candidate =
              rigid_edges.at(static_cast<size_t>(primitive_index));
          if (edgeBody(geometry, candidate.id) != target_body)
            return false;

          if (!candidate.bounds.overlap(pair.iAABB))
            return false;

          if (query_edge.id == candidate.id ||
              geometry.edgesAdjacent(query_edge.id, candidate.id))
            return false;

          if (!edgeEdgeCollisionEnabled(geometry, query_edge.id,
                                        candidate.id))
            return false;

          if (query_edge.id <= candidate.id)
            edge_edges.at(write++) = {query_edge.id, candidate.id};
          else
            edge_edges.at(write++) = {candidate.id, query_edge.id};

          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
        });
  });
}

void gatherGeometryBodyBounds(const GlobalGeometryManager& geometry,
                              const std::vector<PrimitiveRef>& points,
                              const std::vector<PrimitiveRef>& edges,
                              const std::vector<PrimitiveRef>& triangles,
                              ContactWorkList& work) {
  for (GeometryBodyIdx body_id = 0; body_id < geometry.geometryBodies.size();
       body_id++) {
    GeometryBounds bounds;
    bool has_bounds = false;
    const auto expand_body_primitive = [&](const PrimitiveRef& primitive,
                                           GeometryBodyIdx primitive_body) {
      if (primitive_body != body_id) {
        return;
      }
      bounds.expand(primitive.bounds.lo);
      bounds.expand(primitive.bounds.hi);
      has_bounds = true;
    };

    for (const PrimitiveRef& point : points) {
      expand_body_primitive(point, pointBody(geometry, point.id));
    }
    for (const PrimitiveRef& edge : edges) {
      expand_body_primitive(edge, edgeBody(geometry, edge.id));
    }
    for (const PrimitiveRef& triangle : triangles) {
      expand_body_primitive(triangle, triangleBody(geometry, triangle.id));
    }

    if (!has_bounds) {
      bounds = geometry.bodyBounds(body_id);
    }

    assert(isFiniteBounds(bounds));
    work.bodies.push_back(ContactWorkList::GeometryBodyBounds{
        .body = body_id,
        .bounds = bounds,
    });
  }
}

void gatherGlobalDeformablePointTriangleWork(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& deformable_points,
    const std::vector<PrimitiveRef>& deformable_triangles,
    Real query_dilation, std::vector<std::array<int, 2>>& point_triangles) {
  if (deformable_triangles.empty())
    return;

  std::vector<PrimitiveRef> tree_triangles = deformable_triangles;
  const spatify::LBVH<Real> triangle_tree = buildLBVH(tree_triangles);
  std::vector<size_t> counts(deformable_points.size(), size_t{0});

  tbb::parallel_for(size_t{0}, deformable_points.size(), [&](size_t index) {
    const PrimitiveRef& point = deformable_points[index];
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(point.bounds.dilate(query_dilation));
    triangle_tree.runSpatialQuery(
        [&](int primitive_index) {
          const PrimitiveRef& triangle = tree_triangles[primitive_index];

          if (geometry.triangleContainsPoint(triangle.id,
                                             PointIdx{point.id}))
            return false;

          if (!pointTriangleCollisionEnabled(geometry, PointIdx{point.id},
                                             triangle.id))
            return false;

          ++counts[index];
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });

  const std::vector<size_t> offsets = exclusiveOffsets(counts);
  const size_t old_size = point_triangles.size();
  point_triangles.resize(old_size + offsets.back());

  tbb::parallel_for(size_t{0}, deformable_points.size(), [&](size_t index) {
    const PrimitiveRef& point = deformable_points[index];
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(point.bounds.dilate(query_dilation));
    size_t write = old_size + offsets[index];
    triangle_tree.runSpatialQuery(
        [&](int primitive_index) {
          const PrimitiveRef& triangle = tree_triangles[primitive_index];
          if (geometry.triangleContainsPoint(triangle.id,
                                             PointIdx{point.id}))
            return false;

          if (!pointTriangleCollisionEnabled(geometry, PointIdx{point.id},
                                             triangle.id))
            return false;

          point_triangles.at(write++) = {point.id, triangle.id};
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });
}

void gatherGlobalDeformableEdgeEdgeWork(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& deformable_edges, Real query_dilation,
    std::vector<std::array<int, 2>>& edge_edges) {
  if (deformable_edges.empty()) {
    return;
  }

  std::vector<PrimitiveRef> tree_edges = deformable_edges;
  const spatify::LBVH<Real> edge_tree = buildLBVH(tree_edges);
  std::vector<size_t> counts(deformable_edges.size(), size_t{0});

  tbb::parallel_for(size_t{0}, deformable_edges.size(), [&](size_t index) {
    const PrimitiveRef& query_edge = deformable_edges[index];
    const GeometryBounds query_bounds =
        query_edge.bounds.dilate(query_dilation);

    edge_tree.runSpatialQuery(
        [&](int primitive_index) {
          const PrimitiveRef& candidate = tree_edges[primitive_index];

          if (candidate.id <= query_edge.id)
            return false;

          if (geometry.edgesAdjacent(query_edge.id, candidate.id))
            return false;

          if (!edgeEdgeCollisionEnabled(geometry, query_edge.id,
                                        candidate.id))
            return false;

          ++counts[index];
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });

  const std::vector<size_t> offsets = exclusiveOffsets(counts);
  const size_t old_size = edge_edges.size();
  edge_edges.resize(old_size + offsets.back());

  tbb::parallel_for(size_t{0}, deformable_edges.size(), [&](size_t index) {
    const PrimitiveRef& query_edge = deformable_edges[index];
    const GeometryBounds query_bounds =
        query_edge.bounds.dilate(query_dilation);
    size_t write = old_size + offsets[index];

    edge_tree.runSpatialQuery(
        [&](int primitive_index) {
          const PrimitiveRef& candidate =
              tree_edges[static_cast<size_t>(primitive_index)];
          if (candidate.id <= query_edge.id)
            return false;

          if (geometry.edgesAdjacent(query_edge.id, candidate.id))
            return false;

          if (!edgeEdgeCollisionEnabled(geometry, query_edge.id,
                                        candidate.id))
            return false;

          edge_edges.at(write++) = {query_edge.id, candidate.id};
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });
}

void gatherDeformableDeformableCollisionWork(
    const GlobalGeometryManager& geometry, const std::vector<PrimitiveRef>& points,
    const std::vector<PrimitiveRef>& edges,
    const std::vector<PrimitiveRef>& triangles, Real query_dilation,
    ContactWorkList& work) {
  std::vector<PrimitiveRef> deformable_points =
      collectDeformablePointPrimitives(geometry, points);
  std::vector<PrimitiveRef> deformable_triangles =
      collectDeformableTrianglePrimitives(geometry, triangles);
  std::vector<PrimitiveRef> deformable_edges =
      collectDeformableEdgePrimitives(geometry, edges);

  gatherGlobalDeformablePointTriangleWork(
      geometry, deformable_points, deformable_triangles, query_dilation,
      work.pointTriangles);
  gatherGlobalDeformableEdgeEdgeWork(
      geometry, deformable_edges, query_dilation, work.edgeEdges);
}

void gatherRigidInvolvedBodyPairs(const GlobalGeometryManager& geometry,
                                  Real query_dilation,
                                  ContactWorkList& work) {
  if (work.bodies.size() < 2)
    return;

  std::vector<PrimitiveRef> body_refs;
  body_refs.reserve(work.bodies.size());
  for (int body_index = 0; body_index < work.bodies.size(); body_index++) {
    const GeometryBodyIdx body =
        work.bodies.at(static_cast<size_t>(body_index)).body;
    if (!isRigidBody(geometry, body)) {
      continue;
    }
    body_refs.push_back(PrimitiveRef{
        .id = body_index,
        .bounds = work.bodies.at(static_cast<size_t>(body_index)).bounds,
    });
  }
  if (body_refs.empty()) {
    return;
  }

  const spatify::LBVH<Real> body_tree = buildLBVH(body_refs);
  std::vector<size_t> counts(work.bodies.size(), size_t{0});

  tbb::parallel_for(size_t{0}, work.bodies.size(), [&](size_t first) {
    const ContactWorkList::GeometryBodyBounds& first_record =
        work.bodies.at(first);
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(first_record.bounds);
    body_tree.runSpatialQuery(
        [&](int primitive_index) {
          const int second =
              body_refs.at(static_cast<size_t>(primitive_index)).id;
          if (second <= first)
            return false;

          const GeometryBodyIdx first_body = first_record.body;
          const GeometryBodyIdx second_body =
              work.bodies.at(static_cast<size_t>(second)).body;
          assert(isRigidBody(geometry, second_body));
          if (!isRigidBody(geometry, first_body) &&
              !isRigidBody(geometry, second_body))
            return false;

          if (!geometry.collisionPairEnabled(first_body, second_body))
            return false;

          ++counts[first];
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });

  const std::vector<size_t> offsets = exclusiveOffsets(counts);
  const size_t old_size = work.bodyPairs.size();
  if (old_size + offsets.back() > 0)
    work.bodyPairs.resize(old_size + offsets.back());
  else work.bodyPairs.clear();

  tbb::parallel_for(size_t{0}, work.bodies.size(), [&](size_t first) {
    const ContactWorkList::GeometryBodyBounds& first_record =
        work.bodies.at(first);
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(first_record.bounds);
    size_t write = old_size + offsets[first];

    body_tree.runSpatialQuery(
        [&](int primitive_index) {
          const int second =
              body_refs.at(static_cast<size_t>(primitive_index)).id;
          if (second <= static_cast<int>(first)) {
            return false;
          }

          const ContactWorkList::GeometryBodyBounds& second_record =
              work.bodies.at(static_cast<size_t>(second));
          const GeometryBodyIdx first_body = first_record.body;
          const GeometryBodyIdx second_body = second_record.body;
          if (!isRigidBody(geometry, first_body) &&
              !isRigidBody(geometry, second_body))
            return false;

          if (!geometry.collisionPairEnabled(first_body, second_body))
            return false;

          GeometryBounds i_aabb =
              intersectBounds(first_record.bounds, second_record.bounds);
          if (!isFiniteBounds(i_aabb))
            return false;

          i_aabb.dilate(query_dilation);
          work.bodyPairs.at(write++) = ContactWorkList::BodyPairWork{
              .first = first_body,
              .second = second_body,
              .iAABB = i_aabb,
              .policy = ContactWorkList::EContactPairPolicy::BodyPairIAABB,
          };
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });
}

void gatherRigidInvolvedCollisionWork(
    const GlobalGeometryManager& geometry, const std::vector<PrimitiveRef>& points,
    const std::vector<PrimitiveRef>& edges,
    const std::vector<PrimitiveRef>& triangles, Real query_dilation,
    ContactWorkList& work) {
  gatherRigidInvolvedBodyPairs(geometry, query_dilation, work);
  if (work.bodyPairs.empty())
    return;

  std::vector<PrimitiveRef> rigid_triangles =
      collectRigidTrianglePrimitives(geometry, triangles);
  std::vector<PrimitiveRef> rigid_edges =
      collectRigidEdgePrimitives(geometry, edges);

  const spatify::LBVH<Real> rigid_triangle_tree = buildLBVH(rigid_triangles);
  const spatify::LBVH<Real> rigid_edge_tree = buildLBVH(rigid_edges);

  const RigidInvolvedPairRefs rigid_pair_refs =
      buildRigidInvolvedPairRefs(geometry, points, edges, work);

  queryPointRefsAgainstRigidTriangles(
      geometry, points, rigid_triangles, rigid_triangle_tree, rigid_pair_refs,
      work, query_dilation, work.pointTriangles);
  queryEdgeRefsAgainstRigidEdges(geometry, edges, rigid_edges, rigid_edge_tree,
                                 rigid_pair_refs, work, query_dilation,
                                 work.edgeEdges);
}

int stencilPointCount(ContactCase type) {
  switch (type) {
    case ContactCase::PP:
      return 2;
    case ContactCase::PE:
      return 3;
    case ContactCase::PT:
    case ContactCase::EE:
      return 4;
  }
  return 0;
}

void routeStencil(GlobalContactRouter& routed,
                  const GlobalGeometryManager& geometry_manager,
                  const ContactStencil& stencil) {
  const int point_count = stencilPointCount(stencil.type);
  const GeometryStencilInfo info = geometry_manager.classify(
      std::span(stencil.geometryIds.data(), point_count));

  if (info.subsystemCount == 1) {
    auto& subsystem_contacts =
        routed.subsystemInternalContacts[info.subsystems[0]];
    subsystem_contacts.subsystem = info.subsystems[0];
    subsystem_contacts.contacts.push_back(stencil);
    return;
  }

  routed.globalContacts.push_back(stencil);
}

std::optional<ContactStencil> lowerCandidateToBarrierStencil(
    const ContactCandidate& candidate, const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config) {
  switch (candidate.type) {
    case EContactCandidate::PointTriangle:
      return createPointTriangleStencil(PointIdx{candidate.geometryIds[0]},
                                        {PointIdx{candidate.geometryIds[1]},
                                         PointIdx{candidate.geometryIds[2]},
                                         PointIdx{candidate.geometryIds[3]}},
                                        geometry, direction, config, 0.0);
    case EContactCandidate::EdgeEdge:
      return createEdgeEdgeStencil({PointIdx{candidate.geometryIds[0]},
                                    PointIdx{candidate.geometryIds[1]}},
                                   {PointIdx{candidate.geometryIds[2]},
                                    PointIdx{candidate.geometryIds[3]}},
                                   geometry, direction, config, 0.0);
    case EContactCandidate::PointPoint: {
      const PointIdx first{candidate.geometryIds[0]};
      const PointIdx second{candidate.geometryIds[1]};
      const Real reserved_dist =
          pointRadius(geometry, first) + pointRadius(geometry, second);
      const Real distance_sqr = distanceSqrPointPoint(
          geometry.worldPosition(first), geometry.worldPosition(second));
      if (distance_sqr >
          (config.dHat + reserved_dist) * (config.dHat + reserved_dist)) {
        return std::nullopt;
      }
      return createPointPointStencil(first, second, geometry, config);
    }
    case EContactCandidate::PointEdge: {
      const PointIdx point{candidate.geometryIds[0]};
      const PointIdx edge0{candidate.geometryIds[1]};
      const PointIdx edge1{candidate.geometryIds[2]};
      const Real fallback_edge_offset = std::max(
          0.0, candidate.reservedDistance - pointRadius(geometry, point));
      const Real reserved_dist = pointEdgeSurfaceOffset(
          geometry, point, edge0, edge1, fallback_edge_offset);
      const Real distance_sqr = distanceSqrPointLine(
          geometry.worldPosition(point), geometry.worldPosition(edge0),
          geometry.worldPosition(edge1));
      if (distance_sqr >
          (config.dHat + reserved_dist) * (config.dHat + reserved_dist)) {
        return std::nullopt;
      }
      const Real t = pointEdgeProjectionParameter(
          geometry.worldPosition(point), geometry.worldPosition(edge0),
          geometry.worldPosition(edge1));

      if (t <= 0.0)
        return createPointPointStencil(point, edge0, geometry, config);

      if (t >= 1.0)
        return createPointPointStencil(point, edge1, geometry, config);

      return createPointEdgeStencil(point, edge0, edge1, geometry, config,
                                    fallback_edge_offset);
    }
  }
  return std::nullopt;
}

std::optional<Real> computeCandidateTOI(
    const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction, const ContactCandidate& candidate,
    Real toi) {
  switch (candidate.type) {
    case EContactCandidate::PointTriangle: {
      const PointIdx point{candidate.geometryIds[0]};
      const std::array<PointIdx, 3> triangle{
          PointIdx{candidate.geometryIds[1]},
          PointIdx{candidate.geometryIds[2]},
          PointIdx{candidate.geometryIds[3]}};
      return runACCD(
          false,
          {geometry.worldPosition(point), geometry.worldPosition(triangle[0]),
           geometry.worldPosition(triangle[1]),
           geometry.worldPosition(triangle[2])},
          {direction[point], direction[triangle[0]], direction[triangle[1]],
           direction[triangle[2]]},
          toi, candidate.reservedDistance);
    }
    case EContactCandidate::EdgeEdge: {
      const std::array<PointIdx, 2> first{PointIdx{candidate.geometryIds[0]},
                                          PointIdx{candidate.geometryIds[1]}};
      const std::array<PointIdx, 2> second{PointIdx{candidate.geometryIds[2]},
                                           PointIdx{candidate.geometryIds[3]}};
      return runACCD(
          true,
          {geometry.worldPosition(first[0]), geometry.worldPosition(first[1]),
           geometry.worldPosition(second[0]),
           geometry.worldPosition(second[1])},
          {direction[first[0]], direction[first[1]], direction[second[0]],
           direction[second[1]]},
          toi, candidate.reservedDistance);
    }
    case EContactCandidate::PointPoint:
    case EContactCandidate::PointEdge:
      return std::nullopt;
  }
  return std::nullopt;
}
}  // namespace

namespace detail {
namespace {
void addCandidate(ContactCandidateDetectionResult& result,
                  const GlobalGeometryManager& geometry,
                  const GeometryBuffer& geometryDirection,
                  const ContactDetectionConfig& config, bool compute_step_size,
                  ContactCandidate candidate) {
  if (compute_step_size) {
    const std::optional<Real> toi = computeCandidateTOI(
        geometry, geometryDirection.cpu(), candidate, config.toi);
    if (toi) {
      result.stepSizeUpperBound =
          std::min(result.stepSizeUpperBound, std::clamp(*toi, 0.0, 1.0));
    }
  }
  result.candidates.push_back(std::move(candidate));
}

ContactCandidateDetectionResult buildCandidatesFromCollisionWorkListOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config, const ContactWorkList& work,
    bool compute_step_size) {
  ContactCandidateDetectionResult result;

#ifndef NDEBUG
  validateCPUCCDInput(geometry, geometryDirection);
#endif

  for (const auto& pair : work.pointTriangles) {
    const PointIdx point = PointIdx{pair[0]};
    const int triangle_id = pair[1];
    const auto triangle = geometry.globalTriangle(triangle_id);
    const Real primitive_thickness =
        pointRadius(geometry, point) +
        geometry.triangles.at(static_cast<size_t>(triangle_id)).thickness;
    addCandidate(result, geometry, geometryDirection, config, compute_step_size,
                 makeCandidate(EContactCandidate::PointTriangle,
                               {point, triangle[0], triangle[1], triangle[2]},
                               config, primitive_thickness));
  }

  for (const auto& pair : work.edgeEdges) {
    const int first_id = pair[0];
    const int second_id = pair[1];
    const auto first = geometry.globalEdge(first_id);
    const auto second = geometry.globalEdge(second_id);
    const Real primitive_thickness =
        geometry.edges.at(static_cast<size_t>(first_id)).radius +
        geometry.edges.at(static_cast<size_t>(second_id)).radius;
    addCandidate(result, geometry, geometryDirection, config, compute_step_size,
                 makeCandidate(EContactCandidate::EdgeEdge,
                               {first[0], first[1], second[0], second[1]},
                               config, primitive_thickness));
  }

  return result;
}

}  // namespace

ContactWorkList gatherCollisionWorkListAlongDirectionOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config) {
  if (config.storage == ContactDetectionStorage::Device) {
    throw std::runtime_error(
        "CPU contact detection cannot produce device collision work lists");
  }
  ContactWorkList work;

#ifndef NDEBUG
  validateCPUCCDInput(geometry, geometryDirection);
#endif

  const Real query_dilation = config.dHat;

  std::vector<PrimitiveRef> points =
      buildPointPrimitives(geometry, geometryDirection, config.toi);

  std::vector<PrimitiveRef> triangles =
      buildTrianglePrimitives(geometry, geometryDirection, config.toi);

  std::vector<PrimitiveRef> edges =
      buildEdgePrimitives(geometry, geometryDirection, config.toi);

  gatherGeometryBodyBounds(geometry, points, edges, triangles, work);

  gatherDeformableDeformableCollisionWork(
      geometry, points, edges, triangles, query_dilation, work);

  gatherRigidInvolvedCollisionWork(geometry, points, edges, triangles,
                                   query_dilation, work);

  finalizeCollisionWorkList(work);
  return work;
}

ContactCandidates detectContactCandidatesAlongDirectionOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config) {
  if (config.storage == ContactDetectionStorage::Device) {
    throw std::runtime_error(
        "CPU contact detection cannot produce device contact tables");
  }
  const ContactWorkList work = gatherCollisionWorkListAlongDirectionOnCPU(
      geometry, geometryDirection, config);
  return buildCandidatesFromCollisionWorkListOnCPU(geometry, geometryDirection,
                                                   config, work, false)
      .candidates;
}

ContactCandidateDetectionResult
detectContactCandidatesAndStepSizeAlongDirectionOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config) {
  if (config.storage == ContactDetectionStorage::Device) {
    throw std::runtime_error(
        "CPU contact detection cannot produce device contact tables");
  }
  const ContactWorkList work = gatherCollisionWorkListAlongDirectionOnCPU(
      geometry, geometryDirection, config);
  return buildCandidatesFromCollisionWorkListOnCPU(geometry, geometryDirection,
                                                   config, work, true);
}

GlobalContactRouter refreshActiveContactsFromCandidatesOnCPU(
    const GlobalGeometryManager& geometry, const ContactCandidates& candidates,
    const ContactDetectionConfig& config) {
  GlobalContactRouter routed;
  const std::vector zero_direction(geometry.pointCount(), glm::dvec3(0.0));
  std::vector<std::optional<ContactStencil>> active_contacts(candidates.size());
  tbb::parallel_for(size_t{0}, candidates.size(), [&](size_t index) {
    active_contacts[index] = lowerCandidateToBarrierStencil(
        candidates[index], geometry, zero_direction, config);
  });

  for (const std::optional<ContactStencil>& active : active_contacts)
    if (active) routeStencil(routed, geometry, *active);

  return routed;
}

GlobalContactRouter runCCDOnCPU(const GlobalGeometryManager& geometry,
                                const GeometryBuffer& geometryDirection,
                                const ContactDetectionConfig& config) {
  const ContactWorkList work = gatherCollisionWorkListAlongDirectionOnCPU(
      geometry, geometryDirection, config);

  GlobalContactRouter routed;
#ifndef NDEBUG
  validateCPUCCDInput(geometry, geometryDirection);
#endif
  std::vector<std::optional<ContactStencil>> point_triangle_contacts(
      work.pointTriangles.size());

  tbb::parallel_for(size_t{0}, work.pointTriangles.size(), [&](size_t index) {
    const auto& pair = work.pointTriangles[index];
    const PointIdx point = PointIdx{pair[0]};
    const auto triangle = geometry.globalTriangle(pair[1]);
    point_triangle_contacts[index] = createPointTriangleStencil(
        point, triangle, geometry, geometryDirection.cpu(), config, 0.0);
  });

  for (auto&& active : point_triangle_contacts)
    if (active) routeStencil(routed, geometry, *active);

  std::vector<std::optional<ContactStencil>> edge_edge_contacts(
      work.edgeEdges.size());
  tbb::parallel_for(size_t{0}, work.edgeEdges.size(), [&](size_t index) {
    const auto& pair = work.edgeEdges[index];
    const auto first = geometry.globalEdge(pair[0]);
    const auto second = geometry.globalEdge(pair[1]);
    edge_edge_contacts[index] = createEdgeEdgeStencil(
        first, second, geometry, geometryDirection.cpu(), config, 0.0);
  });

  for (auto&& active : edge_edge_contacts)
    if (active) routeStencil(routed, geometry, *active);
  return routed;
}
}  // namespace detail
}  // namespace ksk::runtime
