#include <Runtime/contact-detection.h>

#include <Contact/contact-barrier.h>
#include <Core/dynamic-bitset.h>
#include <Core/profiler.h>
#include <Spatify/lbvh.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cassert>
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

enum class PrimitiveKind {
  Point,
  Edge,
  Triangle,
};

struct ContactPrimitiveTopology {
  const GlobalGeometryManager* geometry = nullptr;
  int point_count = 0;
  int edge_count = 0;
  int triangle_count = 0;
  std::vector<PrimitiveRef> points;
  std::vector<PrimitiveRef> edges;
  std::vector<PrimitiveRef> triangles;
  std::vector<size_t> deformable_points;
  std::vector<size_t> deformable_edges;
  std::vector<size_t> deformable_triangles;
  std::vector<size_t> rigid_edges;
  std::vector<size_t> rigid_triangles;
};

struct ContactPrimitiveRefs {
  std::vector<PrimitiveRef> points;
  std::vector<PrimitiveRef> edges;
  std::vector<PrimitiveRef> triangles;
  std::vector<PrimitiveRef> deformable_points;
  std::vector<PrimitiveRef> deformable_edges;
  std::vector<PrimitiveRef> deformable_triangles;
  std::vector<PrimitiveRef> rigid_edges;
  std::vector<PrimitiveRef> rigid_triangles;
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

struct IndexedPrimitiveAccessor {
  using CoordType = Real;

  const std::vector<PrimitiveRef>* primitives = nullptr;

  [[nodiscard]] spatify::BBox<Real, 3> bbox(int index) const {
    return primitives->at(static_cast<size_t>(index)).bounds;
  }

  [[nodiscard]] int primitiveIndex(int index) const {
    return primitives->at(static_cast<size_t>(index)).id;
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

struct CandidateHit {
  ContactCandidate candidate;
  std::optional<Real> toi;
};

struct CandidateDetectionResult {
  ContactCandidates candidates;
  Real stepSizeUpperBound = 1.0;
};

struct StepSizeUpperBoundResult {
  Real stepSizeUpperBound = 1.0;
  size_t testedPairs = 0;
  size_t collidingPairs = 0;
};

struct CurrentActiveContactWork {
  ContactWorkList broadPhase;
  ContactStencils stencils;
  size_t pointTriangleLeafPairs = 0;
  size_t edgeEdgeLeafPairs = 0;
};

int stencilPointCount(ContactCase type);
Real pointEdgeProjectionParameter(const glm::dvec3& p, const glm::dvec3& a,
                                  const glm::dvec3& b);

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
//  makeBoundsSafeForLBVH(primitives);
  spatify::LBVH<Real> tree;

  if (!primitives.empty())
    tree.update(PrimitiveAccessor{.primitives = &primitives});

  return tree;
}

[[nodiscard]] spatify::LBVH<Real> buildIndexedLBVH(
    std::vector<PrimitiveRef>& primitives) {
//  makeBoundsSafeForLBVH(primitives);
  spatify::LBVH<Real> tree;

  if (!primitives.empty())
    tree.update(IndexedPrimitiveAccessor{.primitives = &primitives});

  return tree;
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

glm::dvec3 safeNormalize(glm::dvec3 value, glm::dvec3 fallback) {
  const Real length = glm::length(value);
  if (std::isfinite(length) && length > 1.0e-12) {
    return value / length;
  }
  const Real fallback_length = glm::length(fallback);
  if (std::isfinite(fallback_length) && fallback_length > 1.0e-12) {
    return fallback / fallback_length;
  }
  return glm::dvec3(0.0, 1.0, 0.0);
}

std::array<Real, 3> triangleBarycentric(const glm::dvec3& p,
                                        const glm::dvec3& a,
                                        const glm::dvec3& b,
                                        const glm::dvec3& c) {
  const glm::dvec3 v0 = b - a;
  const glm::dvec3 v1 = c - a;
  const glm::dvec3 v2 = p - a;
  const Real d00 = glm::dot(v0, v0);
  const Real d01 = glm::dot(v0, v1);
  const Real d11 = glm::dot(v1, v1);
  const Real d20 = glm::dot(v2, v0);
  const Real d21 = glm::dot(v2, v1);
  const Real denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) <= 1.0e-18) {
    return {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  }
  Real v = (d11 * d20 - d01 * d21) / denom;
  Real w = (d00 * d21 - d01 * d20) / denom;
  Real u = 1.0 - v - w;
  u = std::clamp(u, 0.0, 1.0);
  v = std::clamp(v, 0.0, 1.0);
  w = std::clamp(w, 0.0, 1.0);
  const Real sum = u + v + w;
  if (sum > 1.0e-18) {
    u /= sum;
    v /= sum;
    w /= sum;
  }
  return {u, v, w};
}

std::array<Real, 2> closestSegmentParameters(const glm::dvec3& p1,
                                             const glm::dvec3& q1,
                                             const glm::dvec3& p2,
                                             const glm::dvec3& q2) {
  const glm::dvec3 d1 = q1 - p1;
  const glm::dvec3 d2 = q2 - p2;
  const glm::dvec3 r = p1 - p2;
  const Real a = glm::dot(d1, d1);
  const Real e = glm::dot(d2, d2);
  const Real f = glm::dot(d2, r);
  Real s = 0.0;
  Real t = 0.0;

  if (a <= 1.0e-18 && e <= 1.0e-18) {
    return {0.0, 0.0};
  }
  if (a <= 1.0e-18) {
    t = std::clamp(f / e, 0.0, 1.0);
  } else {
    const Real c = glm::dot(d1, r);
    if (e <= 1.0e-18) {
      s = std::clamp(-c / a, 0.0, 1.0);
    } else {
      const Real b = glm::dot(d1, d2);
      const Real denom = a * e - b * b;
      if (std::abs(denom) > 1.0e-18) {
        s = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
      }
      t = (b * s + f) / e;
      if (t < 0.0) {
        t = 0.0;
        s = std::clamp(-c / a, 0.0, 1.0);
      } else if (t > 1.0) {
        t = 1.0;
        s = std::clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }
  return {s, t};
}

Real contactFrictionCoefficient(const ContactStencil& stencil,
                                const GlobalGeometryManager& geometry,
                                const ContactDetectionConfig& config) {
  Real coefficient = config.frictionCoefficient;
  bool has_override = false;
  const int count = stencilPointCount(stencil.type);
  for (int i = 0; i < count; ++i) {
    const int id = stencil.geometryIds[static_cast<size_t>(i)];
    if (id < 0 || id >= geometry.pointCount()) {
      continue;
    }
    const GeometryBodyIdx body = geometry.pointOwner(PointIdx{id}).body;
    if (body < 0 ||
        body >= static_cast<int>(geometry.bodyCollisionFilters.size())) {
      continue;
    }
    const Real body_coefficient =
        geometry.bodyCollisionFilters[static_cast<size_t>(body)]
            .frictionCoefficient;
    if (body_coefficient < 0.0) {
      continue;
    }
    coefficient =
        has_override ? std::min(coefficient, body_coefficient)
                     : body_coefficient;
    has_override = true;
  }
  return coefficient;
}

ksk::engine::contact::EBarrierStencil barrierStencilType(ContactCase type) {
  switch (type) {
    case ContactCase::PP:
      return ksk::engine::contact::EBarrierStencil::PP;
    case ContactCase::PE:
      return ksk::engine::contact::EBarrierStencil::PE;
    case ContactCase::PT:
      return ksk::engine::contact::EBarrierStencil::PT;
    case ContactCase::EE:
      return ksk::engine::contact::EBarrierStencil::EE;
  }
  return ksk::engine::contact::EBarrierStencil::PP;
}

Real estimateFrictionNormalForce(const ContactStencil& stencil,
                                 const std::array<glm::dvec3, 4>& x) {
  engine::contact::GIPCBarrierStencil barrier;
  barrier.type = barrierStencilType(stencil.type);
  barrier.dHat = stencil.dHat;
  barrier.reservedDist = stencil.thickness;
  barrier.kappa = stencil.stiffness;
  const int count = stencilPointCount(stencil.type);

  for (int i = 0; i < count; ++i) {
    barrier.x[i] = x[i];
    barrier.restX[i] = stencil.frictionReferenceX[i];
  }

  const engine::contact::LocalBarrierGradient gradient =
      ksk::engine::contact::computeGIPCBarrierGradient(barrier);
  glm::dvec3 relative_gradient(0.0);
  Real weight_norm_sqr = 0.0;
  for (int i = 0; i < count; ++i) {
    const Real weight = stencil.frictionWeights[static_cast<size_t>(i)];
    relative_gradient += weight * gradient.gradient[static_cast<size_t>(i)];
    weight_norm_sqr += weight * weight;
  }
  if (weight_norm_sqr <= 1.0e-18) {
    return 0.0;
  }
  return std::abs(glm::dot(relative_gradient, stencil.frictionNormal)) /
         weight_norm_sqr;
}

void initializeFriction(ContactStencil& stencil,
                        const GlobalGeometryManager& geometry,
                        const ContactDetectionConfig& config) {
  const Real coefficient =
      contactFrictionCoefficient(stencil, geometry, config);
  if (coefficient <= 0.0) {
    return;
  }

  const int count = stencilPointCount(stencil.type);
  std::array<glm::dvec3, 4> x{
      glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0)};
  for (int i = 0; i < count; ++i) {
    const int id = stencil.geometryIds[i];
    if (id < 0) {
      return;
    }
    x[i] = geometry.worldPosition(PointIdx{id});
    if (config.frictionReferencePositions != nullptr &&
        id < config.frictionReferencePositions->size()) {
      stencil.frictionReferenceX[i] = (*config.frictionReferencePositions)[id];
    } else {
      stencil.frictionReferenceX[i] = x[i];
    }
  }

  switch (stencil.type) {
    case ContactCase::PP:
      stencil.frictionWeights = {1.0, -1.0, 0.0, 0.0};
      break;
    case ContactCase::PE: {
      const Real t = pointEdgeProjectionParameter(x[0], x[1], x[2]);
      stencil.frictionWeights = {1.0, -(1.0 - t), -t, 0.0};
      break;
    }
    case ContactCase::PT: {
      const glm::dvec3 normal = safeNormalize(glm::cross(x[2] - x[1],
                                                        x[3] - x[1]),
                                             x[0] - x[1]);
      const glm::dvec3 projected = x[0] - normal * glm::dot(x[0] - x[1], normal);
      const auto bary = triangleBarycentric(projected, x[1], x[2], x[3]);
      stencil.frictionWeights = {1.0, -bary[0], -bary[1], -bary[2]};
      break;
    }
    case ContactCase::EE: {
      const auto params = closestSegmentParameters(x[0], x[1], x[2], x[3]);
      const Real s = params[0];
      const Real t = params[1];
      stencil.frictionWeights = {1.0 - s, s, -(1.0 - t), -t};
      break;
    }
  }

  glm::dvec3 relative(0.0);
  for (int i = 0; i < count; ++i) {
    relative += stencil.frictionWeights[i] * x[i];
  }
  glm::dvec3 fallback = relative;
  if (stencil.type == ContactCase::PT) {
    fallback = glm::cross(x[2] - x[1], x[3] - x[1]);
  } else if (stencil.type == ContactCase::EE) {
    fallback = glm::cross(x[1] - x[0], x[3] - x[2]);
  }
  stencil.frictionNormal = safeNormalize(relative, fallback);
  stencil.frictionCoefficient = coefficient;
  stencil.frictionEpsilon = std::max<Real>(config.frictionEpsilon, 1.0e-12);
  stencil.frictionNormalForce = estimateFrictionNormalForce(stencil, x);
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
                           const GlobalGeometryManager& geometry,
                           const ContactDetectionConfig& config,
                           Real reserved_dist = 0.0) {
  ContactStencil stencil{
      .type = type,
      .geometryIds = ids,
      .dHat = config.dHat,
      .stiffness = config.stiffness,
      .thickness = reserved_dist,
  };
  initializeFriction(stencil, geometry, config);
  return stencil;
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

void addCandidateHit(CandidateDetectionResult& result,
                     CandidateHit hit) {
  if (hit.toi) {
    result.stepSizeUpperBound =
        std::min(result.stepSizeUpperBound, std::clamp(*hit.toi, 0.0, 1.0));
  }
  result.candidates.push_back(std::move(hit.candidate));
}

void mergeCandidateDetectionResult(CandidateDetectionResult& into,
                                   CandidateDetectionResult&& from) {
  into.stepSizeUpperBound =
      std::min(into.stepSizeUpperBound, from.stepSizeUpperBound);
  into.candidates.insert(into.candidates.end(),
                         std::make_move_iterator(from.candidates.begin()),
                         std::make_move_iterator(from.candidates.end()));
}

void mergeThreadLocalCandidateResults(
    CandidateDetectionResult& into,
    tbb::enumerable_thread_specific<CandidateDetectionResult>& local) {
  for (CandidateDetectionResult& result : local) {
    mergeCandidateDetectionResult(into, std::move(result));
  }
}

void addStepSizeUpperBound(StepSizeUpperBoundResult& result,
                           std::optional<Real> toi) {
  ++result.testedPairs;
  if (!toi) {
    return;
  }
  ++result.collidingPairs;
  result.stepSizeUpperBound =
      std::min(result.stepSizeUpperBound, std::clamp(*toi, 0.0, 1.0));
}

void mergeStepSizeUpperBoundResult(StepSizeUpperBoundResult& into,
                                   const StepSizeUpperBoundResult& from) {
  into.stepSizeUpperBound =
      std::min(into.stepSizeUpperBound, from.stepSizeUpperBound);
  into.testedPairs += from.testedPairs;
  into.collidingPairs += from.collidingPairs;
}

void mergeThreadLocalStepSizeUpperBoundResults(
    StepSizeUpperBoundResult& into,
    tbb::enumerable_thread_specific<StepSizeUpperBoundResult>& local) {
  for (const StepSizeUpperBoundResult& result : local) {
    mergeStepSizeUpperBoundResult(into, result);
  }
}

void appendThreadLocalStencils(
    ContactStencils& into,
    tbb::enumerable_thread_specific<ContactStencils>& local) {
  size_t total = into.size();
  for (const ContactStencils& stencils : local) {
    total += stencils.size();
  }
  into.reserve(total);
  for (ContactStencils& stencils : local) {
    into.insert(into.end(), std::make_move_iterator(stencils.begin()),
                std::make_move_iterator(stencils.end()));
  }
}

Real pointRadius(const GlobalGeometryManager& geometry, PointIdx point) {
  return geometry.points.at(static_cast<size_t>(point)).radius;
}

Real triangleSurfaceOffset(const GlobalGeometryManager& geometry,
                           const std::array<PointIdx, 3>& triangle) {
  Real offset = 0.0;
  for (const GeometryTriangle& candidate : geometry.triangles) {
    const std::array points{candidate.p0, candidate.p1, candidate.p2};
    const bool same =
        (points[0] == triangle[0] || points[0] == triangle[1] ||
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
  Real offset =
      std::max({pointRadius(geometry, p0), pointRadius(geometry, p1), fallback});
  for (const GeometryEdge& edge : geometry.edges) {
    if ((edge.p0 == p0 && edge.p1 == p1) ||
        (edge.p0 == p1 && edge.p1 == p0)) {
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

std::optional<CandidateHit> makeSweptPointTriangleCandidate(
    PointIdx point, int triangle_id, const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config) {
  const auto triangle = geometry.globalTriangle(triangle_id);
  const glm::dvec3 p = geometry.worldPosition(point);
  const glm::dvec3 a = geometry.worldPosition(triangle[0]);
  const glm::dvec3 b = geometry.worldPosition(triangle[1]);
  const glm::dvec3 c = geometry.worldPosition(triangle[2]);
  const Real primitive_thickness =
      pointRadius(geometry, point) +
      geometry.triangles.at(static_cast<size_t>(triangle_id)).thickness;
  const PointTriangleDistanceType type =
      decidePointTriangleDistanceType(p, a, b, c);
  const Real distance_sqr =
      pointTriangleDistanceSqrByType(type, p, a, b, c);
  const std::optional<Real> toi =
      runACCD(false, {p, a, b, c},
              {direction[point], direction[triangle[0]], direction[triangle[1]],
               direction[triangle[2]]},
              config.toi, primitive_thickness);

  if (!shouldKeepByDistanceOrCCD(distance_sqr, toi, config,
                                 primitive_thickness)) {
    return std::nullopt;
  }

  return CandidateHit{
      .candidate = makeCandidate(EContactCandidate::PointTriangle,
                                 {point, triangle[0], triangle[1], triangle[2]},
                                 config, primitive_thickness),
      .toi = toi,
  };
}

std::optional<Real> computeSweptPointTriangleTOI(
    PointIdx point, int triangle_id, const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config) {
  const auto triangle = geometry.globalTriangle(triangle_id);
  const glm::dvec3 p = geometry.worldPosition(point);
  const glm::dvec3 a = geometry.worldPosition(triangle[0]);
  const glm::dvec3 b = geometry.worldPosition(triangle[1]);
  const glm::dvec3 c = geometry.worldPosition(triangle[2]);
  const Real primitive_thickness =
      pointRadius(geometry, point) +
      geometry.triangles.at(static_cast<size_t>(triangle_id)).thickness;
  return runACCD(false, {p, a, b, c},
                 {direction[point], direction[triangle[0]],
                  direction[triangle[1]], direction[triangle[2]]},
                 config.toi, primitive_thickness);
}

std::optional<CandidateHit> makeSweptEdgeEdgeCandidate(
    int first_id, int second_id, const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config) {
  const auto first = geometry.globalEdge(first_id);
  const auto second = geometry.globalEdge(second_id);
  const glm::dvec3 a0 = geometry.worldPosition(first[0]);
  const glm::dvec3 a1 = geometry.worldPosition(first[1]);
  const glm::dvec3 b0 = geometry.worldPosition(second[0]);
  const glm::dvec3 b1 = geometry.worldPosition(second[1]);
  const Real primitive_thickness =
      geometry.edges.at(static_cast<size_t>(first_id)).radius +
      geometry.edges.at(static_cast<size_t>(second_id)).radius;
  const EdgeEdgeDistanceType type = decideEdgeEdgeDistanceType(a0, a1, b0, b1);
  const Real distance_sqr =
      edgeEdgeDistanceSqrByType(type, a0, a1, b0, b1);
  const std::optional<Real> toi =
      runACCD(true, {a0, a1, b0, b1},
              {direction[first[0]], direction[first[1]], direction[second[0]],
               direction[second[1]]},
              config.toi, primitive_thickness);

  if (!shouldKeepByDistanceOrCCD(distance_sqr, toi, config,
                                 primitive_thickness)) {
    return std::nullopt;
  }

  return CandidateHit{
      .candidate = makeCandidate(EContactCandidate::EdgeEdge,
                                 {first[0], first[1], second[0], second[1]},
                                 config, primitive_thickness),
      .toi = toi,
  };
}

std::optional<Real> computeSweptEdgeEdgeTOI(
    int first_id, int second_id, const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config) {
  const auto first = geometry.globalEdge(first_id);
  const auto second = geometry.globalEdge(second_id);
  const glm::dvec3 a0 = geometry.worldPosition(first[0]);
  const glm::dvec3 a1 = geometry.worldPosition(first[1]);
  const glm::dvec3 b0 = geometry.worldPosition(second[0]);
  const glm::dvec3 b1 = geometry.worldPosition(second[1]);
  const Real primitive_thickness =
      geometry.edges.at(static_cast<size_t>(first_id)).radius +
      geometry.edges.at(static_cast<size_t>(second_id)).radius;
  return runACCD(true, {a0, a1, b0, b1},
                 {direction[first[0]], direction[first[1]],
                  direction[second[0]], direction[second[1]]},
                 config.toi, primitive_thickness);
}

ContactStencil createPointPointStencil(PointIdx first, PointIdx second,
                                       const GlobalGeometryManager& geometry,
                                       const ContactDetectionConfig& config) {
  return makeStencil(
      ContactCase::PP, {first, second, -1, -1}, geometry, config,
      pointRadius(geometry, first) + pointRadius(geometry, second));
}

ContactStencil createPointEdgeStencil(PointIdx point, PointIdx edge0,
                                      PointIdx edge1,
                                      const GlobalGeometryManager& geometry,
                                      const ContactDetectionConfig& config,
                                      Real fallback_edge_offset = 0.0) {
  return makeStencil(ContactCase::PE, {point, edge0, edge1, -1}, geometry, config,
                     pointEdgeSurfaceOffset(geometry, point, edge0, edge1,
                                             fallback_edge_offset));
}

std::optional<ContactStencil> createPointTriangleStencil(
    PointIdx point, const std::array<PointIdx, 3>& triangle,
    const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction, const ContactDetectionConfig& config,
    Real triangle_thickness, bool enable_ccd) {
  const glm::dvec3 p = geometry.worldPosition(point);
  const glm::dvec3 a = geometry.worldPosition(triangle[0]);
  const glm::dvec3 b = geometry.worldPosition(triangle[1]);
  const glm::dvec3 c = geometry.worldPosition(triangle[2]);
  const Real point_offset = pointRadius(geometry, point);
  const Real triangle_radius =
      std::max(triangle_thickness, triangleSurfaceOffset(geometry, triangle));
  const PointTriangleDistanceType type =
      decidePointTriangleDistanceType(p, a, b, c);
  const Real distance_sqr = pointTriangleDistanceSqrByType(type, p, a, b, c);
  std::optional<Real> toi;

  if (enable_ccd) {
    toi =
        runACCD(false, {p, a, b, c},
                {direction[point], direction[triangle[0]],
                 direction[triangle[1]], direction[triangle[2]]},
                config.toi, point_offset + triangle_radius);
  }

  if (!shouldKeepByDistanceOrCCD(distance_sqr, toi, config,
                                 point_offset + triangle_radius)) {
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
                                    config, triangle_radius);
    case PointTriangleDistanceType::P_BC:
      return createPointEdgeStencil(point, triangle[1], triangle[2], geometry,
                                    config, triangle_radius);
    case PointTriangleDistanceType::P_CA:
      return createPointEdgeStencil(point, triangle[2], triangle[0], geometry,
                                    config, triangle_radius);
    case PointTriangleDistanceType::P_ABC:
      return makeStencil(ContactCase::PT,
                         {point, triangle[0], triangle[1], triangle[2]},
                         geometry, config,
                         point_offset + triangle_radius);
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
                         {first[0], first[1], second[0], second[1]},
                         geometry, config,
                         first_offset + second_offset);
  }
  return makeStencil(ContactCase::EE, {-1, -1, -1, -1}, geometry, config);
}

ContactStencil createMollifiedEdgeEdgeStencil(
    EdgeEdgeDistanceType type, const std::array<PointIdx, 2>& first,
    const std::array<PointIdx, 2>& second,
    const GlobalGeometryManager& geometry,
    const ContactDetectionConfig& config, Real reserved_dist) {
  std::array<int, 4> ids{first[0], first[1], second[0], second[1]};
  ContactMollifierDistance distance = ContactMollifierDistance::EE;

  switch (type) {
    case EdgeEdgeDistanceType::A_C:
      ids = {first[0], first[1], second[0], second[1]};
      distance = ContactMollifierDistance::PP;
      break;
    case EdgeEdgeDistanceType::A_D:
      ids = {first[0], first[1], second[1], second[0]};
      distance = ContactMollifierDistance::PP;
      break;
    case EdgeEdgeDistanceType::B_C:
      ids = {first[1], first[0], second[0], second[1]};
      distance = ContactMollifierDistance::PP;
      break;
    case EdgeEdgeDistanceType::B_D:
      ids = {first[1], first[0], second[1], second[0]};
      distance = ContactMollifierDistance::PP;
      break;
    case EdgeEdgeDistanceType::AB_C:
      ids = {second[0], second[1], first[0], first[1]};
      distance = ContactMollifierDistance::PE;
      break;
    case EdgeEdgeDistanceType::AB_D:
      ids = {second[1], second[0], first[0], first[1]};
      distance = ContactMollifierDistance::PE;
      break;
    case EdgeEdgeDistanceType::A_CD:
      ids = {first[0], first[1], second[0], second[1]};
      distance = ContactMollifierDistance::PE;
      break;
    case EdgeEdgeDistanceType::B_CD:
      ids = {first[1], first[0], second[0], second[1]};
      distance = ContactMollifierDistance::PE;
      break;
    case EdgeEdgeDistanceType::AB_CD:
      ids = {first[0], first[1], second[0], second[1]};
      distance = ContactMollifierDistance::EE;
      break;
  }

  ContactStencil stencil =
      makeStencil(ContactCase::EE, ids, geometry, config, reserved_dist);
  stencil.mollifierDistance = distance;
  return stencil;
}

bool edgeEdgeUsesGIPCMollifier(const std::array<PointIdx, 2>& first,
                               const std::array<PointIdx, 2>& second,
                               const GlobalGeometryManager& geometry) {
  const glm::dvec3 ea0 = geometry.worldPosition(first[0]);
  const glm::dvec3 ea1 = geometry.worldPosition(first[1]);
  const glm::dvec3 eb0 = geometry.worldPosition(second[0]);
  const glm::dvec3 eb1 = geometry.worldPosition(second[1]);
  const glm::dvec3 rea0 = geometry.restPosition(first[0]);
  const glm::dvec3 rea1 = geometry.restPosition(first[1]);
  const glm::dvec3 reb0 = geometry.restPosition(second[0]);
  const glm::dvec3 reb1 = geometry.restPosition(second[1]);

  const glm::dvec3 current_cross = glm::cross(ea1 - ea0, eb1 - eb0);
  const Real current_cross_sqr = glm::dot(current_cross, current_cross);
  const Real rest_edge_a_sqr = glm::dot(rea1 - rea0, rea1 - rea0);
  const Real rest_edge_b_sqr = glm::dot(reb1 - reb0, reb1 - reb0);
  const Real eps_x = 1.0e-3 * rest_edge_a_sqr * rest_edge_b_sqr;
  return current_cross_sqr < eps_x;
}

std::optional<ContactStencil> createEdgeEdgeStencil(
    const std::array<PointIdx, 2>& first, const std::array<PointIdx, 2>& second,
    const GlobalGeometryManager& geometry,
    std::span<const glm::dvec3> direction, const ContactDetectionConfig& config,
    Real first_offset, Real second_offset, bool enable_ccd) {
  const glm::dvec3 a0 = geometry.worldPosition(first[0]);
  const glm::dvec3 a1 = geometry.worldPosition(first[1]);
  const glm::dvec3 b0 = geometry.worldPosition(second[0]);
  const glm::dvec3 b1 = geometry.worldPosition(second[1]);
  if (first_offset == 0.0) {
    first_offset = pointRadius(geometry, first[0]);
  }
  if (second_offset == 0.0) {
    second_offset = pointRadius(geometry, second[0]);
  }
  const Real reserved_dist = first_offset + second_offset;
  const EdgeEdgeDistanceType type = decideEdgeEdgeDistanceType(a0, a1, b0, b1);
  const Real distance_sqr = edgeEdgeDistanceSqrByType(type, a0, a1, b0, b1);
  std::optional<Real> toi;
  if (enable_ccd) {
    toi =
        runACCD(true, {a0, a1, b0, b1},
                {direction[first[0]], direction[first[1]], direction[second[0]],
                 direction[second[1]]},
                config.toi, reserved_dist);
  }
  if (!shouldKeepByDistanceOrCCD(distance_sqr, toi, config, reserved_dist)) {
    return std::nullopt;
  }

  if (edgeEdgeUsesGIPCMollifier(first, second, geometry)) {
    return createMollifiedEdgeEdgeStencil(type, first, second, geometry,
                                          config, reserved_dist);
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

[[nodiscard]] std::vector<PrimitiveRef> buildPointPrimitiveTopology(
    const GlobalGeometryManager& geometry) {
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
    primitives.push_back(PrimitiveRef{
        .id = point,
        .reserved_dist = geometry.points[static_cast<size_t>(point)].radius,
    });
  }
  return primitives;
}

[[nodiscard]] std::vector<PrimitiveRef> buildEdgePrimitiveTopology(
    const GlobalGeometryManager& geometry) {
  std::vector<PrimitiveRef> primitives;
  primitives.reserve(geometry.edgeCount());
  for (int edge = 0; edge < geometry.edgeCount(); ++edge) {
    primitives.push_back(PrimitiveRef{
        .id = edge,
        .reserved_dist = geometry.edges[static_cast<size_t>(edge)].radius,
    });
  }
  return primitives;
}

[[nodiscard]] std::vector<PrimitiveRef> buildTrianglePrimitiveTopology(
    const GlobalGeometryManager& geometry) {
  std::vector<PrimitiveRef> primitives;
  primitives.reserve(static_cast<size_t>(geometry.triangleCount()));
  for (int triangle = 0; triangle < geometry.triangleCount(); ++triangle) {
    primitives.push_back(PrimitiveRef{
        .id = triangle,
        .reserved_dist =
            geometry.triangles[static_cast<size_t>(triangle)].thickness,
    });
  }
  return primitives;
}

[[nodiscard]] ContactPrimitiveTopology buildPrimitiveTopology(
    const GlobalGeometryManager& geometry) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/BuildPrimitiveTopology");
  ContactPrimitiveTopology topology;
  topology.geometry = &geometry;
  topology.point_count = geometry.pointCount();
  topology.edge_count = geometry.edgeCount();
  topology.triangle_count = geometry.triangleCount();
  topology.points = buildPointPrimitiveTopology(geometry);
  topology.edges = buildEdgePrimitiveTopology(geometry);
  topology.triangles = buildTrianglePrimitiveTopology(geometry);

  topology.deformable_points.reserve(topology.points.size());
  for (size_t index = 0; index < topology.points.size(); ++index) {
    if (isDeformableBody(geometry, pointBody(geometry, topology.points[index].id))) {
      topology.deformable_points.push_back(index);
    }
  }
  topology.deformable_edges.reserve(topology.edges.size());
  topology.rigid_edges.reserve(topology.edges.size());
  for (size_t index = 0; index < topology.edges.size(); ++index) {
    const GeometryBodyIdx body = edgeBody(geometry, topology.edges[index].id);
    if (isDeformableBody(geometry, body)) {
      topology.deformable_edges.push_back(index);
    }
    if (isRigidBody(geometry, body)) {
      topology.rigid_edges.push_back(index);
    }
  }
  topology.deformable_triangles.reserve(topology.triangles.size());
  topology.rigid_triangles.reserve(topology.triangles.size());
  for (size_t index = 0; index < topology.triangles.size(); ++index) {
    const GeometryBodyIdx body =
        triangleBody(geometry, topology.triangles[index].id);
    if (isDeformableBody(geometry, body)) {
      topology.deformable_triangles.push_back(index);
    }
    if (isRigidBody(geometry, body)) {
      topology.rigid_triangles.push_back(index);
    }
  }
  return topology;
}

[[nodiscard]] const ContactPrimitiveTopology& cachedPrimitiveTopology(
    const GlobalGeometryManager& geometry) {
  thread_local ContactPrimitiveTopology topology;
  if (topology.geometry != &geometry ||
      topology.point_count != geometry.pointCount() ||
      topology.edge_count != geometry.edgeCount() ||
      topology.triangle_count != geometry.triangleCount()) {
    topology = buildPrimitiveTopology(geometry);
  }
  return topology;
}

void refreshPrimitiveBounds(const GlobalGeometryManager& geometry,
                            std::span<const glm::dvec3> direction, Real toi,
                            PrimitiveKind kind,
                            std::vector<PrimitiveRef>& primitives) {
  tbb::parallel_for(size_t{0}, primitives.size(), [&](size_t index) {
    PrimitiveRef& primitive = primitives[index];
    switch (kind) {
      case PrimitiveKind::Point:
        primitive.bounds =
            geometry.trajectoryPointBounds(PointIdx{primitive.id}, direction,
                                           toi)
                .dilate(primitive.reserved_dist);
        break;
      case PrimitiveKind::Edge:
        primitive.bounds =
            geometry
                .trajectoryEdgeBounds(primitive.id, direction, toi,
                                      primitive.reserved_dist)
                .dilate(primitive.reserved_dist);
        break;
      case PrimitiveKind::Triangle:
        primitive.bounds =
            geometry
                .trajectoryTriangleBounds(primitive.id, direction, toi,
                                          primitive.reserved_dist)
                .dilate(primitive.reserved_dist);
        break;
    }
  });
}

void refreshCurrentPrimitiveBounds(const GlobalGeometryManager& geometry,
                                   PrimitiveKind kind,
                                   std::vector<PrimitiveRef>& primitives) {
  tbb::parallel_for(size_t{0}, primitives.size(), [&](size_t index) {
    PrimitiveRef& primitive = primitives[index];
    switch (kind) {
      case PrimitiveKind::Point:
        primitive.bounds =
            geometry.pointBounds(PointIdx{primitive.id})
                .dilate(primitive.reserved_dist);
        break;
      case PrimitiveKind::Edge:
        primitive.bounds =
            geometry.edgeBounds(primitive.id).dilate(primitive.reserved_dist);
        break;
      case PrimitiveKind::Triangle:
        primitive.bounds =
            geometry.triangleBounds(primitive.id)
                .dilate(primitive.reserved_dist);
        break;
    }
  });
}

[[nodiscard]] std::vector<PrimitiveRef> selectPrimitiveRefs(
    const std::vector<PrimitiveRef>& source, const std::vector<size_t>& indices) {
  std::vector<PrimitiveRef> selected(indices.size());
  tbb::parallel_for(size_t{0}, indices.size(), [&](size_t index) {
    selected[index] = source[indices[index]];
  });
  return selected;
}

[[nodiscard]] ContactPrimitiveRefs buildContactPrimitiveRefs(
    const GlobalGeometryManager& geometry, const GeometryBuffer& direction,
    Real toi) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/RefreshPrimitiveBounds");
  const ContactPrimitiveTopology& topology = cachedPrimitiveTopology(geometry);
  ContactPrimitiveRefs refs;
  refs.points = topology.points;
  refs.edges = topology.edges;
  refs.triangles = topology.triangles;

  const std::span<const glm::dvec3> direction_view = direction.cpu();
  refreshPrimitiveBounds(geometry, direction_view, toi, PrimitiveKind::Point,
                         refs.points);
  refreshPrimitiveBounds(geometry, direction_view, toi, PrimitiveKind::Edge,
                         refs.edges);
  refreshPrimitiveBounds(geometry, direction_view, toi, PrimitiveKind::Triangle,
                         refs.triangles);

  refs.deformable_points =
      selectPrimitiveRefs(refs.points, topology.deformable_points);
  refs.deformable_edges =
      selectPrimitiveRefs(refs.edges, topology.deformable_edges);
  refs.deformable_triangles =
      selectPrimitiveRefs(refs.triangles, topology.deformable_triangles);
  refs.rigid_edges = selectPrimitiveRefs(refs.edges, topology.rigid_edges);
  refs.rigid_triangles =
      selectPrimitiveRefs(refs.triangles, topology.rigid_triangles);
  return refs;
}

[[nodiscard]] ContactPrimitiveRefs buildCurrentContactPrimitiveRefs(
    const GlobalGeometryManager& geometry) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/RefreshCurrentPrimitiveBounds");
  const ContactPrimitiveTopology& topology = cachedPrimitiveTopology(geometry);
  ContactPrimitiveRefs refs;
  refs.points = topology.points;
  refs.edges = topology.edges;
  refs.triangles = topology.triangles;

  refreshCurrentPrimitiveBounds(geometry, PrimitiveKind::Point, refs.points);
  refreshCurrentPrimitiveBounds(geometry, PrimitiveKind::Edge, refs.edges);
  refreshCurrentPrimitiveBounds(geometry, PrimitiveKind::Triangle,
                                refs.triangles);

  refs.deformable_points =
      selectPrimitiveRefs(refs.points, topology.deformable_points);
  refs.deformable_edges =
      selectPrimitiveRefs(refs.edges, topology.deformable_edges);
  refs.deformable_triangles =
      selectPrimitiveRefs(refs.triangles, topology.deformable_triangles);
  refs.rigid_edges = selectPrimitiveRefs(refs.edges, topology.rigid_edges);
  refs.rigid_triangles =
      selectPrimitiveRefs(refs.triangles, topology.rigid_triangles);
  return refs;
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

size_t queryPointRefsAgainstRigidTrianglesForActiveStencils(
    const GlobalGeometryManager& geometry, const std::vector<PrimitiveRef>& points,
    const std::vector<PrimitiveRef>& rigid_triangles,
    const spatify::LBVH<Real>& rigid_triangle_tree,
    const RigidInvolvedPairRefs& refs, const ContactWorkList& work,
    Real query_dilation, const ContactDetectionConfig& config,
    ContactStencils& stencils) {
  if (rigid_triangles.empty()) {
    return 0;
  }
  tbb::enumerable_thread_specific<ContactStencils> local_stencils;
  tbb::enumerable_thread_specific<size_t> local_leaf_pairs(size_t{0});

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
    ContactStencils& local = local_stencils.local();
    size_t& leaf_pairs = local_leaf_pairs.local();
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

          ++leaf_pairs;
          const Real triangle_thickness =
              geometry.triangles[static_cast<size_t>(triangle.id)].thickness;
          std::optional<ContactStencil> stencil =
              createPointTriangleStencil(
                  PointIdx{point.id}, geometry.globalTriangle(triangle.id),
                  geometry, std::span<const glm::dvec3>{}, config,
                  triangle_thickness, false);
          if (stencil) {
            local.push_back(std::move(*stencil));
          }
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
        });
  });

  appendThreadLocalStencils(stencils, local_stencils);
  size_t leaf_pairs = 0;
  for (const size_t count : local_leaf_pairs) {
    leaf_pairs += count;
  }
  return leaf_pairs;
}

size_t queryEdgeRefsAgainstRigidEdgesForActiveStencils(
    const GlobalGeometryManager& geometry, const std::vector<PrimitiveRef>& edges,
    const std::vector<PrimitiveRef>& rigid_edges,
    const spatify::LBVH<Real>& rigid_edge_tree,
    const RigidInvolvedPairRefs& refs, const ContactWorkList& work,
    Real query_dilation, const ContactDetectionConfig& config,
    ContactStencils& stencils) {
  if (rigid_edges.empty()) {
    return 0;
  }
  tbb::enumerable_thread_specific<ContactStencils> local_stencils;
  tbb::enumerable_thread_specific<size_t> local_leaf_pairs(size_t{0});

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
    ContactStencils& local = local_stencils.local();
    size_t& leaf_pairs = local_leaf_pairs.local();
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

          ++leaf_pairs;
          const int first_edge = std::min(query_edge.id, candidate.id);
          const int second_edge = std::max(query_edge.id, candidate.id);
          const Real first_offset = query_edge.reserved_dist;
          const Real second_offset = candidate.reserved_dist;
          std::optional<ContactStencil> stencil =
              createEdgeEdgeStencil(
                  geometry.globalEdge(first_edge),
                  geometry.globalEdge(second_edge), geometry,
                  std::span<const glm::dvec3>{}, config, first_offset,
                  second_offset, false);
          if (stencil) {
            local.push_back(std::move(*stencil));
          }
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
        });
  });

  appendThreadLocalStencils(stencils, local_stencils);
  size_t leaf_pairs = 0;
  for (const size_t count : local_leaf_pairs) {
    leaf_pairs += count;
  }
  return leaf_pairs;
}

void gatherGeometryBodyBounds(const GlobalGeometryManager& geometry,
                              const std::vector<PrimitiveRef>& points,
                              const std::vector<PrimitiveRef>& edges,
                              const std::vector<PrimitiveRef>& triangles,
                              ContactWorkList& work) {
  // TODO: slow when we have many bodies
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

size_t gatherGlobalDeformablePointTriangleStencils(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& deformable_points,
    const std::vector<PrimitiveRef>& deformable_triangles,
    Real query_dilation, const ContactDetectionConfig& config,
    ContactStencils& stencils) {
  if (deformable_triangles.empty())
    return 0;

  std::vector<PrimitiveRef> tree_triangles = deformable_triangles;
  const spatify::LBVH<Real> triangle_tree = buildLBVH(tree_triangles);
  tbb::enumerable_thread_specific<ContactStencils> local_stencils;
  tbb::enumerable_thread_specific<size_t> local_leaf_pairs(size_t{0});

  tbb::parallel_for(size_t{0}, deformable_points.size(), [&](size_t index) {
    const PrimitiveRef& point = deformable_points[index];
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(point.bounds.dilate(query_dilation));
    ContactStencils& local = local_stencils.local();
    size_t& leaf_pairs = local_leaf_pairs.local();
    triangle_tree.runSpatialQuery(
        [&](int primitive_index) {
          const PrimitiveRef& triangle = tree_triangles[primitive_index];

          if (geometry.triangleContainsPoint(triangle.id,
                                             PointIdx{point.id}))
            return false;

          if (!pointTriangleCollisionEnabled(geometry, PointIdx{point.id},
                                             triangle.id))
            return false;

          ++leaf_pairs;
          const Real triangle_thickness =
              geometry.triangles[static_cast<size_t>(triangle.id)].thickness;
          std::optional<ContactStencil> stencil =
              createPointTriangleStencil(
                  PointIdx{point.id}, geometry.globalTriangle(triangle.id),
                  geometry, std::span<const glm::dvec3>{}, config,
                  triangle_thickness, false);
          if (stencil) {
            local.push_back(std::move(*stencil));
          }
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        });
  });

  appendThreadLocalStencils(stencils, local_stencils);
  size_t leaf_pairs = 0;
  for (const size_t count : local_leaf_pairs) {
    leaf_pairs += count;
  }
  return leaf_pairs;
}

size_t gatherGlobalDeformableEdgeEdgeStencils(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& deformable_edges,
    Real query_dilation, const ContactDetectionConfig& config,
    ContactStencils& stencils) {
  if (deformable_edges.empty())
    return 0;

  std::vector<PrimitiveRef> tree_edges = deformable_edges;
  const spatify::LBVH<Real> edge_tree = buildIndexedLBVH(tree_edges);
  assert(edge_tree.hasSubtreeMaxPrimitiveIndex());
  tbb::enumerable_thread_specific<ContactStencils> local_stencils;
  tbb::enumerable_thread_specific<size_t> local_leaf_pairs(size_t{0});

  tbb::parallel_for(size_t{0}, deformable_edges.size(), [&](size_t index) {
    const PrimitiveRef& query_edge = deformable_edges[index];
    const GeometryBounds query_bounds =
        ensureFiniteBVHBounds(query_edge.bounds.dilate(query_dilation));
    ContactStencils& local = local_stencils.local();
    size_t& leaf_pairs = local_leaf_pairs.local();
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

          ++leaf_pairs;
          const Real first_offset = query_edge.reserved_dist;
          const Real second_offset = candidate.reserved_dist;
          std::optional<ContactStencil> stencil =
              createEdgeEdgeStencil(
                  geometry.globalEdge(query_edge.id),
                  geometry.globalEdge(candidate.id), geometry,
                  std::span<const glm::dvec3>{}, config, first_offset,
                  second_offset, false);
          if (stencil) {
            local.push_back(std::move(*stencil));
          }
          return true;
        },
        [&](const GeometryBounds& bounds) {
          return query_bounds.overlap(bounds);
        },
        [&](int node_index) {
          return edge_tree.subtreeMaxPrimitiveIndex(node_index) >
                 query_edge.id;
        });
  });

  appendThreadLocalStencils(stencils, local_stencils);
  size_t leaf_pairs = 0;
  for (const size_t count : local_leaf_pairs) {
    leaf_pairs += count;
  }
  return leaf_pairs;
}

CandidateDetectionResult detectDeformableDeformableCandidatesAtBVHLeaves(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& deformable_points,
    const std::vector<PrimitiveRef>& deformable_edges,
    const std::vector<PrimitiveRef>& deformable_triangles,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/DetectDeformableDeformableLeaves");
  CandidateDetectionResult result;
  const Real query_dilation = config.dHat;

  if (!deformable_triangles.empty()) {
    std::vector<PrimitiveRef> tree_triangles = deformable_triangles;
    const spatify::LBVH<Real> triangle_tree = buildLBVH(tree_triangles);
    tbb::enumerable_thread_specific<CandidateDetectionResult> local;

    tbb::parallel_for(
        size_t{0}, deformable_points.size(), [&](size_t index) {
          const PrimitiveRef& point = deformable_points[index];
          const GeometryBounds query_bounds = point.bounds.dilate(query_dilation);

          CandidateDetectionResult& local_result = local.local();

          triangle_tree.runSpatialQuery(
              [&](int primitive_index) {
                const PrimitiveRef& triangle = tree_triangles[primitive_index];
                if (geometry.triangleContainsPoint(triangle.id,
                                                   PointIdx{point.id}))
                  return false;

                if (!pointTriangleCollisionEnabled(
                        geometry, PointIdx{point.id}, triangle.id))
                  return false;

                std::optional<CandidateHit> hit =
                    makeSweptPointTriangleCandidate(
                        PointIdx{point.id}, triangle.id, geometry, direction,
                        config);

                if (hit)
                  addCandidateHit(local_result, std::move(*hit));

                return true;
              },
              [&](const GeometryBounds& bounds) {
                return query_bounds.overlap(bounds);
              });
        });

    mergeThreadLocalCandidateResults(result, local);
  }

  if (!deformable_edges.empty()) {
    std::vector<PrimitiveRef> tree_edges = deformable_edges;
    const spatify::LBVH<Real> edge_tree = buildIndexedLBVH(tree_edges);
    assert(edge_tree.hasSubtreeMaxPrimitiveIndex());
    tbb::enumerable_thread_specific<CandidateDetectionResult> local;

    tbb::parallel_for(
        size_t{0}, deformable_edges.size(), [&](size_t index) {
          const PrimitiveRef& query_edge = deformable_edges[index];
          const GeometryBounds query_bounds =
              ensureFiniteBVHBounds(query_edge.bounds.dilate(query_dilation));
          CandidateDetectionResult& local_result = local.local();

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

                std::optional<CandidateHit> hit =
                    makeSweptEdgeEdgeCandidate(query_edge.id, candidate.id,
                                               geometry, direction, config);
                if (hit)
                  addCandidateHit(local_result, std::move(*hit));
                return true;
              },
              [&](const GeometryBounds& bounds) {
                return query_bounds.overlap(bounds);
              },
              [&](int node_index) {
                return edge_tree.subtreeMaxPrimitiveIndex(node_index) >
                       query_edge.id;
              });
        });

    mergeThreadLocalCandidateResults(result, local);
  }

  KSK_PROFILE_VALUE("contact.cpu.deformable_leaf_candidates",
                    static_cast<double>(result.candidates.size()));
  return result;
}

void gatherRigidInvolvedBodyPairs(const GlobalGeometryManager& geometry,
                                  Real query_dilation,
                                  ContactWorkList& work) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/GatherRigidInvolvedBodyPairs");
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
  KSK_PROFILE_VALUE("contact.cpu.rigid_body_pairs",
                    static_cast<double>(work.bodyPairs.size()));
}

CandidateDetectionResult detectRigidInvolvedCandidatesAtBVHLeaves(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& points,
    const std::vector<PrimitiveRef>& edges,
    const std::vector<PrimitiveRef>& rigid_edges,
    const std::vector<PrimitiveRef>& rigid_triangles,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config,
    ContactWorkList& work) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/DetectRigidInvolvedLeaves");
  CandidateDetectionResult result;
  const Real query_dilation = config.dHat;

  gatherRigidInvolvedBodyPairs(geometry, query_dilation, work);
  if (work.bodyPairs.empty()) {
    return result;
  }

  std::vector<PrimitiveRef> tree_rigid_triangles = rigid_triangles;
  std::vector<PrimitiveRef> tree_rigid_edges = rigid_edges;
  const spatify::LBVH<Real> rigid_triangle_tree =
      buildLBVH(tree_rigid_triangles);
  const spatify::LBVH<Real> rigid_edge_tree = buildLBVH(tree_rigid_edges);

  const RigidInvolvedPairRefs rigid_pair_refs =
      buildRigidInvolvedPairRefs(geometry, points, edges, work);

  if (!rigid_triangles.empty()) {
    tbb::enumerable_thread_specific<CandidateDetectionResult> local;
    tbb::parallel_for(
        size_t{0}, rigid_pair_refs.points.size(), [&](size_t index) {
          const BodyPairPrimitiveRef& ref = rigid_pair_refs.points[index];
          const ContactWorkList::BodyPairWork& pair =
              work.bodyPairs.at(static_cast<size_t>(ref.pair));
          const GeometryBodyIdx target_body = oppositeBody(pair, ref.side);
          if (!isRigidBody(geometry, target_body)) {
            return;
          }
          const PrimitiveRef& point =
              points.at(static_cast<size_t>(ref.primitive));
          const GeometryBounds query_bounds =
              ensureFiniteBVHBounds(point.bounds.dilate(query_dilation));
          CandidateDetectionResult& local_result = local.local();
          rigid_triangle_tree.runSpatialQuery(
              [&](int primitive_index) {
                const PrimitiveRef& triangle =
                    tree_rigid_triangles[static_cast<size_t>(primitive_index)];
                if (triangleBody(geometry, triangle.id) != target_body)
                  return false;

                if (!triangle.bounds.overlap(pair.iAABB))
                  return false;

                if (geometry.triangleContainsPoint(triangle.id,
                                                   PointIdx{point.id}))
                  return false;

                if (!pointTriangleCollisionEnabled(
                        geometry, PointIdx{point.id}, triangle.id))
                  return false;

                std::optional<CandidateHit> hit =
                    makeSweptPointTriangleCandidate(
                        PointIdx{point.id}, triangle.id, geometry, direction,
                        config);
                if (hit) {
                  addCandidateHit(local_result, std::move(*hit));
                }
                return true;
              },
              [&](const GeometryBounds& bounds) {
                return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
              });
        });
    mergeThreadLocalCandidateResults(result, local);
  }

  if (!rigid_edges.empty()) {
    tbb::enumerable_thread_specific<CandidateDetectionResult> local;
    tbb::parallel_for(
        size_t{0}, rigid_pair_refs.edges.size(), [&](size_t index) {
          const BodyPairPrimitiveRef& ref = rigid_pair_refs.edges[index];
          const ContactWorkList::BodyPairWork& pair =
              work.bodyPairs.at(static_cast<size_t>(ref.pair));
          const GeometryBodyIdx target_body = oppositeBody(pair, ref.side);
          if (!isRigidBody(geometry, target_body)) {
            return;
          }
          const PrimitiveRef& query_edge =
              edges.at(static_cast<size_t>(ref.primitive));
          const GeometryBounds query_bounds =
              ensureFiniteBVHBounds(query_edge.bounds.dilate(query_dilation));
          CandidateDetectionResult& local_result = local.local();
          rigid_edge_tree.runSpatialQuery(
              [&](int primitive_index) {
                const PrimitiveRef& candidate =
                    tree_rigid_edges.at(static_cast<size_t>(primitive_index));
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

                const int first_edge =
                    std::min(query_edge.id, candidate.id);
                const int second_edge =
                    std::max(query_edge.id, candidate.id);
                std::optional<CandidateHit> hit =
                    makeSweptEdgeEdgeCandidate(first_edge, second_edge,
                                               geometry, direction, config);
                if (hit) {
                  addCandidateHit(local_result, std::move(*hit));
                }
                return true;
              },
              [&](const GeometryBounds& bounds) {
                return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
              });
        });
    mergeThreadLocalCandidateResults(result, local);
  }

  KSK_PROFILE_VALUE("contact.cpu.rigid_leaf_candidates",
                    static_cast<double>(result.candidates.size()));
  return result;
}

StepSizeUpperBoundResult computeDeformableDeformableStepSizeUpperBoundAtBVHLeaves(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& deformable_points,
    const std::vector<PrimitiveRef>& deformable_edges,
    const std::vector<PrimitiveRef>& deformable_triangles,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/StepBoundDeformableDeformableLeaves");
  StepSizeUpperBoundResult result;
  const Real query_dilation = config.dHat;

  if (!deformable_triangles.empty()) {
    std::vector<PrimitiveRef> tree_triangles = deformable_triangles;
    const spatify::LBVH<Real> triangle_tree = buildLBVH(tree_triangles);
    tbb::enumerable_thread_specific<StepSizeUpperBoundResult> local;

    tbb::parallel_for(size_t{0}, deformable_points.size(), [&](size_t index) {
      const PrimitiveRef& point = deformable_points[index];
      const GeometryBounds query_bounds = point.bounds.dilate(query_dilation);
      StepSizeUpperBoundResult& local_result = local.local();
      triangle_tree.runSpatialQuery(
          [&](int primitive_index) {
            const PrimitiveRef& triangle = tree_triangles[primitive_index];
            if (geometry.triangleContainsPoint(triangle.id,
                                               PointIdx{point.id}))
              return false;

            if (!pointTriangleCollisionEnabled(geometry, PointIdx{point.id},
                                               triangle.id))
              return false;

            addStepSizeUpperBound(
                local_result,
                computeSweptPointTriangleTOI(PointIdx{point.id}, triangle.id,
                                             geometry, direction, config));
            return true;
          },
          [&](const GeometryBounds& bounds) {
            return query_bounds.overlap(bounds);
          });
    });

    mergeThreadLocalStepSizeUpperBoundResults(result, local);
  }

  if (!deformable_edges.empty()) {
    std::vector<PrimitiveRef> tree_edges = deformable_edges;
    const spatify::LBVH<Real> edge_tree = buildIndexedLBVH(tree_edges);
    assert(edge_tree.hasSubtreeMaxPrimitiveIndex());
    tbb::enumerable_thread_specific<StepSizeUpperBoundResult> local;

    tbb::parallel_for(size_t{0}, deformable_edges.size(), [&](size_t index) {
      const PrimitiveRef& query_edge = deformable_edges[index];
      const GeometryBounds query_bounds =
          ensureFiniteBVHBounds(query_edge.bounds.dilate(query_dilation));
      StepSizeUpperBoundResult& local_result = local.local();
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

            addStepSizeUpperBound(
                local_result,
                computeSweptEdgeEdgeTOI(query_edge.id, candidate.id, geometry,
                                        direction, config));
            return true;
          },
          [&](const GeometryBounds& bounds) {
            return query_bounds.overlap(bounds);
          },
          [&](int node_index) {
            return edge_tree.subtreeMaxPrimitiveIndex(node_index) >
                   query_edge.id;
          });
    });

    mergeThreadLocalStepSizeUpperBoundResults(result, local);
  }

  KSK_PROFILE_VALUE("contact.cpu.step_bound_deformable_pairs",
                    static_cast<double>(result.testedPairs));
  return result;
}

StepSizeUpperBoundResult computeRigidInvolvedStepSizeUpperBoundAtBVHLeaves(
    const GlobalGeometryManager& geometry,
    const std::vector<PrimitiveRef>& points,
    const std::vector<PrimitiveRef>& edges,
    const std::vector<PrimitiveRef>& rigid_edges,
    const std::vector<PrimitiveRef>& rigid_triangles,
    std::span<const glm::dvec3> direction,
    const ContactDetectionConfig& config,
    ContactWorkList& work) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/StepBoundRigidInvolvedLeaves");
  StepSizeUpperBoundResult result;
  const Real query_dilation = config.dHat;

  gatherRigidInvolvedBodyPairs(geometry, query_dilation, work);
  if (work.bodyPairs.empty()) {
    return result;
  }

  std::vector<PrimitiveRef> tree_rigid_triangles = rigid_triangles;
  std::vector<PrimitiveRef> tree_rigid_edges = rigid_edges;
  const spatify::LBVH<Real> rigid_triangle_tree =
      buildLBVH(tree_rigid_triangles);
  const spatify::LBVH<Real> rigid_edge_tree = buildLBVH(tree_rigid_edges);

  const RigidInvolvedPairRefs rigid_pair_refs =
      buildRigidInvolvedPairRefs(geometry, points, edges, work);

  if (!rigid_triangles.empty()) {
    tbb::enumerable_thread_specific<StepSizeUpperBoundResult> local;
    tbb::parallel_for(size_t{0}, rigid_pair_refs.points.size(), [&](size_t index) {
      const BodyPairPrimitiveRef& ref = rigid_pair_refs.points[index];
      const ContactWorkList::BodyPairWork& pair =
          work.bodyPairs.at(static_cast<size_t>(ref.pair));
      const GeometryBodyIdx target_body = oppositeBody(pair, ref.side);
      if (!isRigidBody(geometry, target_body)) {
        return;
      }
      const PrimitiveRef& point = points.at(static_cast<size_t>(ref.primitive));
      const GeometryBounds query_bounds =
          ensureFiniteBVHBounds(point.bounds.dilate(query_dilation));
      StepSizeUpperBoundResult& local_result = local.local();
      rigid_triangle_tree.runSpatialQuery(
          [&](int primitive_index) {
            const PrimitiveRef& triangle =
                tree_rigid_triangles[static_cast<size_t>(primitive_index)];
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

            addStepSizeUpperBound(
                local_result,
                computeSweptPointTriangleTOI(PointIdx{point.id}, triangle.id,
                                             geometry, direction, config));
            return true;
          },
          [&](const GeometryBounds& bounds) {
            return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
          });
    });
    mergeThreadLocalStepSizeUpperBoundResults(result, local);
  }

  if (!rigid_edges.empty()) {
    tbb::enumerable_thread_specific<StepSizeUpperBoundResult> local;
    tbb::parallel_for(size_t{0}, rigid_pair_refs.edges.size(), [&](size_t index) {
      const BodyPairPrimitiveRef& ref = rigid_pair_refs.edges[index];
      const ContactWorkList::BodyPairWork& pair =
          work.bodyPairs.at(static_cast<size_t>(ref.pair));
      const GeometryBodyIdx target_body = oppositeBody(pair, ref.side);
      if (!isRigidBody(geometry, target_body)) {
        return;
      }
      const PrimitiveRef& query_edge =
          edges.at(static_cast<size_t>(ref.primitive));
      const GeometryBounds query_bounds =
          ensureFiniteBVHBounds(query_edge.bounds.dilate(query_dilation));
      StepSizeUpperBoundResult& local_result = local.local();
      rigid_edge_tree.runSpatialQuery(
          [&](int primitive_index) {
            const PrimitiveRef& candidate =
                tree_rigid_edges.at(static_cast<size_t>(primitive_index));
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

            const int first_edge = std::min(query_edge.id, candidate.id);
            const int second_edge = std::max(query_edge.id, candidate.id);
            addStepSizeUpperBound(
                local_result,
                computeSweptEdgeEdgeTOI(first_edge, second_edge, geometry,
                                        direction, config));
            return true;
          },
          [&](const GeometryBounds& bounds) {
            return query_bounds.overlap(bounds) && bounds.overlap(pair.iAABB);
          });
    });
    mergeThreadLocalStepSizeUpperBoundResults(result, local);
  }

  KSK_PROFILE_VALUE("contact.cpu.step_bound_rigid_pairs",
                    static_cast<double>(result.testedPairs));
  return result;
}

CurrentActiveContactWork gatherCurrentActiveContactWork(
    const GlobalGeometryManager& geometry, const ContactPrimitiveRefs& primitives,
    const ContactDetectionConfig& config) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/GatherCurrentActiveContactWork");
  CurrentActiveContactWork active_work;
  ContactWorkList& work = active_work.broadPhase;
  const Real query_dilation = config.dHat;

  {
    KSK_PROFILE_SCOPE("ContactDetectionCPU/CurrentActive/BodyBounds");
    gatherGeometryBodyBounds(geometry, primitives.points, primitives.edges,
                             primitives.triangles, work);
  }

  active_work.pointTriangleLeafPairs +=
      gatherGlobalDeformablePointTriangleStencils(
          geometry, primitives.deformable_points,
          primitives.deformable_triangles, query_dilation, config,
          active_work.stencils);

  active_work.edgeEdgeLeafPairs +=
      gatherGlobalDeformableEdgeEdgeStencils(
          geometry, primitives.deformable_edges, query_dilation, config,
          active_work.stencils);

  gatherRigidInvolvedBodyPairs(geometry, query_dilation, work);
  if (!work.bodyPairs.empty()) {
    std::vector<PrimitiveRef> tree_rigid_triangles =
        primitives.rigid_triangles;
    std::vector<PrimitiveRef> tree_rigid_edges = primitives.rigid_edges;
    const spatify::LBVH<Real> rigid_triangle_tree =
        buildLBVH(tree_rigid_triangles);
    const spatify::LBVH<Real> rigid_edge_tree = buildLBVH(tree_rigid_edges);

    const RigidInvolvedPairRefs rigid_pair_refs =
        buildRigidInvolvedPairRefs(geometry, primitives.points,
                                   primitives.edges, work);
    active_work.pointTriangleLeafPairs +=
        queryPointRefsAgainstRigidTrianglesForActiveStencils(
            geometry, primitives.points, tree_rigid_triangles,
            rigid_triangle_tree, rigid_pair_refs, work, query_dilation, config,
            active_work.stencils);
    active_work.edgeEdgeLeafPairs +=
        queryEdgeRefsAgainstRigidEdgesForActiveStencils(
            geometry, primitives.edges, tree_rigid_edges, rigid_edge_tree,
            rigid_pair_refs, work, query_dilation, config,
            active_work.stencils);
  }
  return active_work;
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
    auto subsystem_idx = info.subsystems[0];
    auto& subsystem_contacts =
        routed.subsystemInternalContacts[subsystem_idx];
    subsystem_contacts.subsystem = subsystem_idx;
    subsystem_contacts.contacts.push_back(stencil);
    return;
  }

  routed.globalContacts.push_back(stencil);
}

GlobalContactRouter routeActiveStencils(const GlobalGeometryManager& geometry,
                                        const ContactStencils& active_stencils) {
  GlobalContactRouter routed;
  for (const ContactStencil& active : active_stencils) {
    routeStencil(routed, geometry, active);
  }
  return routed;
}

}  // namespace

namespace detail {
namespace {
CandidateDetectionResult detectCandidatesAtBVHLeavesOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/DetectCandidatesAtBVHLeaves");
  CandidateDetectionResult result;

#ifndef NDEBUG
  validateCPUCCDInput(geometry, geometryDirection);
#endif

  const ContactPrimitiveRefs primitives =
      buildContactPrimitiveRefs(geometry, geometryDirection, config.toi);

  ContactWorkList work;
  {
    KSK_PROFILE_SCOPE("ContactDetectionCPU/DetectLeaves/BodyBounds");
    gatherGeometryBodyBounds(geometry, primitives.points, primitives.edges,
                             primitives.triangles, work);
  }

  mergeCandidateDetectionResult(
      result,
      detectDeformableDeformableCandidatesAtBVHLeaves(
          geometry, primitives.deformable_points, primitives.deformable_edges,
          primitives.deformable_triangles, geometryDirection.cpu(), config));

  mergeCandidateDetectionResult(
      result,
      detectRigidInvolvedCandidatesAtBVHLeaves(
          geometry, primitives.points, primitives.edges, primitives.rigid_edges,
          primitives.rigid_triangles, geometryDirection.cpu(), config, work));

  KSK_PROFILE_VALUE("contact.cpu.body_bounds",
                    static_cast<double>(work.bodies.size()));
  KSK_PROFILE_VALUE("contact.cpu.candidates",
                    static_cast<double>(result.candidates.size()));
  KSK_PROFILE_VALUE("contact.cpu.step_bound", result.stepSizeUpperBound);
  return result;
}

StepSizeUpperBoundResult computeStepSizeUpperBoundAtBVHLeavesOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/ComputeStepSizeUpperBoundAtBVHLeaves");
  StepSizeUpperBoundResult result;

#ifndef NDEBUG
  validateCPUCCDInput(geometry, geometryDirection);
#endif

  const ContactPrimitiveRefs primitives =
      buildContactPrimitiveRefs(geometry, geometryDirection, config.toi);

  ContactWorkList work;
  {
    KSK_PROFILE_SCOPE("ContactDetectionCPU/StepBound/BodyBounds");
    gatherGeometryBodyBounds(geometry, primitives.points, primitives.edges,
                             primitives.triangles, work);
  }

  mergeStepSizeUpperBoundResult(
      result,
      computeDeformableDeformableStepSizeUpperBoundAtBVHLeaves(
          geometry, primitives.deformable_points, primitives.deformable_edges,
          primitives.deformable_triangles, geometryDirection.cpu(), config));

  mergeStepSizeUpperBoundResult(
      result,
      computeRigidInvolvedStepSizeUpperBoundAtBVHLeaves(
          geometry, primitives.points, primitives.edges, primitives.rigid_edges,
          primitives.rigid_triangles, geometryDirection.cpu(), config, work));

  return result;
}

}  // namespace

ContactCandidates detectContactCandidatesAlongDirectionOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/DetectCandidates");

  if (config.storage == ContactDetectionStorage::Device) {
    throw std::runtime_error(
        "CPU contact detection cannot produce device contact tables");
  }

  return detectCandidatesAtBVHLeavesOnCPU(geometry, geometryDirection, config)
      .candidates;
}

Real computeStepSizeUpperBoundAlongDirectionOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/ComputeStepSizeUpperBound");

  if (config.storage == ContactDetectionStorage::Device) {
    throw std::runtime_error(
        "CPU contact detection cannot produce device contact tables");
  }

  return computeStepSizeUpperBoundAtBVHLeavesOnCPU(
             geometry, geometryDirection, config)
      .stepSizeUpperBound;
}

GlobalContactRouter rebuildActiveContactsOnCPU(
    const GlobalGeometryManager& geometry,
    const ContactDetectionConfig& config) {
  KSK_PROFILE_SCOPE("ContactDetectionCPU/RebuildActiveContacts");

  if (config.storage == ContactDetectionStorage::Device) {
    throw std::runtime_error(
        "CPU contact detection cannot produce device contact tables");
  }

  const ContactPrimitiveRefs primitives =
      buildCurrentContactPrimitiveRefs(geometry);
  CurrentActiveContactWork active_work =
      gatherCurrentActiveContactWork(geometry, primitives, config);
  GlobalContactRouter routed =
      routeActiveStencils(geometry, active_work.stencils);

  return routed;
}

}  // namespace detail
}  // namespace ksk::runtime
