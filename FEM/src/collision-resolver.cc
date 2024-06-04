//
// Created by creeper on 5/25/24.
//
#include <fem/collision-resolver.h>
#include <fem/system.h>
#include <Maths/equations.h>
#include <Maths/linalg-utils.h>
namespace fem {
using spatify::parallel_for;
using maths::mixedProduct;

class TriangleIntersectionQuery : public spatify::SpatialQuery<TriangleIntersectionQuery> {
 public:
  explicit TriangleIntersectionQuery(int triangle_id_, const System *system_)
      : triangle_id(triangle_id_), system(system_) {
    const auto &t = system->surfaces;
    for (int i = 0; i < 3; i++) {
      Vector<Real, 3> current_pos = system->x.segment<3>(3 * t(i));
      Vector<Real, 3> candidate_pos = current_pos + system->xdot.segment<3>(3 * t(i)) * dt;
      Real a = current_pos(0);
      Real b = current_pos(1);
      Real c = current_pos(2);
      Real d = candidate_pos(0);
      Real e = candidate_pos(1);
      Real f = candidate_pos(2);
      current_bbox.expand({a, b, c}).expand({d, e, f});
    }
  }
  bool query(const BBox<Real, 3> &bbox) const {
    return current_bbox.overlap(bbox);
  }
  bool query(int pr_id) override {
    if (pr_id <= triangle_id) return false;
    if (isAdjacentTriangle(triangle_id, pr_id)) return false;

    return true;
  }
  std::optional<Contact> collision_info{};
 private:
  int triangle_id{};
  const System *system;
  BBox<Real, 3> current_bbox;
  Real dt;
  [[nodiscard]] bool isAdjacentTriangle(int i, int j) const {
    const auto &t = system->surfaces;
    for (int k = 0; k < 3; k++)
      if (t(k, i) == t(0, j) || t(k, i) == t(1, j) || t(k, i) == t(2, j))
        return true;
    return false;
  }
};
class WallCollisionQuery : public spatify::SpatialQuery<WallCollisionQuery> {
 public:
  explicit WallCollisionQuery(const System *system_) : system(system_) {
    const auto &t = system->surfaces;
    Real dt;
    for (int i = 0; i < 3; i++) {
      Vector<Real, 3> current_pos = system->x.segment<3>(3 * t(i));
      Vector<Real, 3> candidate_pos = current_pos + system->xdot.segment<3>(3 * t(i)) * dt;
      Real a = current_pos(0);
      Real b = current_pos(1);
      Real c = current_pos(2);
      Real d = candidate_pos(0);
      Real e = candidate_pos(1);
      Real f = candidate_pos(2);
      current_bbox.expand({a, b, c}).expand({d, e, f});
    }
  }
  bool query(const BBox<Real, 3> &bbox) const {
    return current_bbox.overlap(bbox);
  }
  bool query(int pr_id) override {
    return true;
  }
  std::optional<Contact> collision_info{};
 private:
  const System *system;
  BBox<Real, 3> current_bbox;
};
void CollisionDetector::detect(const System &system, Real dt) {
  const auto &triangles = system.surfaces;
  const auto &x = system.x;
  const auto &xdot = system.xdot;
  auto getBBox = [&](int id) -> BBox<Real, 3> {
    const auto &t = triangles.col(id);
    BBox<Real, 3> bbox;
    for (int i = 0; i < 3; i++) {
      Vector<Real, 3> current_pos = x.segment<3>(3 * t(i));
      Vector<Real, 3> candidate_pos = current_pos + xdot.segment<3>(3 * t(i));
      Real a = current_pos(0);
      Real b = current_pos(1);
      Real c = current_pos(2);
      Real d = candidate_pos(0);
      Real e = candidate_pos(1);
      Real f = candidate_pos(2);
      bbox.expand({a, b, c}).expand({d, e, f});
    }
    return bbox;
  };
  lbvh->update(system.numTriangles(), getBBox);
  parallel_for(0, system.numTriangles() - 1, [&](int i) {
    auto intersection_query = TriangleIntersectionQuery(i, &system);
    lbvh->runSpatialQuery(intersection_query);
    if (!intersection_query.collision_info)
      return;
    active_contacts.push_back(*intersection_query.collision_info);
  });
}

static maths::CubicEquationRoots solveCoplanarTime(const CCDQuery &query, Real toi) {
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  Real a = mixedProduct(u2_u1, u3_u1, u4_u1);
  Real b = mixedProduct(x2_x1, u3_u1, u4_u1) + mixedProduct(u2_u1, x3_x1, u4_u1) + mixedProduct(u2_u1, u3_u1, x4_x1);
  Real c = mixedProduct(u2_u1, x3_x1, x4_x1) + mixedProduct(x2_x1, u3_u1, x4_x1) + mixedProduct(x2_x1, x3_x1, u4_u1);
  Real d = mixedProduct(x2_x1, x3_x1, x4_x1);
  return maths::clampedCubicSolve({a, b, c, d}, 0.0, toi, 1e-10);
}

std::optional<Contact> eeCCD(const CCDQuery &query, Real toi) {
  auto solution = solveCoplanarTime(query, toi);
  if (!solution.num_roots) return std::nullopt;
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  for (int i = 0; i < solution.num_roots; i++) {
    Real t = solution.roots[i];
    auto updated_x1 = x1 + u1 * t;
    auto updated_x2_x1 = x2_x1 + u2_u1 * t;
    auto updated_x3_x1 = x3_x1 + u3_u1 * t;
    auto updated_x4_x1 = x4_x1 + u4_u1 * t;
    auto updated_x4_x3 = updated_x4_x1 - updated_x3_x1;
    Real ta, tb;
    if (!maths::binaryLinearSolve({.a00 = updated_x2_x1.squaredNorm(), .a01 = -updated_x2_x1.dot(updated_x4_x3),
                                      .a10 =-updated_x2_x1.dot(updated_x4_x3), .a11 = updated_x4_x3.squaredNorm(),
                                      .b0 = updated_x2_x1.dot(updated_x3_x1), .b1 = -updated_x4_x3.dot(updated_x3_x1)},
                                  ta, tb))
      continue;
    if (ta < 0 || ta > 1 || tb < 0 || tb > 1) continue;
    auto pos = updated_x1 + ta * updated_x2_x1;
    auto normal = updated_x4_x3.cross(updated_x2_x1);
    return std::make_optional(pos, maths::constructFrame(normal), t);
  }
  return std::nullopt;
}

std::optional<Contact> vtCCD(const CCDQuery &query, Real toi) {
  auto solution = solveCoplanarTime(query, toi);
  if (!solution.num_roots) return std::nullopt;
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  for (int i = 0; i < solution.num_roots; i++) {
    Real t = solution.roots[i];
    auto updated_x1 = x1 + u1 * t;
    auto updated_x2_x1 = x2_x1 + u2_u1 * t;
    auto updated_x3_x1 = x3_x1 + u3_u1 * t;
    auto updated_x4_x1 = x4_x1 + u4_u1 * t;
    auto updated_x4_x3 = updated_x4_x1 - updated_x3_x1;
    Real ta, tb;
    if (!maths::binaryLinearSolve({.a00 = updated_x2_x1.squaredNorm(), .a01 = -updated_x2_x1.dot(updated_x4_x3),
                                      .a10 =-updated_x2_x1.dot(updated_x4_x3), .a11 = updated_x4_x3.squaredNorm(),
                                      .b0 = updated_x2_x1.dot(updated_x3_x1), .b1 = -updated_x4_x3.dot(updated_x3_x1)},
                                  ta,
                                  tb))
      continue;
    if (ta < 0 || ta > 1 || tb < 0 || tb > 1) continue;
    auto pos = updated_x1 + ta * updated_x2_x1;
    auto normal = updated_x4_x3.cross(updated_x2_x1).normalized();
    return std::make_optional(pos, maths::constructFrame(normal), t);
  }
  return std::nullopt;
}

}