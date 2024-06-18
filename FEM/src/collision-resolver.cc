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

class TriangleIntersectionQuery {
 public:
  using CoordType = Real;
  explicit TriangleIntersectionQuery(int triangle_id_, const System &system_, const VecXd &p)
      : triangle_id(triangle_id_), system(system_) {
    const auto &t = system.surfaces();
    for (int i = 0; i < 3; i++) {
      Vector<Real, 3> current_pos = system.get().currentPos(t(i));
      Vector<Real, 3> candidate_pos = current_pos + p.segment<3>(3 * t(i)) * dt;
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
  bool query(int pr_id) {
    if (pr_id <= triangle_id) return false;
    if (isAdjacentTriangle(triangle_id, pr_id)) return false;

    return true;
  }
  std::optional<Contact> collision_info{};
 private:
  int triangle_id{};
  std::reference_wrapper<System> system;
  std::reference_wrapper<VecXd> p;
  BBox<Real, 3> current_bbox;
  Real dt;
  [[nodiscard]] bool isAdjacentTriangle(int i, int j) const {
    const auto &t = system.get().surfaces();
    for (int k = 0; k < 3; k++)
      if (t(k, i) == t(0, j) || t(k, i) == t(1, j) || t(k, i) == t(2, j))
        return true;
    return false;
  }
};

struct Trajectory {
  using CoordType = Real;
  Matrix<Real, 3, 3> x;
  Matrix<Real, 3, 3> u;
  [[nodiscard]] BBox<Real, 3> bbox() const {
    BBox<Real, 3> box;
    for (int i = 0; i < 3; i++) {
      spatify::Vector<Real, 3> pos{x(0, i), x(1, i), x(2, i)};
      spatify::Vector<Real, 3> uvec{u(0, i), u(1, i), u(2, i)};
      box.expand(pos);
      box.expand(pos + uvec);
    }
    return box;
  }
};

struct SystemTrajectoryAccessor {
  std::reference_wrapper<const VecXd> x;
  std::reference_wrapper<const Matrix<int, 3, Dynamic>> triangles;
  std::reference_wrapper<const VecXd> p;
  Real dt;
  using PrimitiveType = Trajectory;

  [[nodiscard]]
  size_t size() const {
    return triangles.get().cols();
  }
  Trajectory operator()(int idx) const {
    Trajectory traj;
    for (int i = 0; i < 3; i++) {
      traj.x.col(i) = x.get().segment<3>(3 * triangles.get()(i, idx));
      traj.u.col(i) = p.get().segment<3>(3 * triangles.get()(i, idx)) * dt;
    }
    return traj;
  }
};

void CollisionDetector::detect(const System &system, const VecXd &p, Real dt) {
  const auto &triangles = system.surfaces();
  const auto &x = system.currentConfig();
  const auto &xdot = system.xdot;

  lbvh->update(SystemTrajectoryAccessor{x, triangles, p, dt});
  parallel_for(0, system.numTriangles() - 1, [&](int i) {
    auto intersection_query = TriangleIntersectionQuery(i, system, p);
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
    return std::make_optional(pos, t);
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
    auto normal = updated_x4_x3.cross(updated_x2_x1).float_normalized();
    return std::make_optional(pos, t);
  }
  return std::nullopt;
}

}