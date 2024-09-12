//
// Created by creeper on 5/25/24.
//
#include <tbb/tbb.h>
#include <fem/ipc/collision-detector.h>
#include <fem/system.h>
#include <Maths/equations.h>
#include <Maths/linalg-utils.h>
#include <fem/ipc/distances.h>
namespace fem::ipc {
using spatify::parallel_for;
using maths::mixedProduct;

struct Trajectory {
  const System &system;
  const VecXd &p;
  Real toi = 1.0;
  int idx{};
};

static BBox<Real, 3> computeTrajectoryBBox(const Trajectory &trajectory) {
  const auto &[system, p, toi, idx] = trajectory;
  BBox<Real, 3> box;
  for (int i = 0; i < 3; i++) {
    auto pos = system.currentPos(system.surfaces()(i, idx));
    auto u = p.segment<3>(3 * system.surfaces()(i, idx)) * toi;
    auto pos_next = pos + u;
    box.expand({pos(0), pos(1), pos(2)}).expand({pos_next(0), pos_next(1), pos_next(2)});
  }
  return box;
}

struct SystemTrajectoryAccessor {
  const VecXd &x;
  const System &system;
  const VecXd &p;
  using CoordType = Real;

  [[nodiscard]]
  size_t size() const {
    return system.numTriangles();
  }
  [[nodiscard]] BBox<Real, 3> bbox(int idx) const {
    return computeTrajectoryBBox({.system = system, .p = p, .idx = idx});
  }
};

static BBox<Real, 3> computeTriangleBBox(const System &system, int idx) {
  BBox<Real, 3> box;
  for (int i = 0; i < 3; i++) {
    auto pos = system.currentPos(system.surfaces()(i, idx));
    box.expand({pos(0), pos(1), pos(2)});
  }
  return box;
}

maths::CubicEquationRoots solveCoplanarTime(const CCDQuery &query, Real toi) {
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  Real a = mixedProduct(u2_u1, u3_u1, u4_u1);
  Real b = mixedProduct(x2_x1, u3_u1, u4_u1) + mixedProduct(u2_u1, x3_x1, u4_u1) + mixedProduct(u2_u1, u3_u1, x4_x1);
  Real c = mixedProduct(u2_u1, x3_x1, x4_x1) + mixedProduct(x2_x1, u3_u1, x4_x1) + mixedProduct(x2_x1, x3_x1, u4_u1);
  Real d = mixedProduct(x2_x1, x3_x1, x4_x1);
  return maths::clampedCubicSolve({.a = a, .b = b, .c = c, .d = d}, 0.0, toi, 1e-10);
}

std::optional<Contact> eeCCD(const CCDQuery &query, Real toi) {
  auto solution = solveCoplanarTime(query, toi);
  if (!solution.numRoots)
    return std::nullopt;
  if (solution.infiniteSolutions)
    throw std::runtime_error("Infinite solutions detected in eeCCD, cannot be handled");
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  std::optional<maths::BinaryLinearSolution> sol{};
  for (int i = 0; i < solution.numRoots; i++) {
    Real t = solution.roots[i];
    auto updated_x2_x1 = x2_x1 + u2_u1 * t;
    auto updated_x3_x1 = x3_x1 + u3_u1 * t;
    auto updated_x4_x1 = x4_x1 + u4_u1 * t;
    auto updated_x4_x3 = updated_x4_x1 - updated_x3_x1;
    bool hasSol = true;
    for (int j = 0; j < 3; j++) {
      int k = (j + 1) % 3;
      Real a00 = updated_x2_x1(j), a01 = -updated_x4_x3(j), a10 = updated_x2_x1(k), a11 = -updated_x4_x3(k);
      Real b0 = updated_x3_x1(j), b1 = -updated_x3_x1(k);
      auto linearSystem = maths::BinaryLinearSystem{.a00 = a00, .a01 = a01, .a10 = a10, .a11 = a11, .b0 = b0, .b1 = b1};
      if (hasInfiniteSolutions(linearSystem)) continue;
      sol = maths::binaryLinearSolve(linearSystem);
      if (!sol)
        hasSol = false;
      break;
    }
    if (!hasSol) continue;
    [[unlikely]] if (!sol) {
      // the two edges are on the same line
      Real la = updated_x2_x1.norm();
      Real lb = updated_x4_x3.norm();
      auto updated_x4_x2 = updated_x4_x1 - updated_x2_x1;
      auto updated_x3_x2 = updated_x3_x1 - updated_x2_x1;
      Real longest_distance =
          std::max({la, lb, updated_x4_x1.norm(), updated_x4_x2.norm(), updated_x3_x1.norm(), updated_x3_x2.norm()});
      if (longest_distance > la + lb) continue;
      auto updated_x1 = x1 + u1 * t;
      auto updated_x2 = updated_x2_x1 + updated_x1;
      auto updated_x3 = updated_x3_x1 + updated_x1;
      if (longest_distance == la)
        return std::make_optional<Contact>(0.5 * updated_x4_x3 + updated_x3, t);
      if (longest_distance == lb)
        return std::make_optional<Contact>(0.5 * updated_x2_x1 + updated_x1, t);
      if (longest_distance == updated_x4_x1.norm())
        return std::make_optional<Contact>(0.5 * updated_x3_x2 + updated_x2, t);
      if (longest_distance == updated_x4_x2.norm())
        return std::make_optional<Contact>(0.5 * updated_x3_x1 + updated_x1, t);
      if (longest_distance == updated_x3_x1.norm())
        return std::make_optional<Contact>(0.5 * updated_x4_x2 + updated_x2, t);
      else
        return std::make_optional<Contact>(0.5 * updated_x4_x1 + updated_x1, t);
    }
    auto [a, b] = *sol;
    if (a < 0.0 || a > 1.0 || b < 0.0 || b > 1.0)
      continue;
    auto pos = x1 + u1 * t + a * updated_x2_x1;
    return std::make_optional<Contact>(pos, t);
  }
  return std::nullopt;
}

std::optional<Contact> vtCCD(const CCDQuery &query, Real toi) {
  auto solution = solveCoplanarTime(query, toi);
  if (!solution.numRoots)
    return std::nullopt;
  if (solution.infiniteSolutions)
    throw std::runtime_error("Infinite solutions detected in vtCCD, cannot be handled");
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  std::optional<maths::BinaryLinearSolution> sol{};
  for (int i = 0; i < solution.numRoots; i++) {
    Real t = solution.roots[i];
    auto updated_x1 = x1 + u1 * t;
    auto updated_x2 = x2_x1 + u2_u1 * t + updated_x1;
    auto updated_x3 = x3_x1 + u3_u1 * t + updated_x1;
    auto updated_x4 = x4_x1 + u4_u1 * t + updated_x1;
    auto updated_x1_x4 = updated_x1 - updated_x4;
    auto updated_x2_x4 = updated_x2 - updated_x4;
    auto updated_x3_x4 = updated_x3 - updated_x4;
    bool hasSol = true;
    for (int j = 0; j < 3; j++) {
      int k = (j + 1) % 3;
      Real a00 = updated_x2_x4(j), a01 = updated_x3_x4(j), a10 = updated_x2_x4(k), a11 = updated_x3_x4(k);
      Real b0 = updated_x1_x4(j), b1 = updated_x1_x4(k);
      auto linearSystem = maths::BinaryLinearSystem{.a00 = a00, .a01 = a01, .a10 = a10, .a11 = a11, .b0 = b0, .b1 = b1};
      if (hasInfiniteSolutions(linearSystem)) continue;
      sol = maths::binaryLinearSolve(linearSystem);
      if (!sol) hasSol = false;
      break;
    }
    if (!hasSol)
      continue;
    [[unlikely]] if (!sol)
      throw std::runtime_error("Infinite solutions detected in vtCCD for barycentric coordinates, which is unexpected");

    const auto &[a, b] = *sol;
    if (a < 0.0 || b < 0.0 || a + b > 1.0)
      continue;
    return std::make_optional<Contact>(updated_x1, t);
  }
  return std::nullopt;
}

std::optional<Real> CollisionDetector::runACCDReserved(CCDMode mode,
                                                       std::array<Vector<Real, 3>, 4> &x,
                                                       std::array<Vector<Real, 3>, 4> &p,
                                                       Real toi,
                                                       Real reservedDistance) const {
  auto computeSquaredDistance = [&]() -> Real {
    Real dist = 0.0;
    if (mode == CCDMode::EE)
      dist = distanceEdgeEdge(x[0], x[1], x[2], x[3]);
    else dist = distancePointTriangle(x[0], x[1], x[2], x[3]);
    return dist * dist;
  };
  Real lp = 0.0;
  if (mode == CCDMode::EE)
    lp = std::max(p[0].norm(), p[1].norm()) + std::max(p[2].norm(), p[3].norm());
  else lp = p[0].norm() + std::max(p[1].norm(), std::max(p[2].norm(), p[3].norm()));
  if (lp == 0.0)
    return std::nullopt;
  Real dSqr = computeSquaredDistance();
  Real g = s * (dSqr - reservedDistance * reservedDistance) / (std::sqrt(dSqr) + reservedDistance);
  Real t = 0.0;
  Real tl = (1.0 - s) * (dSqr - reservedDistance * reservedDistance) / ((std::sqrt(dSqr) + reservedDistance) * lp);
  while (true) {
    for (int i = 0; i < 4; i++)
      x[i] += p[i] * tl;
    dSqr = computeSquaredDistance();
    if (t > 0.0 && (dSqr - reservedDistance * reservedDistance) / (std::sqrt(dSqr) + reservedDistance) < g + 1e-10)
      break;
    t += tl;
    if (t > toi)
      return std::nullopt;
    tl = 0.9 * (dSqr - reservedDistance * reservedDistance) / ((std::sqrt(dSqr) + reservedDistance) * lp);
  }
  return t;
}

std::optional<Real> CollisionDetector::runACCD(CCDMode mode,
                                               std::array<Vector<Real, 3>, 4> &x,
                                               std::array<Vector<Real, 3>, 4> &p,
                                               Real toi) const {
  auto computeDistance = [&]() -> Real {
    if (mode == CCDMode::EE)
      return distanceEdgeEdge(x[0], x[1], x[2], x[3]);
    else
      return distancePointTriangle(x[0], x[1], x[2], x[3]);
  };
  Real lp = 0.0;
  if (mode == CCDMode::EE)
    lp = std::max(p[0].norm(), p[1].norm()) + std::max(p[2].norm(), p[3].norm());
  else lp = p[0].norm() + std::max(p[1].norm(), std::max(p[2].norm(), p[3].norm()));
  if (lp == 0.0)
    return std::nullopt;
  Real dis = computeDistance();
  Real g = s * dis;
  Real t = 0.0;
  Real tl = (1.0 - s) * (dis / lp);
  while (true) {
    for (int i = 0; i < 4; i++)
      x[i] += p[i] * tl;
    dis = computeDistance();
    if (dis < g + 1e-10) {
      [[unlikely]] if (t == 0.0) t = tl;
      break;
    }
    t += tl;
    if (t > toi)
      return std::nullopt;
    tl = 0.9 * dis / lp;
  }
  return t;
}

std::optional<Real> CollisionDetector::runACCD(const ACCDOptions &options) {
  const auto &[mode, query, toi, reservedDistance] = options;
  const auto &[x1, x2, x3, x4, u1, u2, u3, u4] = query;
  auto pBar = (u1 + u2 + u3 + u4) * 0.25;
  std::array<Vector<Real, 3>, 4> x{x1, x2, x3, x4};
  std::array<Vector<Real, 3>, 4> p{u1 - pBar, u2 - pBar, u3 - pBar, u4 - pBar};
  if (reservedDistance)
    return runACCDReserved(mode, x, p, toi, reservedDistance.value());
  return runACCD(mode, x, p, toi);
}

std::optional<Real> CollisionDetector::trianglePairCCD(const TrianglePairCCDQuery &query) {
  const auto &[system, p, ta, tb, _] = query;
  Real toi = query.toi;
  const auto &triangles = system.surfaces();
  auto trajectory_bbox_ta = computeTrajectoryBBox({.system = system, .p = p, .toi = toi, .idx = ta});
  auto trajectory_bbox_tb = computeTrajectoryBBox({.system = system, .p = p, .toi = toi, .idx = tb});
  if (!trajectory_bbox_ta.overlap(trajectory_bbox_tb)) return std::nullopt;
  std::array<Vec3d, 3> pa{system.currentPos(triangles(0, ta)),
                          system.currentPos(triangles(1, ta)),
                          system.currentPos(triangles(2, ta))};
  std::array<Vec3d, 3> pb{system.currentPos(triangles(0, tb)),
                          system.currentPos(triangles(1, tb)),
                          system.currentPos(triangles(2, tb))};
  std::array<Vec3d, 3> ua{p.segment<3>(3 * triangles(0, ta)),
                          p.segment<3>(3 * triangles(1, ta)),
                          p.segment<3>(3 * triangles(2, ta))};
  std::array<Vec3d, 3> ub{p.segment<3>(3 * triangles(0, tb)),
                          p.segment<3>(3 * triangles(1, tb)),
                          p.segment<3>(3 * triangles(2, tb))};
  std::optional<Real> firstContact{};
  for (int i = 0; i < 3; i++) {
    std::optional<Real> contact =
        runACCD(ACCDOptions{CCDMode::VT, {.x1 = pa[i], .x2 = pb[0], .x3 = pb[1], .x4 = pb[2], .u1 = ua[i], .u2 = ub[0],
            .u3 = ub[1], .u4 = ub[2]}, toi});
    if (!contact) continue;
    if (!firstContact || contact < firstContact) {
      firstContact = contact;
      toi = *firstContact;
    }
  }
  for (int i = 0; i < 3; i++) {
    std::optional<Real> contact =
        runACCD({CCDMode::VT, {.x1 = pb[i], .x2 = pa[0], .x3 = pa[1], .x4 = pa[2], .u1 = ub[i], .u2 = ua[0],
            .u3 = ua[1], .u4 = ua[2]}, toi});
    if (!contact) continue;
    if (!firstContact || *contact < firstContact) {
      firstContact = contact;
      toi = *firstContact;
    }
  }
  for (int i = 0; i < 3; i++) {
    int ea_1 = i, ea_2 = (i + 1) % 3;
    for (int j = 0; j < 3; j++) {
      int eb_1 = j, eb_2 = (j + 1) % 3;
      std::optional<Real> contact =
          runACCD({CCDMode::EE, {.x1 = pa[ea_1], .x2 = pa[ea_2], .x3 = pb[eb_1], .x4 = pb[eb_2], .u1 = ua[ea_1],
              .u2 = ua[ea_2], .u3 = ub[eb_1], .u4 = ub[eb_2]}, toi});
      if (!contact) continue;
      if (!firstContact || *contact < toi) {
        firstContact = contact;
        toi = *firstContact;
      }
    }
  }
  return toi;
}

std::optional<Real> CollisionDetector::detect(const System &system, const VecXd &p) {
  const auto &x = system.currentConfig();
  m_bvh->update(SystemTrajectoryAccessor{x, system, p});
  std::atomic<Real> toi = 1.0;
  std::atomic<bool> hasContact{false};
  tbb::parallel_for (0, system.numTriangles(), [&](int i) {
    auto trajectory_bbox = computeTrajectoryBBox({.system = system, .p = p, .toi = toi, .idx = i});
    m_bvh->runSpatialQuery(
        [&](int primitiveID) -> bool {
          if (i >= primitiveID) return false;
          if (system.checkTriangleAdjacent(i, primitiveID)) return false;
          auto t = trianglePairCCD({.system = system, .p = p, .ta = i, .tb = primitiveID, .toi = toi});
          if (!t) return false;
          if (*t > toi) return false;
          hasContact.store(true);
          toi.store(*t);
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return trajectory_bbox.overlap(bbox);
        });
  });
  if (hasContact) return toi;
  return std::nullopt;
}

}