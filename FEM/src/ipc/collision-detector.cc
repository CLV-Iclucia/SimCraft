//
// Created by creeper on 5/25/24.
//
#include <Maths/equations.h>
#include <Maths/linalg-utils.h>
#include <fem/ipc/collision-detector.h>
#include <fem/ipc/distances.h>
#include <fem/system.h>
#include <tbb/tbb.h>
namespace sim::fem::ipc {
using maths::mixedProduct;
using spatify::parallel_for;

maths::CubicEquationRoots solveCoplanarTime(const CCDQuery &query, Real toi) {
  const auto &[x1, x2_x1, x3_x1, x4_x1, u1, u2_u1, u3_u1, u4_u1] = query;
  Real a = mixedProduct(u2_u1, u3_u1, u4_u1);
  Real b = mixedProduct(x2_x1, u3_u1, u4_u1) +
           mixedProduct(u2_u1, x3_x1, u4_u1) +
           mixedProduct(u2_u1, u3_u1, x4_x1);
  Real c = mixedProduct(u2_u1, x3_x1, x4_x1) +
           mixedProduct(x2_x1, u3_u1, x4_x1) +
           mixedProduct(x2_x1, x3_x1, u4_u1);
  Real d = mixedProduct(x2_x1, x3_x1, x4_x1);
  return maths::clampedCubicSolve({.a = a, .b = b, .c = c, .d = d}, 0.0, toi,
                                  1e-10);
}

std::optional<Contact> eeCCD(const CCDQuery &query, Real toi) {
  auto solution = solveCoplanarTime(query, toi);
  if (!solution.numRoots)
    return std::nullopt;
  if (solution.infiniteSolutions)
    throw std::runtime_error(
        "Infinite solutions detected in eeCCD, cannot be handled");
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
      Real a00 = updated_x2_x1(j), a01 = -updated_x4_x3(j),
           a10 = updated_x2_x1(k), a11 = -updated_x4_x3(k);
      Real b0 = updated_x3_x1(j), b1 = -updated_x3_x1(k);
      auto linearSystem = maths::BinaryLinearSystem{
          .a00 = a00, .a01 = a01, .a10 = a10, .a11 = a11, .b0 = b0, .b1 = b1};
      if (hasInfiniteSolutions(linearSystem))
        continue;
      sol = maths::binaryLinearSolve(linearSystem);
      if (!sol)
        hasSol = false;
      break;
    }
    if (!hasSol)
      continue;
    [[unlikely]] if (!sol) {
      // the two edges are on the same line
      Real la = updated_x2_x1.norm();
      Real lb = updated_x4_x3.norm();
      auto updated_x4_x2 = updated_x4_x1 - updated_x2_x1;
      auto updated_x3_x2 = updated_x3_x1 - updated_x2_x1;
      Real longest_distance =
          std::max({la, lb, updated_x4_x1.norm(), updated_x4_x2.norm(),
                    updated_x3_x1.norm(), updated_x3_x2.norm()});
      if (longest_distance > la + lb)
        continue;
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
    throw std::runtime_error(
        "Infinite solutions detected in vtCCD, cannot be handled");
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
      Real a00 = updated_x2_x4(j), a01 = updated_x3_x4(j),
           a10 = updated_x2_x4(k), a11 = updated_x3_x4(k);
      Real b0 = updated_x1_x4(j), b1 = updated_x1_x4(k);
      auto linearSystem = maths::BinaryLinearSystem{
          .a00 = a00, .a01 = a01, .a10 = a10, .a11 = a11, .b0 = b0, .b1 = b1};
      if (hasInfiniteSolutions(linearSystem))
        continue;
      sol = maths::binaryLinearSolve(linearSystem);
      if (!sol)
        hasSol = false;
      break;
    }
    if (!hasSol)
      continue;
    [[unlikely]] if (!sol)
      throw std::runtime_error("Infinite solutions detected in vtCCD for "
                               "barycentric coordinates, which is unexpected");

    const auto &[a, b] = *sol;
    if (a < 0.0 || b < 0.0 || a + b > 1.0)
      continue;
    return std::make_optional<Contact>(updated_x1, t);
  }
  return std::nullopt;
}

std::optional<Real> CollisionDetector::runACCDReserved(
    CCDMode mode, std::array<Vector<Real, 3>, 4> &x,
    std::array<Vector<Real, 3>, 4> &p, Real toi, Real reservedDistance) const {
  auto computeSquaredDistance = [&]() -> Real {
    if (mode == CCDMode::EE)
      return distanceSqrEdgeEdge(x[0], x[1], x[2], x[3]);
    return distanceSqrPointTriangle(x[0], x[1], x[2], x[3]);
  };
  Real lp = 0.0;
  if (mode == CCDMode::EE)
    lp =
        std::max(p[0].norm(), p[1].norm()) + std::max(p[2].norm(), p[3].norm());
  else
    lp =
        p[0].norm() + std::max(p[1].norm(), std::max(p[2].norm(), p[3].norm()));
  if (lp == 0.0)
    return std::nullopt;
  Real dSqr = computeSquaredDistance();
  Real g = s * (dSqr - reservedDistance * reservedDistance) /
           (std::sqrt(dSqr) + reservedDistance);
  Real t = 0.0;
  Real tl = (1.0 - s) * (dSqr - reservedDistance * reservedDistance) /
            ((std::sqrt(dSqr) + reservedDistance) * lp);
  while (true) {
    for (int i = 0; i < 4; i++)
      x[i] += p[i] * tl;
    dSqr = computeSquaredDistance();
    if (t > 0.0 && (dSqr - reservedDistance * reservedDistance) /
                           (std::sqrt(dSqr) + reservedDistance) <
                       g + 1e-10)
      break;
    t += tl;
    if (t > toi)
      return std::nullopt;
    tl = 0.9 * (dSqr - reservedDistance * reservedDistance) /
         ((std::sqrt(dSqr) + reservedDistance) * lp);
  }
  return t;
}

std::optional<Real>
CollisionDetector::runACCD(CCDMode mode, std::array<Vector<Real, 3>, 4> &x,
                           std::array<Vector<Real, 3>, 4> &p, Real toi) const {
  auto computeDistance = [&]() -> Real {
    if (mode == CCDMode::EE)
      return std::sqrt(distanceSqrEdgeEdge(x[0], x[1], x[2], x[3]));
    else
      return std::sqrt(distanceSqrPointTriangle(x[0], x[1], x[2], x[3]));
  };
  Real lp = 0.0;
  if (mode == CCDMode::EE)
    lp =
        std::max(p[0].norm(), p[1].norm()) + std::max(p[2].norm(), p[3].norm());
  else
    lp =
        p[0].norm() + std::max(p[1].norm(), std::max(p[2].norm(), p[3].norm()));
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
      [[unlikely]] if (t == 0.0)
        t = tl;
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

std::optional<Real>
CollisionDetector::detectVertexTriangleCollision(const VecXd &p) {
  std::atomic<Real> toi = 1.0;
  std::atomic<bool> hasContact{false};
  
  tbb::parallel_for(0, system.numVertices(), [&](int vertexIdx) {
    auto vertexTrajectoryBBox = system.geometryManager().getTrajectoryAccessor(
        system.currentConfig(), p, toi).vertexBBox(vertexIdx);
    
    trianglesBVH().runSpatialQuery(
        [&](int triangleIdx) -> bool {
          if (system.triangleContainsVertex(triangleIdx, vertexIdx))
            return false;

          auto triangleVertices = system.getTriangleVertices(triangleIdx);
          
          CCDQuery query{
              .x1 = system.currentConfig().segment<3>(3 * vertexIdx),
              .x2 = system.currentConfig().segment<3>(3 * triangleVertices.x),
              .x3 = system.currentConfig().segment<3>(3 * triangleVertices.y),
              .x4 = system.currentConfig().segment<3>(3 * triangleVertices.z),
              .u1 = p.segment<3>(3 * vertexIdx),
              .u2 = p.segment<3>(3 * triangleVertices.x),
              .u3 = p.segment<3>(3 * triangleVertices.y),
              .u4 = p.segment<3>(3 * triangleVertices.z)
          };
          
          auto contact = runACCD(ACCDOptions{CCDMode::VT, query, toi});
          if (!contact)
            return false;
          
          if (*contact < toi) {
            hasContact.store(true);
            toi.store(*contact);
          }
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return vertexTrajectoryBBox.overlap(bbox);
        });
  });
  
  if (hasContact)
    return toi;
  return std::nullopt;
}

std::optional<Real>
CollisionDetector::detectEdgeEdgeCollision(const VecXd &p) {
  std::atomic<Real> toi = 1.0;
  std::atomic<bool> hasContact{false};
  
  tbb::parallel_for(0, system.numEdges(), [&](int edgeIdx) {
    auto edgeTrajectoryBBox = system.geometryManager().getTrajectoryAccessor(
        system.currentConfig(), p, toi).edgeBBox(edgeIdx);
    
    edgesBVH().runSpatialQuery(
        [&](int otherEdgeIdx) -> bool {
          if (system.checkEdgeAdjacent(edgeIdx, otherEdgeIdx))
            return false;

          auto edgeVertices = system.getGlobalEdge(edgeIdx);
          auto otherEdgeVertices = system.getGlobalEdge(otherEdgeIdx);
          
          CCDQuery query{
              .x1 = system.currentConfig().segment<3>(3 * edgeVertices[0]),
              .x2 = system.currentConfig().segment<3>(3 * edgeVertices[1]),
              .x3 = system.currentConfig().segment<3>(3 * otherEdgeVertices[0]),
              .x4 = system.currentConfig().segment<3>(3 * otherEdgeVertices[1]),
              .u1 = p.segment<3>(3 * edgeVertices[0]),
              .u2 = p.segment<3>(3 * edgeVertices[1]),
              .u3 = p.segment<3>(3 * otherEdgeVertices[0]),
              .u4 = p.segment<3>(3 * otherEdgeVertices[1])
          };
          
          auto contact = runACCD(ACCDOptions{CCDMode::EE, query, toi});
          if (!contact)
            return false;

          if (*contact < toi) {
            hasContact.store(true);
            toi.store(*contact);
          }
          return true;
        },
        [&](const BBox<Real, 3> &bbox) -> bool {
          return edgeTrajectoryBBox.overlap(bbox);
        });
  });
  
  if (hasContact)
    return toi;
  return std::nullopt;
}

std::optional<Real> CollisionDetector::detect(const VecXd &p) {
  auto vtCollision = detectVertexTriangleCollision(p);
  auto eeCollision = detectEdgeEdgeCollision(p);
  
  if (!vtCollision && !eeCollision)
    return std::nullopt;

  if (!vtCollision)
    return eeCollision;

  if (!eeCollision)
    return vtCollision;

  return std::min(*vtCollision, *eeCollision);
}

void CollisionDetector::updateBVHs(const VecXd &p, Real toi) {
  auto trajectoryAccessor = system.geometryManager().getTrajectoryAccessor(
      system.currentConfig(), p, toi);
  
  if (system.numTriangles() > 0) {
    struct BVHAdapter {
      using CoordType = Real;
      const decltype(trajectoryAccessor)& accessor;
      
      explicit BVHAdapter(const decltype(trajectoryAccessor)& acc) : accessor(acc) {}
      
      [[nodiscard]] BBox<Real, 3> bbox(int idx) const {
        return accessor.triangleBBox(idx);
      }
      
      [[nodiscard]] int size() const {
        return accessor.triangleSize();
      }
    };
    
    BVHAdapter adapter(trajectoryAccessor);
    triangles_bvh.update(adapter);
  }
  
  if (system.numEdges() > 0) {
    struct BVHAdapter {
      using CoordType = Real;
      const decltype(trajectoryAccessor)& accessor;
      
      explicit BVHAdapter(const decltype(trajectoryAccessor)& acc) : accessor(acc) {}
      
      [[nodiscard]] BBox<Real, 3> bbox(int idx) const {
        return accessor.edgeBBox(idx);
      }
      
      [[nodiscard]] int size() const {
        return accessor.edgeSize();
      }
    };
    
    BVHAdapter adapter(trajectoryAccessor);
    edges_bvh.update(adapter);
  }
}

} // namespace fem::ipc