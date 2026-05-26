//
// Created by creeper on 7/19/24.
//

#include <fem/ipc/barrier-functions.h>
#include <fem/ipc/constraint.h>
#include <fem/ipc/mollifier.h>
#include <Maths/block-types.h>
#include <glm/geometric.hpp>
#include <fem/system.h>

namespace sim::fem::ipc {
using maths::assembleLocalGrad;
using maths::assembleLocalHessian;

// =========================================================================
// VertexTriangleConstraint
// =========================================================================

void VertexTriangleConstraint::updateDistanceType() {
  auto p = x[globalVertex];
  auto a = x[globalTriVerts[0]];
  auto b = x[globalTriVerts[1]];
  auto c = x[globalTriVerts[2]];
  type = decidePointTriangleDistanceType(p, a, b, c);
}

Real VertexTriangleConstraint::distanceSqr() const {
  auto p = x[globalVertex];
  auto a = x[globalTriVerts[0]];
  auto b = x[globalTriVerts[1]];
  auto c = x[globalTriVerts[2]];
  switch (type) {
    case PointTriangleDistanceType::P_A:return distanceSqrPointPoint(p, a);
    case PointTriangleDistanceType::P_B:return distanceSqrPointPoint(p, b);
    case PointTriangleDistanceType::P_C:return distanceSqrPointPoint(p, c);
    case PointTriangleDistanceType::P_AB:return distanceSqrPointLine(p, a, b);
    case PointTriangleDistanceType::P_BC:return distanceSqrPointLine(p, b, c);
    case PointTriangleDistanceType::P_CA:return distanceSqrPointLine(p, c, a);
    case PointTriangleDistanceType::P_ABC:return distanceSqrPointPlane(p, a, b, c);
    default:throw std::runtime_error("Unknown distance type");
  }
}

void VertexTriangleConstraint::assembleBarrierGradient(
    const LogBarrier &barrier,
    maths::BlockVector<3> &globalGradient,
    Real kappa) const {
  Real dist = distanceSqr();
  if (dist >= barrier.dHatSqr()) return;
  Real scale = kappa * barrier.distanceSqrGradient(dist);
  
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      auto grad = localDistanceSqrPointPointGradient(x[globalVertex], x[globalTriVerts[0]]);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[0]};
      assembleLocalGrad<2>(globalGradient, idx, grad, scale);
      return;
    }
    case PointTriangleDistanceType::P_B: {
      auto grad = localDistanceSqrPointPointGradient(x[globalVertex], x[globalTriVerts[1]]);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[1]};
      assembleLocalGrad<2>(globalGradient, idx, grad, scale);
      return;
    }
    case PointTriangleDistanceType::P_C: {
      auto grad = localDistanceSqrPointPointGradient(x[globalVertex], x[globalTriVerts[2]]);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[2]};
      assembleLocalGrad<2>(globalGradient, idx, grad, scale);
      return;
    }
    case PointTriangleDistanceType::P_AB: {
      auto grad = localDistanceSqrPointLineGradient(x[globalVertex], x[globalTriVerts[0]], x[globalTriVerts[1]]);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[0], globalTriVerts[1]};
      assembleLocalGrad<3>(globalGradient, idx, grad, scale);
      return;
    }
    case PointTriangleDistanceType::P_BC: {
      auto grad = localDistanceSqrPointLineGradient(x[globalVertex], x[globalTriVerts[1]], x[globalTriVerts[2]]);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[1], globalTriVerts[2]};
      assembleLocalGrad<3>(globalGradient, idx, grad, scale);
      return;
    }
    case PointTriangleDistanceType::P_CA: {
      auto grad = localDistanceSqrPointLineGradient(x[globalVertex], x[globalTriVerts[2]], x[globalTriVerts[0]]);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[2], globalTriVerts[0]};
      assembleLocalGrad<3>(globalGradient, idx, grad, scale);
      return;
    }
    case PointTriangleDistanceType::P_ABC: {
      auto grad = localDistanceSqrPointPlaneGradient(x[globalVertex], x[globalTriVerts[0]], x[globalTriVerts[1]], x[globalTriVerts[2]]);
      std::array<int, 4> idx = {globalVertex, globalTriVerts[0], globalTriVerts[1], globalTriVerts[2]};
      assembleLocalGrad<4>(globalGradient, idx, grad, scale);
      return;
    }
    default:
      throw std::runtime_error("Unknown distance type in VT constraint gradient");
  }
}

void VertexTriangleConstraint::assembleBarrierHessian(
    const LogBarrier &barrier,
    maths::BlockSparseMatrix<3> &globalHessian,
    Real kappa) const {
  auto p = x[globalVertex];
  auto a = x[globalTriVerts[0]];
  auto b = x[globalTriVerts[1]];
  auto c = x[globalTriVerts[2]];
  Real dist = distanceSqr();
  if (dist >= barrier.dHatSqr()) return;
  Real bGrad = barrier.distanceSqrGradient(dist);
  Real bHess = barrier.distanceSqrHessian(dist);
  
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      auto grad = localDistanceSqrPointPointGradient(p, a);
      auto hess = localDistanceSqrPointPointHessian(p, a);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[0]};
      assembleLocalHessian<2>(globalHessian, idx, hess, grad, bGrad, bHess, kappa);
      return;
    }
    case PointTriangleDistanceType::P_B: {
      auto grad = localDistanceSqrPointPointGradient(p, b);
      auto hess = localDistanceSqrPointPointHessian(p, b);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[1]};
      assembleLocalHessian<2>(globalHessian, idx, hess, grad, bGrad, bHess, kappa);
      return;
    }
    case PointTriangleDistanceType::P_C: {
      auto grad = localDistanceSqrPointPointGradient(p, c);
      auto hess = localDistanceSqrPointPointHessian(p, c);
      std::array<int, 2> idx = {globalVertex, globalTriVerts[2]};
      assembleLocalHessian<2>(globalHessian, idx, hess, grad, bGrad, bHess, kappa);
      return;
    }
    case PointTriangleDistanceType::P_AB: {
      auto grad = localDistanceSqrPointLineGradient(p, a, b);
      auto hess = localDistanceSqrPointLineHessian(p, a, b);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[0], globalTriVerts[1]};
      assembleLocalHessian<3>(globalHessian, idx, hess, grad, bGrad, bHess, kappa);
      return;
    }
    case PointTriangleDistanceType::P_BC: {
      auto grad = localDistanceSqrPointLineGradient(p, b, c);
      auto hess = localDistanceSqrPointLineHessian(p, b, c);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[1], globalTriVerts[2]};
      assembleLocalHessian<3>(globalHessian, idx, hess, grad, bGrad, bHess, kappa);
      return;
    }
    case PointTriangleDistanceType::P_CA: {
      auto grad = localDistanceSqrPointLineGradient(p, c, a);
      auto hess = localDistanceSqrPointLineHessian(p, c, a);
      std::array<int, 3> idx = {globalVertex, globalTriVerts[2], globalTriVerts[0]};
      assembleLocalHessian<3>(globalHessian, idx, hess, grad, bGrad, bHess, kappa);
      return;
    }
    case PointTriangleDistanceType::P_ABC: {
      auto grad = localDistanceSqrPointPlaneGradient(p, a, b, c);
      auto hess = localDistanceSqrPointPlaneHessian(p, a, b, c);
      std::array<int, 4> idx = {globalVertex, globalTriVerts[0], globalTriVerts[1], globalTriVerts[2]};
      assembleLocalHessian<4>(globalHessian, idx, hess, grad, bGrad, bHess, kappa);
      return;
    }
    default:
      throw std::runtime_error("Unknown distance type in VT constraint hessian");
  }
}

// =========================================================================
// EdgeEdgeConstraint
// =========================================================================

void EdgeEdgeConstraint::updateDistanceType() {
  constexpr double PARALLEL_THRESHOLD = 1.0e-20;
  auto ea0 = x[globalEdgeA[0]];
  auto ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]];
  auto eb1 = x[globalEdgeB[1]];
  
  const glm::dvec3 u = ea1 - ea0;
  const glm::dvec3 v = eb1 - eb0;
  const glm::dvec3 w = ea0 - eb0;
  
  Real a = glm::dot(u, u);
  Real b = glm::dot(u, v);
  Real c = glm::dot(v, v);
  Real d = glm::dot(u, w);
  Real e = glm::dot(v, w);
  Real D = a * c - b * b;
  
  if (a == 0.0 && c == 0.0) {
    type = EdgeEdgeDistanceType::A_C;
    return;
  } else if (a == 0.0) {
    type = EdgeEdgeDistanceType::A_CD;
    return;
  } else if (c == 0.0) {
    type = EdgeEdgeDistanceType::AB_C;
    return;
  }
  
  Real parallel_tolerance = PARALLEL_THRESHOLD * std::max(1.0, a * c);
  if (glm::dot(glm::cross(u, v), glm::cross(u, v)) < parallel_tolerance) {
    type = decideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);
    return;
  }
  
  EdgeEdgeDistanceType default_case = EdgeEdgeDistanceType::AB_CD;
  
  Real sN = (b * e - c * d);
  double tN, tD;
  if (sN <= 0.0) {
    tN = e;
    tD = c;
    default_case = EdgeEdgeDistanceType::A_CD;
  } else if (sN >= D) {
    tN = e + b;
    tD = c;
    default_case = EdgeEdgeDistanceType::B_CD;
  } else {
    tN = (a * e - b * d);
    tD = D;
    if (tN > 0.0 && tN < tD &&
        glm::dot(glm::cross(u, v), glm::cross(u, v)) < parallel_tolerance) {
      if (sN < D / 2) {
        tN = e;
        tD = c;
        default_case = EdgeEdgeDistanceType::A_CD;
      } else {
        tN = e + b;
        tD = c;
        default_case = EdgeEdgeDistanceType::B_CD;
      }
    }
  }
  
  if (tN <= 0.0) {
    if (-d <= 0.0) {
      type = EdgeEdgeDistanceType::A_C;
    } else if (-d >= a) {
      type = EdgeEdgeDistanceType::B_C;
    } else {
      type = EdgeEdgeDistanceType::AB_C;
    }
  } else if (tN >= tD) {
    if ((-d + b) <= 0.0) {
      type = EdgeEdgeDistanceType::A_D;
    } else if ((-d + b) >= a) {
      type = EdgeEdgeDistanceType::B_D;
    } else {
      type = EdgeEdgeDistanceType::AB_D;
    }
  } else {
    type = default_case;
  }
}

Real EdgeEdgeConstraint::distanceSqr() const {
  auto A = x[globalEdgeA[0]];
  auto B = x[globalEdgeA[1]];
  auto C = x[globalEdgeB[0]];
  auto D = x[globalEdgeB[1]];
  switch (type) {
    case EdgeEdgeDistanceType::A_C:return distanceSqrPointPoint(A, C);
    case EdgeEdgeDistanceType::A_D:return distanceSqrPointPoint(A, D);
    case EdgeEdgeDistanceType::B_C:return distanceSqrPointPoint(B, C);
    case EdgeEdgeDistanceType::B_D:return distanceSqrPointPoint(B, D);
    case EdgeEdgeDistanceType::AB_C:return distanceSqrPointLine(A, B, C);
    case EdgeEdgeDistanceType::AB_D:return distanceSqrPointLine(A, B, D);
    case EdgeEdgeDistanceType::A_CD:return distanceSqrPointLine(A, C, D);
    case EdgeEdgeDistanceType::B_CD:return distanceSqrPointLine(B, C, D);
    case EdgeEdgeDistanceType::AB_CD:return distanceSqrLineLine(A, B, C, D);
    default:throw std::runtime_error("Unknown distance type in EE constraint");
  }
}

Real EdgeEdgeConstraint::epsCross() const {
  glm::dvec3 ea0 = X[globalEdgeA[0]];
  glm::dvec3 ea1 = X[globalEdgeA[1]];
  glm::dvec3 eb0 = X[globalEdgeB[0]];
  glm::dvec3 eb1 = X[globalEdgeB[1]];
  return 1e-3 * glm::dot(ea1 - ea0, ea1 - ea0) * glm::dot(eb1 - eb0, eb1 - eb0);
}

Real EdgeEdgeConstraint::crossSquaredNorm() const {
  glm::dvec3 ea0 = x[globalEdgeA[0]];
  glm::dvec3 ea1 = x[globalEdgeA[1]];
  glm::dvec3 eb0 = x[globalEdgeB[0]];
  glm::dvec3 eb1 = x[globalEdgeB[1]];
  glm::dvec3 cross = glm::cross(ea1 - ea0, eb1 - eb0);
  return glm::dot(cross, cross);
}

Real EdgeEdgeConstraint::mollifier() const {
  Real e_x = 1e-3 * epsCross();
  Real c = crossSquaredNorm();
  return mollifier(c, e_x);
}

LocalGrad<4> EdgeEdgeConstraint::crossedNormGradient() const {
  auto ea0 = x[globalEdgeA[0]];
  auto ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]];
  auto eb1 = x[globalEdgeB[1]];
  return edgeEdgeCrossSquareNormGradient(ea0, ea1, eb0, eb1);
}

LocalHessian<4> EdgeEdgeConstraint::crossedNormHessian() const {
  auto ea0 = x[globalEdgeA[0]];
  auto ea1 = x[globalEdgeA[1]];
  auto eb0 = x[globalEdgeB[0]];
  auto eb1 = x[globalEdgeB[1]];
  return edgeEdgeCrossSquaredNormHessian(ea0, ea1, eb0, eb1);
}

LocalGrad<4> EdgeEdgeConstraint::mollifierGradient() const {
  Real e_x = 1e-3 * epsCross();
  Real c = crossSquaredNorm();
  if (c >= e_x) return {};  // zero-initialized
  Real p_m_p_c = mollifierDerivative(c, e_x);
  auto p_c_p_x = crossedNormGradient();
  // p_m_p_x = p_m_p_c * p_c_p_x
  LocalGrad<4> result{};
  for (int i = 0; i < 4; i++)
    result[i] = p_m_p_c * p_c_p_x[i];
  return result;
}

LocalHessian<4> EdgeEdgeConstraint::mollifierHessian() const {
  Real e_x = epsCross();
  Real c = crossSquaredNorm();
  if (c >= e_x) return {};  // zero-initialized
  Real p_m_p_c = mollifierDerivative(c, e_x);
  LocalGrad<4> p_c_p_x = crossedNormGradient();
  LocalHessian<4> p2_c_p_x2 = crossedNormHessian();
  Real p2_m_p_c2 = mollifierSecondDerivative(c, e_x);
  // H = p2_m_p_c2 * outer(p_c_p_x) + p_m_p_c * p2_c_p_x2
  LocalHessian<4> result{};
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      glm::dmat3 outer = glm::outerProduct(p_c_p_x[i], p_c_p_x[j]);
      result[i][j] = p2_m_p_c2 * outer + p_m_p_c * p2_c_p_x2[i][j];
    }
  }
  return result;
}

LocalGrad<4> EdgeEdgeConstraint::mollifiedBarrierGradient(const LogBarrier &barrier) const {
  Real c = crossSquaredNorm();
  Real e_x = epsCross();
  Real dist = distanceSqr();
  Real b = barrier(dist);
  Real p_b_p_d = barrier.distanceSqrGradient(dist);
  LocalGrad<4> p_d_p_x = localDistanceSqrEdgeEdgeGradient(
      x[globalEdgeA[0]], x[globalEdgeA[1]],
      x[globalEdgeB[0]], x[globalEdgeB[1]]);
  if (c > e_x) return p_b_p_d * p_d_p_x;
  Real m = mollifier();
  LocalGrad<4> p_m_p_x = mollifierGradient();
  LocalGrad<4> result{};
  for (int i = 0; i < 4; i++)
    result[i] = m * p_b_p_d * p_d_p_x[i] + b * p_m_p_x[i];
  return result;
}

LocalHessian<4> EdgeEdgeConstraint::mollifiedBarrierHessian(const LogBarrier &barrier) const {
  Real c = crossSquaredNorm();
  Real e_x = epsCross();
  Real dist = distanceSqr();
  Real b = barrier(dist);
  Real p_b_p_d = barrier.distanceSqrGradient(dist);
  Real p2_b_p_d2 = barrier.distanceSqrHessian(dist);
  LocalGrad<4> p_d_p_x = localDistanceSqrEdgeEdgeGradient(
      x[globalEdgeA[0]], x[globalEdgeA[1]],
      x[globalEdgeB[0]], x[globalEdgeB[1]]);
  LocalGrad<4> p_b_p_x = p_b_p_d * p_d_p_x;
  LocalHessian<4> p2_b_p_x2 = p_b_p_d * localDistanceSqrEdgeEdgeHessian(
      x[globalEdgeA[0]], x[globalEdgeA[1]],
      x[globalEdgeB[0]], x[globalEdgeB[1]])
      + p2_b_p_d2 * outerProductMatrix(p_d_p_x);
  if (c > e_x) return p2_b_p_x2;
  Real m = mollifier();
  LocalGrad<4> p_m_p_x = mollifierGradient();
  LocalHessian<4> p2_m_p_x2 = mollifierHessian();
  LocalHessian<4> result{};
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // m * p2_b_p_x2 + (b * p2_m_p_x2 + outer(p_b_p_x, p_m_p_x) + outer(p_m_p_x, p_b_p_x))
      result[i][j] = m * p2_b_p_x2[i][j]
                    + glm::outerProduct(p_b_p_x[i], p_m_p_x[j])
                    + glm::outerProduct(p_m_p_x[i], p_b_p_x[j])
                    + b * p2_m_p_x2[i][j];
    }
  }
  return result;
}

void EdgeEdgeConstraint::assembleMollifiedBarrierGradient(
    const LogBarrier &barrier,
    maths::BlockVector<3> &globalGradient,
    Real kappa) const {
  Real dist = distanceSqr();
  if (dist >= barrier.dHatSqr()) return;
  auto localGrad = kappa * mollifiedBarrierGradient(barrier);
  std::array<int, 4> idx = {
      globalEdgeA[0], globalEdgeA[1],
      globalEdgeB[0], globalEdgeB[1]
  };
  assembleLocalGrad<4>(globalGradient, idx, localGrad, 1.0);
}

void EdgeEdgeConstraint::assembleMollifiedBarrierHessian(
    const LogBarrier &barrier,
    maths::BlockSparseMatrix<3> &globalHessian,
    Real kappa) const {
  Real dist = distanceSqr();
  if (dist >= barrier.dHatSqr()) return;
  auto localHess = mollifiedBarrierHessian(barrier);
  std::array<int, 4> idx = {
      globalEdgeA[0], globalEdgeA[1],
      globalEdgeB[0], globalEdgeB[1]
  };
  // mollifiedBarrierHessian 已包含完整的 mollified barrier hessian 公式
  // 直接使用简单重载组装，避免 barrier formula 被应用两次
  assembleLocalHessian<4>(globalHessian, idx, localHess, kappa);
}

} // namespace fem::ipc
