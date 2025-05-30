//
// Created by creeper on 7/19/24.
//
#include <fem/ipc/barrier-functions.h>
#include <fem/ipc/constraint.h>
#include <fem/ipc/mollifier.h>
#include <Maths/sparse-matrix-builder.h>
#include <fem/types.h>
#include <fem/system.h>

namespace sim::fem::ipc {

void VertexTriangleConstraint::updateDistanceType() {
  auto p = xv.segment<3>(iv * 3);
  auto a = xt.segment<3>(triangle.x * 3);
  auto b = xt.segment<3>(triangle.y * 3);
  auto c = xt.segment<3>(triangle.z * 3);
  type = decidePointTriangleDistanceType(p, a, b, c);
}

Real VertexTriangleConstraint::distanceSqr() const {
  auto p = xv.segment<3>(iv * 3);
  auto a = xt.segment<3>(triangle.x * 3);
  auto b = xt.segment<3>(triangle.y * 3);
  auto c = xt.segment<3>(triangle.z * 3);
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

void VertexTriangleConstraint::assembleBarrierGradient(const LogBarrier &barrier,
                                                       VecXd &globalGradient,
                                                       Real kappa) const {
  auto p = xv.segment<3>(iv * 3);
  int ia = triangle.x;
  auto a = xt.segment<3>(ia * 3);
  int ib = triangle.y;
  auto b = xt.segment<3>(ib * 3);
  int ic = triangle.z;
  auto c = xt.segment<3>(ic * 3);
  Real dist = distanceSqr();
  if (dist >= barrier.dHatSqr()) return;
  Real barrier_derivative = barrier.distanceSqrGradient(dist);
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      Vector<Real, 6> grad = kappa * barrier_derivative * localDistanceSqrPointPointGradient(p, a);
      globalGradient.segment<3>(iv * 3) += grad.segment<3>(0);
      globalGradient.segment<3>(ia * 3) += grad.segment<3>(3);
      return;
    }
    case PointTriangleDistanceType::P_B: {
      Vector<Real, 6> grad = kappa * barrier_derivative * localDistanceSqrPointPointGradient(p, b);
      globalGradient.segment<3>(iv * 3) += grad.segment<3>(0);
      globalGradient.segment<3>(ib * 3) += grad.segment<3>(3);
      return;
    }
    case PointTriangleDistanceType::P_C: {
      Vector<Real, 6> grad = kappa * barrier_derivative * localDistanceSqrPointPointGradient(p, c);
      globalGradient.segment<3>(iv * 3) += grad.segment<3>(0);
      globalGradient.segment<3>(ic * 3) += grad.segment<3>(3);
      return;
    }
    case PointTriangleDistanceType::P_AB: {
      Vector<Real, 9> grad = kappa * barrier_derivative * localDistanceSqrPointLineGradient(p, a, b);
      globalGradient.segment<3>(iv * 3) += grad.segment<3>(0);
      globalGradient.segment<3>(ia * 3) += grad.segment<3>(3);
      globalGradient.segment<3>(ib * 3) += grad.segment<3>(6);
      return;
    }
    case PointTriangleDistanceType::P_BC: {
      Vector<Real, 9> grad = kappa * barrier_derivative * localDistanceSqrPointLineGradient(p, b, c);
      globalGradient.segment<3>(iv * 3) += grad.segment<3>(0);
      globalGradient.segment<3>(ib * 3) += grad.segment<3>(3);
      globalGradient.segment<3>(ic * 3) += grad.segment<3>(6);
      return;
    }
    case PointTriangleDistanceType::P_CA: {
      Vector<Real, 9> grad = kappa * barrier_derivative * localDistanceSqrPointLineGradient(p, a, c);
      globalGradient.segment<3>(iv * 3) += grad.segment<3>(0);
      globalGradient.segment<3>(ia * 3) += grad.segment<3>(3);
      globalGradient.segment<3>(ic * 3) += grad.segment<3>(6);
      return;
    }
    case PointTriangleDistanceType::P_ABC: {
      Vector<Real, 12> grad = kappa * barrier_derivative * localDistanceSqrPointPlaneGradient(p, a, b, c);
      globalGradient.segment<3>(iv * 3) += grad.segment<3>(0);
      globalGradient.segment<3>(ia * 3) += grad.segment<3>(3);
      globalGradient.segment<3>(ib * 3) += grad.segment<3>(6);
      globalGradient.segment<3>(ic * 3) += grad.segment<3>(9);
      return;
    }
    default:throw std::runtime_error("Unknown distance type encountered when assembling edge-edge constraint gradient");
  }
}

void VertexTriangleConstraint::assembleBarrierHessian(const LogBarrier &barrier,
                                                     maths::SparseMatrixBuilder<Real> &globalHessian,
                                                     Real kappa) const {
  auto p = xv.segment<3>(iv * 3);
  int ia = triangle.x;
  auto a = xt.segment<3>(ia * 3);
  int ib = triangle.y;
  auto b = xt.segment<3>(ib * 3);
  int ic = triangle.z;
  auto c = xt.segment<3>(ic * 3);
  Real dist = distanceSqr();
  if (dist >= barrier.dHatSqr()) return;
  Real barrier_derivative = barrier.distanceSqrGradient(dist);
  Real barrier_hessian = barrier.distanceSqrHessian(dist);
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      auto local_dist_grad = localDistanceSqrPointPointGradient(p, a);
      Matrix<Real, 6, 6> local_hessian =
          kappa * barrier_derivative * localDistanceSqrPointPointHessian(p, a)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      globalHessian.assembleBlock<6, 3>(local_hessian, iv, ia);
      return;
    }
    case PointTriangleDistanceType::P_B: {
      auto local_dist_grad = localDistanceSqrPointPointGradient(p, b);
      Matrix<Real, 6, 6> local_hessian =
          kappa * barrier_derivative * localDistanceSqrPointPointHessian(p, b)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      globalHessian.assembleBlock<6, 3>(local_hessian, iv, ib);
      return;
    }
    case PointTriangleDistanceType::P_C: {
      auto local_dist_grad = localDistanceSqrPointPointGradient(p, c);
      Matrix<Real, 6, 6> local_hessian =
          kappa * barrier_derivative * localDistanceSqrPointPointHessian(p, c)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      globalHessian.assembleBlock<6, 3>(local_hessian, iv, ic);
      return;
    }
    case PointTriangleDistanceType::P_AB: {
      auto local_dist_grad = localDistanceSqrPointLineGradient(p, a, b);
      Matrix<Real, 9, 9> local_hessian =
          kappa * barrier_derivative * localDistanceSqrPointLineHessian(p, a, b)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      globalHessian.assembleBlock<9, 3>(local_hessian, iv, ia, ib);
      return;
    }
    case PointTriangleDistanceType::P_BC: {
      auto local_dist_grad = localDistanceSqrPointLineGradient(p, b, c);
      Matrix<Real, 9, 9> local_hessian =
          kappa * barrier_derivative * localDistanceSqrPointLineHessian(p, b, c)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      globalHessian.assembleBlock<9, 3>(local_hessian, iv, ib, ic);
      return;
    }
    case PointTriangleDistanceType::P_CA: {
      auto local_dist_grad = localDistanceSqrPointLineGradient(p, a, c);
      Matrix<Real, 9, 9> local_hessian =
          kappa * barrier_derivative * localDistanceSqrPointLineHessian(p, a, c)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      globalHessian.assembleBlock<9, 3>(local_hessian, iv, ia, ic);
      return;
    }
    case PointTriangleDistanceType::P_ABC: {
      auto local_dist_grad = localDistanceSqrPointPlaneGradient(p, a, b, c);
      Matrix<Real, 12, 12> local_hessian =
          kappa * barrier_derivative * localDistanceSqrPointPlaneHessian(p, a, b, c)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      globalHessian.assembleBlock<12, 3>(local_hessian, iv, ia, ib, ic);
      return;
    }
    default:
      throw std::runtime_error("Unknown distance type encountered when assembling vertex-triangle constraint hessian");
  }
}

Vector<Real, 12> VertexTriangleConstraint::localBarrierGradient(const LogBarrier& barrier, Real kappa) const {
  Vector<Real, 12> grad = Vector<Real, 12>::Zero();
  auto p = xv.segment<3>(iv * 3);
  int ia = triangle.x;
  auto a = xt.segment<3>(ia * 3);
  int ib = triangle.y;
  auto b = xt.segment<3>(ib * 3);
  int ic = triangle.z;
  auto c = xt.segment<3>(ic * 3);
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      grad.segment<3>(0) = localDistanceSqrPointPointGradient(p, a).segment<3>(0);
      grad.segment<3>(3) = localDistanceSqrPointPointGradient(p, a).segment<3>(3);
      break;
    }
    case PointTriangleDistanceType::P_B: {
      grad.segment<3>(0) = localDistanceSqrPointPointGradient(p, b).segment<3>(0);
      grad.segment<3>(6) = localDistanceSqrPointPointGradient(p, b).segment<3>(3);
      break;
    }
    case PointTriangleDistanceType::P_C: {
      grad.segment<3>(0) = localDistanceSqrPointPointGradient(p, c).segment<3>(0);
      grad.segment<3>(9) = localDistanceSqrPointPointGradient(p, c).segment<3>(3);
      break;
    }
    case PointTriangleDistanceType::P_AB: {
      grad.segment<3>(0) = localDistanceSqrPointLineGradient(p, a, b).segment<3>(0);
      grad.segment<3>(3) = localDistanceSqrPointLineGradient(p, a, b).segment<3>(3);
      grad.segment<3>(6) = localDistanceSqrPointLineGradient(p, a, b).segment<3>(6);
      break;
    }
    case PointTriangleDistanceType::P_BC: {
      grad.segment<3>(0) = localDistanceSqrPointLineGradient(p, b, c).segment<3>(0);
      grad.segment<3>(6) = localDistanceSqrPointLineGradient(p, b, c).segment<3>(3);
      grad.segment<3>(9) = localDistanceSqrPointLineGradient(p, b, c).segment<3>(6);
      break;
    }
    case PointTriangleDistanceType::P_CA: {
      grad.segment<3>(0) = localDistanceSqrPointLineGradient(p, a, c).segment<3>(0);
      grad.segment<3>(3) = localDistanceSqrPointLineGradient(p, a, c).segment<3>(3);
      grad.segment<3>(9) = localDistanceSqrPointLineGradient(p, a, c).segment<3>(6);
      break;
    }
    case PointTriangleDistanceType::P_ABC: {
      grad.segment<3>(0) = localDistanceSqrPointPlaneGradient(p, a, b, c).segment<3>(0);
      grad.segment<3>(3) = localDistanceSqrPointPlaneGradient(p, a, b, c).segment<3>(3);
      grad.segment<3>(6) = localDistanceSqrPointPlaneGradient(p, a, b, c).segment<3>(6);
      grad.segment<3>(9) = localDistanceSqrPointPlaneGradient(p, a, b, c).segment<3>(9);
      break;
    }
    default:
      throw std::runtime_error("Unknown distance type encountered when computing vertex-triangle constraint gradient");
  }
  if (distanceSqr() >= barrier.dHatSqr()) return Vector<Real, 12>::Zero();
  Real barrier_derivative = barrier.distanceSqrGradient(distanceSqr());
  return kappa * barrier_derivative * grad;
}

void EdgeEdgeConstraint::updateDistanceType() {
  constexpr double PARALLEL_THRESHOLD = 1.0e-20;
  auto ea0 = xa.segment<3>(ea.x * 3);
  auto ea1 = xa.segment<3>(ea.y * 3);
  auto eb0 = xb.segment<3>(eb.x * 3);
  auto eb1 = xb.segment<3>(eb.y * 3);
  const Eigen::Vector3d u = ea1 - ea0;
  const Eigen::Vector3d v = eb1 - eb0;
  const Eigen::Vector3d w = ea0 - eb0;

  Real a = u.squaredNorm(); // always ≥ 0
  Real b = u.dot(v);
  Real c = v.squaredNorm(); // always ≥ 0
  Real d = u.dot(w);
  Real e = v.dot(w);
  Real D = a * c - b * b; // always ≥ 0

  // Degenerate cases should not happen in practice, but we handle them
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

  // Special handling for parallel edges
  Real parallel_tolerance = PARALLEL_THRESHOLD * std::max(1.0, a * c);
  if (u.cross(v).squaredNorm() < parallel_tolerance) {
    type = decideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);
  }

  EdgeEdgeDistanceType default_case = EdgeEdgeDistanceType::AB_CD;

  // compute the line parameters of the two closest points
  Real sN = (b * e - c * d);
  double tN, tD;   // tc = tN / tD
  if (sN <= 0.0) { // sc < 0 ⟹ the s=0 edge is visible
    tN = e;
    tD = c;
    default_case = EdgeEdgeDistanceType::A_CD;
  } else if (sN >= D) { // sc > 1 ⟹ the s=1 edge is visible
    tN = e + b;
    tD = c;
    default_case = EdgeEdgeDistanceType::B_CD;
  } else {
    tN = (a * e - b * d);
    tD = D; // default tD = D ≥ 0
    if (tN > 0.0 && tN < tD
        && u.cross(v).squaredNorm() < parallel_tolerance) {
      // avoid coplanar or nearly parallel EE
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
    // else default_case stays EdgeEdgeDistanceType::EA_EB
  }

  if (tN <= 0.0) { // tc < 0 ⟹ the t=0 edge is visible
    // recompute sc for this edge
    if (-d <= 0.0) {
      type = EdgeEdgeDistanceType::A_C;
    } else if (-d >= a) {
      type = EdgeEdgeDistanceType::B_C;
    } else {
      type = EdgeEdgeDistanceType::AB_C;
    }
  } else if (tN >= tD) { // tc > 1 ⟹ the t=1 edge is visible
    // recompute sc for this edge
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
  auto A = xa.segment<3>(ea.x * 3);
  auto B = xa.segment<3>(ea.y * 3);
  auto C = xb.segment<3>(eb.x * 3);
  auto D = xb.segment<3>(eb.y * 3);
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
    default:throw std::runtime_error("Unknown distance type encountered when computing edge-edge distance");
  }
}

void EdgeEdgeConstraint::assembleMollifiedBarrierGradient(const LogBarrier &barrier,
                                                          VecXd &globalGradient,
                                                          Real kappa) const {
  int iA = ea.x;
  int iB = ea.y;
  int iC = eb.x;
  int iD = eb.y;
  Real dist = distanceSqr();
  if (dist >= barrier.dHatSqr()) return;
  auto localGradient = mollifiedBarrierGradient(barrier);
  globalGradient.segment<3>(iA * 3) += kappa * localGradient.segment<3>(0);
  globalGradient.segment<3>(iB * 3) += kappa * localGradient.segment<3>(3);
  globalGradient.segment<3>(iC * 3) += kappa * localGradient.segment<3>(6);
  globalGradient.segment<3>(iD * 3) += kappa * localGradient.segment<3>(9);
}

void EdgeEdgeConstraint::assembleMollifiedBarrierHessian(const LogBarrier &barrier,
                                                         maths::SparseMatrixBuilder<Real> &globalHessian,
                                                         Real kappa) const {
  int iA = ea.x;
  int iB = ea.y;
  int iC = eb.x;
  int iD = eb.y;
  Real dist = distanceSqr();
  if (dist >= barrier.dHatSqr()) return;
  auto localHessian = mollifiedBarrierHessian(barrier);
  globalHessian.assembleBlock<12, 3>(kappa * localHessian, iA, iB, iC, iD);
}

Real EdgeEdgeConstraint::epsCross() const {
  return 1e-3 * (Xa.segment<3>(ea.y * 3) - Xa.segment<3>(ea.x * 3)).squaredNorm()
      * (Xb.segment<3>(eb.y * 3) - Xb.segment<3>(eb.x * 3)).squaredNorm();
}

Real EdgeEdgeConstraint::crossSquaredNorm() const {
  return (xa.segment<3>(ea.y * 3) - xa.segment<3>(ea.x * 3))
      .cross(xb.segment<3>(eb.y * 3) - xb.segment<3>(eb.x * 3))
      .squaredNorm();
}

Real EdgeEdgeConstraint::mollifier() const {
  Real e_x = 1e-3 * epsCross();
  Real c = crossSquaredNorm();
  return mollifier(c, e_x);
}

Vector<Real, 12> EdgeEdgeConstraint::mollifierGradient() const {
  Real e_x = 1e-3 * epsCross();
  Real c = crossSquaredNorm();
  if (c < e_x) return Vector<Real, 12>::Zero();
  Real p_m_p_c = mollifierDerivative(c, e_x);
  auto p_c_p_x = crossedNormGradient();
  return p_m_p_c * p_c_p_x;
}

Matrix<Real, 12, 12> EdgeEdgeConstraint::mollifierHessian() const {
  Real e_x = epsCross();
  Real c = crossSquaredNorm();
  if (c < e_x) return Matrix<Real, 12, 12>::Zero();
  Real p_m_p_c = mollifierDerivative(c, e_x);
  Vector<Real, 12> p_c_p_x = crossedNormGradient();
  Matrix<Real, 12, 12> p2_c_p_x2 = crossedNormHessian();
  Real p2_m_p_c2 = mollifierSecondDerivative(c, e_x);
  return p2_m_p_c2 * p_c_p_x * p_c_p_x.transpose() + p_m_p_c * p2_c_p_x2;
}

Vector<Real, 12> EdgeEdgeConstraint::crossedNormGradient() const {
  const auto &ea0 = xa.segment<3>(ea.x * 3);
  const auto &ea1 = xa.segment<3>(ea.y * 3);
  const auto &eb0 = xb.segment<3>(eb.x * 3);
  const auto &eb1 = xb.segment<3>(eb.y * 3);
  return edgeEdgeCrossSquareNormGradient(ea0, ea1, eb0, eb1);
}

Matrix<Real, 12, 12> EdgeEdgeConstraint::crossedNormHessian() const {
  const auto &ea0 = xa.segment<3>(ea.x * 3);
  const auto &ea1 = xa.segment<3>(ea.y * 3);
  const auto &eb0 = xb.segment<3>(eb.x * 3);
  const auto &eb1 = xb.segment<3>(eb.y * 3);
  return edgeEdgeCrossSquaredNormHessian(ea0, ea1, eb0, eb1);
}

Vector<Real, 12> EdgeEdgeConstraint::mollifiedBarrierGradient(const LogBarrier &barrier) const {
  Real c = crossSquaredNorm();
  Real e_x = epsCross();
  Real dist = distanceSqr();
  Real b = barrier(dist);
  Real p_b_p_d = barrier.distanceSqrGradient(dist);
  Vector<Real, 12> p_b_p_x = p_b_p_d * localDistanceSqrEdgeEdgeGradient(
      xa.segment<3>(ea.x * 3),
      xa.segment<3>(ea.y * 3),
      xb.segment<3>(eb.x * 3),
      xb.segment<3>(eb.y * 3));
  if (c > e_x) return p_b_p_x;
  Real m = mollifier();
  Vector<Real, 12> p_m_p_x = mollifierGradient();
  return m * p_b_p_x + b * p_m_p_x;
}

Matrix<Real, 12, 12> EdgeEdgeConstraint::mollifiedBarrierHessian(const LogBarrier &barrier) const {
  Real c = crossSquaredNorm();
  Real e_x = epsCross();
  Real dist = distanceSqr();
  Real b = barrier(dist);
  Real p_b_p_d = barrier.distanceSqrGradient(dist);
  Real p2_b_p_d2 = barrier.distanceSqrHessian(dist);
  Vector<Real, 12> p_d_p_x = localDistanceSqrEdgeEdgeGradient(
      xa.segment<3>(ea.x * 3),
      xa.segment<3>(ea.y * 3),
      xb.segment<3>(eb.x * 3),
      xb.segment<3>(eb.y * 3));
  Vector<Real, 12> p_b_p_x = p_b_p_d * p_d_p_x;
  Matrix<Real, 12, 12> p2_b_p_x2 = p_b_p_d * localDistanceSqrEdgeEdgeHessian(
      xa.segment<3>(ea.x * 3),
      xa.segment<3>(ea.y * 3),
      xb.segment<3>(eb.x * 3),
      xb.segment<3>(eb.y * 3)) + p2_b_p_d2 * p_d_p_x * p_d_p_x.transpose();
  if (c > e_x) return p2_b_p_x2;
  Real m = mollifier();
  Vector<Real, 12> p_m_p_x = mollifierGradient();
  Matrix<Real, 12, 12> p2_m_p_x2 = mollifierHessian();
  return m * p2_b_p_x2 + p_b_p_x * p_m_p_x.transpose() + p_m_p_x * p_b_p_x.transpose() + b * p2_m_p_x2;
}

}// namespace fem::ipc