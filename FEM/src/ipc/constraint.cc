//
// Created by creeper on 7/19/24.
//
#include <fem/ipc/barrier-functions.h>
#include <fem/ipc/constraint.h>
#include <fem/ipc/mollifier.h>
#include <Maths/sparse-matrix-builder.h>
#include <fem/types.h>
#include <fem/system.h>
namespace fem::ipc {

void VertexTriangleConstraint::updateDistanceType() {
  auto p = system.currentPos(iv);
  auto a = system.currentPos(system.triangleVertexIndex(it, 0));
  auto b = system.currentPos(system.triangleVertexIndex(it, 1));
  auto c = system.currentPos(system.triangleVertexIndex(it, 2));
  type = decidePointTriangleDistanceType(p, a, b, c);
}

Real VertexTriangleConstraint::distance() const {
  auto p = system.currentPos(iv);
  auto a = system.currentPos(system.triangleVertexIndex(it, 0));
  auto b = system.currentPos(system.triangleVertexIndex(it, 1));
  auto c = system.currentPos(system.triangleVertexIndex(it, 2));
  switch (type) {
    case PointTriangleDistanceType::P_A:return distancePointPoint(p, a);
    case PointTriangleDistanceType::P_B:return distancePointPoint(p, b);
    case PointTriangleDistanceType::P_C:return distancePointPoint(p, c);
    case PointTriangleDistanceType::P_AB:return distancePointLine(p, a, b);
    case PointTriangleDistanceType::P_BC:return distancePointLine(p, b, c);
    case PointTriangleDistanceType::P_CA:return distancePointLine(p, c, a);
    case PointTriangleDistanceType::P_ABC:return distancePointPlane(p, a, b, c);
    default:
      throw std::runtime_error("Unknown distance type");
  }
}

void VertexTriangleConstraint::assembleBarrierGradient(const LogBarrier &barrier,
                                                       VecXd &global_gradient,
                                                       Real kappa) const {
  auto p = system.currentPos(iv);
  int ia = system.triangleVertexIndex(it, 0);
  auto a = system.currentPos(ia);
  int ib = system.triangleVertexIndex(it, 1);
  auto b = system.currentPos(ib);
  int ic = system.triangleVertexIndex(it, 2);
  auto c = system.currentPos(ic);
  Real dist = distance();
  if (dist >= barrier.d_hat) return;
  Real barrier_derivative = barrier.distanceGradient(dist);
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      Vector<Real, 6> local_grad = kappa * barrier_derivative * localDistancePointPointGradient(p, a);
      global_gradient.segment<3>(iv * 3) += local_grad.segment<3>(0);
      global_gradient.segment<3>(ia * 3) += local_grad.segment<3>(3);
    }
    case PointTriangleDistanceType::P_B: {
      Vector<Real, 6> grad = kappa * barrier_derivative * localDistancePointPointGradient(p, b);
      global_gradient.segment<3>(iv * 3) += grad.segment<3>(0);
      global_gradient.segment<3>(ib * 3) += grad.segment<3>(3);
    }
    case PointTriangleDistanceType::P_C: {
      Vector<Real, 6> grad = kappa * barrier_derivative * localDistancePointPointGradient(p, c);
      global_gradient.segment<3>(iv * 3) += grad.segment<3>(0);
      global_gradient.segment<3>(ic * 3) += grad.segment<3>(3);
    }
    case PointTriangleDistanceType::P_AB: {
      Vector<Real, 9> grad = kappa * barrier_derivative * localDistancePointLineGradient(p, a, b);
      global_gradient.segment<3>(iv * 3) += grad.segment<3>(0);
      global_gradient.segment<3>(ia * 3) += grad.segment<3>(3);
      global_gradient.segment<3>(ib * 3) += grad.segment<3>(6);
    }
    case PointTriangleDistanceType::P_BC: {
      Vector<Real, 9> grad = kappa * barrier_derivative * localDistancePointLineGradient(p, b, c);
      global_gradient.segment<3>(iv * 3) += grad.segment<3>(0);
      global_gradient.segment<3>(ib * 3) += grad.segment<3>(3);
      global_gradient.segment<3>(ic * 3) += grad.segment<3>(6);
    }
    case PointTriangleDistanceType::P_CA: {
      Vector<Real, 9> grad = kappa * barrier_derivative * localDistancePointLineGradient(p, a, c);
      global_gradient.segment<3>(iv * 3) += grad.segment<3>(0);
      global_gradient.segment<3>(ia * 3) += grad.segment<3>(3);
      global_gradient.segment<3>(ic * 3) += grad.segment<3>(6);
    }
    case PointTriangleDistanceType::P_ABC: {
      Vector<Real, 12> grad = kappa * barrier_derivative * localDistancePointPlaneGradient(p, a, b, c);
      global_gradient.segment<3>(iv * 3) += grad.segment<3>(0);
      global_gradient.segment<3>(ia * 3) += grad.segment<3>(3);
      global_gradient.segment<3>(ib * 3) += grad.segment<3>(6);
      global_gradient.segment<3>(ic * 3) += grad.segment<3>(9);
    }
    default:
      throw std::runtime_error("Unknown distance type");
  }
}

void VertexTriangleConstraint::assembleBarrierHessian(const LogBarrier &barrier,
                                                      maths::SparseMatrixBuilder<Real> &global_hessian,
                                                      Real kappa) const {
  auto p = system.currentPos(iv);
  int ia = system.triangleVertexIndex(it, 0);
  auto a = system.currentPos(ia);
  int ib = system.triangleVertexIndex(it, 1);
  auto b = system.currentPos(ib);
  int ic = system.triangleVertexIndex(it, 2);
  auto c = system.currentPos(ic);
  Real dist = distance();
  if (dist >= barrier.d_hat) return;
  Real barrier_derivative = barrier.distanceGradient(dist);
  Real barrier_hessian = barrier.distanceHessian(dist);
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      auto local_dist_grad = localDistancePointPointGradient(p, a);
      Matrix<Real, 6, 6> local_hessian =
          kappa * barrier_derivative * localDistancePointPointHessian(p, a)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      global_hessian.assembleBlock<6, 3>(local_hessian, iv, ia);
    }
    case PointTriangleDistanceType::P_B: {
      auto local_dist_grad = localDistancePointPointGradient(p, b);
      Matrix<Real, 6, 6> local_hessian =
          kappa * barrier_derivative * localDistancePointPointHessian(p, b)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      global_hessian.assembleBlock<6, 3>(local_hessian, iv, ib);
    }
    case PointTriangleDistanceType::P_C: {
      auto local_dist_grad = localDistancePointPointGradient(p, c);
      Matrix<Real, 6, 6> local_hessian =
          kappa * barrier_derivative * localDistancePointPointHessian(p, c)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      global_hessian.assembleBlock<6, 3>(local_hessian, iv, ic);
    }
    case PointTriangleDistanceType::P_AB: {
      auto local_dist_grad = localDistancePointLineGradient(p, a, b);
      Matrix<Real, 9, 9> local_hessian =
          kappa * barrier_derivative * localDistancePointLineHessian(p, a, b)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      global_hessian.assembleBlock<9, 3>(local_hessian, iv, ia, ib);
    }
    case PointTriangleDistanceType::P_BC: {
      auto local_dist_grad = localDistancePointLineGradient(p, b, c);
      Matrix<Real, 9, 9> local_hessian =
          kappa * barrier_derivative * localDistancePointLineHessian(p, b, c)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      global_hessian.assembleBlock<9, 3>(local_hessian, iv, ib, ic);
    }
    case PointTriangleDistanceType::P_CA: {
      auto local_dist_grad = localDistancePointLineGradient(p, a, c);
      Matrix<Real, 9, 9> local_hessian =
          kappa * barrier_derivative * localDistancePointLineHessian(p, a, c)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      global_hessian.assembleBlock<9, 3>(local_hessian, iv, ia, ic);
    }
    case PointTriangleDistanceType::P_ABC: {
      auto local_dist_grad = localDistancePointPlaneGradient(p, a, b, c);
      Matrix<Real, 12, 12> local_hessian =
          kappa * barrier_derivative * localDistancePointPlaneHessian(p, a, b, c)
              + kappa * barrier_hessian * local_dist_grad * local_dist_grad.transpose();
      global_hessian.assembleBlock<12, 3>(local_hessian, iv, ia, ib, ic);
    }
    default:
      throw std::runtime_error("Unknown distance type");
  }
}

void EdgeEdgeConstraint::updateDistanceType() {
  constexpr double PARALLEL_THRESHOLD = 1.0e-20;
  auto ea0 = system.currentPos(system.edgeVertexIndex(ia, 0));
  auto ea1 = system.currentPos(system.edgeVertexIndex(ia, 1));
  auto eb0 = system.currentPos(system.edgeVertexIndex(ib, 0));
  auto eb1 = system.currentPos(system.edgeVertexIndex(ib, 1));
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

Real EdgeEdgeConstraint::distance() const {
  auto A = system.currentPos(system.edgeVertexIndex(ia, 0));
  auto B = system.currentPos(system.edgeVertexIndex(ia, 1));
  auto C = system.currentPos(system.edgeVertexIndex(ib, 0));
  auto D = system.currentPos(system.edgeVertexIndex(ib, 1));
  switch (type) {
    case EdgeEdgeDistanceType::A_C:return distancePointPoint(A, C);
    case EdgeEdgeDistanceType::A_D:return distancePointPoint(A, D);
    case EdgeEdgeDistanceType::B_C:return distancePointPoint(B, C);
    case EdgeEdgeDistanceType::B_D:return distancePointPoint(B, D);
    case EdgeEdgeDistanceType::AB_C:return distancePointLine(A, B, C);
    case EdgeEdgeDistanceType::AB_D:return distancePointLine(A, B, D);
    case EdgeEdgeDistanceType::A_CD:return distancePointLine(A, C, D);
    case EdgeEdgeDistanceType::B_CD:return distancePointLine(B, C, D);
    case EdgeEdgeDistanceType::AB_CD:return distanceLineLine(A, B, C, D);
    default:
      throw std::runtime_error("Unknown distance type");
  }
}

void EdgeEdgeConstraint::assembleMollifiedBarrierGradient(const LogBarrier &barrier,
                                                          VecXd &globalGradient,
                                                          Real kappa) const {
  int iA = system.edgeVertexIndex(ia, 0);
  int iB = system.edgeVertexIndex(ia, 1);
  int iC = system.edgeVertexIndex(ib, 0);
  int iD = system.edgeVertexIndex(ib, 1);
  Real dist = distance();
  if (dist >= barrier.d_hat) return;
  auto localGradient = mollifiedBarrierGradient(barrier);
  globalGradient.segment<3>(iA * 3) += kappa * localGradient.segment<3>(0);
  globalGradient.segment<3>(iB * 3) += kappa * localGradient.segment<3>(3);
  globalGradient.segment<3>(iC * 3) += kappa * localGradient.segment<3>(6);
  globalGradient.segment<3>(iD * 3) += kappa * localGradient.segment<3>(9);
}

void EdgeEdgeConstraint::assembleMollifiedBarrierHessian(const LogBarrier &barrier,
                                                         maths::SparseMatrixBuilder<Real> &globalHessian,
                                                         Real kappa) const {
  int iA = system.edgeVertexIndex(ia, 0);
  int iB = system.edgeVertexIndex(ia, 1);
  int iC = system.edgeVertexIndex(ib, 0);
  int iD = system.edgeVertexIndex(ib, 1);
  Real dist = distance();
  if (dist >= barrier.d_hat) return;
  auto localHessian = mollifiedBarrierHessian(barrier);
  globalHessian.assembleBlock<12, 3>(kappa * localHessian, iA, iB, iC, iD);
}

Real EdgeEdgeConstraint::epsCross() const {
  return 1e-3 * (system.referencePos(system.edgeVertexIndex(ia, 1))
      - system.referencePos(system.edgeVertexIndex(ia, 0))).squaredNorm()
      * (system.referencePos(system.edgeVertexIndex(ib, 1))
          - system.referencePos(system.edgeVertexIndex(ib, 0))).squaredNorm();
}

Real EdgeEdgeConstraint::crossSquaredNorm() const {
  return (system.currentPos(system.edgeVertexIndex(ia, 1)) - system.currentPos(system.edgeVertexIndex(ia, 0)))
      .cross(system.currentPos(system.edgeVertexIndex(ib, 1)) - system.currentPos(system.edgeVertexIndex(ib, 0)))
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
  const auto &ea0 = system.currentPos(system.edgeVertexIndex(ia, 0));
  const auto &ea1 = system.currentPos(system.edgeVertexIndex(ia, 1));
  const auto &eb0 = system.currentPos(system.edgeVertexIndex(ib, 0));
  const auto &eb1 = system.currentPos(system.edgeVertexIndex(ib, 1));
  return edgeEdgeCrossSquareNormGradient(ea0, ea1, eb0, eb1);
}

Matrix<Real, 12, 12> EdgeEdgeConstraint::crossedNormHessian() const {
  const auto &ea0 = system.currentPos(system.edgeVertexIndex(ia, 0));
  const auto &ea1 = system.currentPos(system.edgeVertexIndex(ia, 1));
  const auto &eb0 = system.currentPos(system.edgeVertexIndex(ib, 0));
  const auto &eb1 = system.currentPos(system.edgeVertexIndex(ib, 1));
  return edgeEdgeCrossSquaredNormHessian(ea0, ea1, eb0, eb1);
}

Vector<Real, 12> EdgeEdgeConstraint::mollifiedBarrierGradient(const LogBarrier &barrier) const {
  Real c = crossSquaredNorm();
  Real e_x = epsCross();
  Real dist = distance();
  Real b = barrier(dist);
  Real p_b_p_d = barrier.distanceGradient(dist);
  Vector<Real, 12> p_b_p_x = p_b_p_d * localDistanceEdgeEdgeGradient(
      system.currentPos(system.edgeVertexIndex(ia, 0)),
      system.currentPos(system.edgeVertexIndex(ia, 1)),
      system.currentPos(system.edgeVertexIndex(ib, 0)),
      system.currentPos(system.edgeVertexIndex(ib, 1)));
  if (c > e_x) return p_b_p_x;
  Real m = mollifier();
  Vector<Real, 12> p_m_p_x = mollifierGradient();
  return m * p_b_p_x + b * p_m_p_x;
}

Matrix<Real, 12, 12> EdgeEdgeConstraint::mollifiedBarrierHessian(const LogBarrier &barrier) const {
  Real c = crossSquaredNorm();
  Real e_x = epsCross();
  Real dist = distance();
  Real b = barrier(dist);
  Real p_b_p_d = barrier.distanceGradient(dist);
  Real p2_b_p_d2 = barrier.distanceHessian(dist);
  Vector<Real, 12> p_d_p_x = localDistanceEdgeEdgeGradient(
      system.currentPos(system.edgeVertexIndex(ia, 0)),
      system.currentPos(system.edgeVertexIndex(ia, 1)),
      system.currentPos(system.edgeVertexIndex(ib, 0)),
      system.currentPos(system.edgeVertexIndex(ib, 1)));
  Vector<Real, 12> p_b_p_x = p_b_p_d * p_d_p_x;
  Matrix<Real, 12, 12> p2_b_p_x2 = p_b_p_d * localDistanceEdgeEdgeHessian(
      system.currentPos(system.edgeVertexIndex(ia, 0)),
      system.currentPos(system.edgeVertexIndex(ia, 1)),
      system.currentPos(system.edgeVertexIndex(ib, 0)),
      system.currentPos(system.edgeVertexIndex(ib, 1))) + p2_b_p_d2 * p_d_p_x * p_d_p_x.transpose();
  if (c > e_x) return p2_b_p_x2;
  Real m = mollifier();
  Vector<Real, 12> p_m_p_x = mollifierGradient();
  Matrix<Real, 12, 12> p2_m_p_x2 = mollifierHessian();
  return m * p2_b_p_x2 + p_b_p_x * p_m_p_x.transpose() + p_m_p_x * p_b_p_x.transpose() + b * p2_m_p_x2;
}

}// namespace fem::ipc