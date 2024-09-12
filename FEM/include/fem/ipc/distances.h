//
// Created by creeper on 6/2/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
#include <fem/types.h>
#include <fem/ipc/external/distances.h>
#include <cstdint>
namespace fem::ipc {
inline Real distancePointPoint(const Vector<Real, 3> &p1, const Vector<Real, 3> &p2) {
  return (p1 - p2).norm();
}
inline Vector<Real, 6> localDistancePointPointGradient(const Vector<Real, 3> &p1, const Vector<Real, 3> &p2) {
  Vector<Real, 6> local_grad;
  local_grad.segment<3>(0) = (p1 - p2).normalized();
  local_grad.segment<3>(3) = -local_grad.segment<3>(0);
  return local_grad;
}
inline Matrix<Real, 6, 6> localDistancePointPointHessian(const Vector<Real, 3> &p1, const Vector<Real, 3> &p2) {
  Matrix<Real, 6, 6> local_hessian;
  local_hessian.block<3, 3>(0, 0) = Matrix<Real, 3, 3>::Identity();
  local_hessian.block<3, 3>(3, 3) = Matrix<Real, 3, 3>::Identity();
  return local_hessian;
}
inline Real distancePointLine(const Vector<Real, 3> &p, const Vector<Real, 3> &l1, const Vector<Real, 3> &l2) {
  assert((l2 - l1).norm() != 0.0);
  return ((l2 - l1).cross(p - l1)).norm() / (l2 - l1).norm();
}
inline Vector<Real, 9> localDistancePointLineGradient(const Vector<Real, 3> &p,
                                                      const Vector<Real, 3> &l1,
                                                      const Vector<Real, 3> &l2) {
  Vector<Real, 9> local_grad;
  autogen::point_line_distance_gradient_3D(p(0),
                                           p(1),
                                           p(2),
                                           l1(0),
                                           l1(1),
                                           l1(2),
                                           l2(0),
                                           l2(1),
                                           l2(2),
                                           local_grad.data());
  return local_grad;
}
inline Matrix<Real, 9, 9> localDistancePointLineHessian(const Vector<Real, 3> &p,
                                                        const Vector<Real, 3> &l1,
                                                        const Vector<Real, 3> &l2) {
  Matrix<Real, 9, 9> local_hessian;
  autogen::point_line_distance_hessian_3D(p(0),
                                          p(1),
                                          p(2),
                                          l1(0),
                                          l1(1),
                                          l1(2),
                                          l2(0),
                                          l2(1),
                                          l2(2),
                                          local_hessian.data());
  return local_hessian;
}

inline Real distancePointPlane(const Vector<Real, 3> &p,
                               const Vector<Real, 3> &p1,
                               const Vector<Real, 3> &p2,
                               const Vector<Real, 3> &p3) {
  assert((p2 - p1).cross(p3 - p1).norm() != 0.0);
  return std::abs((p - p1).dot((p2 - p1).cross(p3 - p1))) / (p2 - p1).cross(p3 - p1).norm();
}
inline Vector<Real, 12> localDistancePointPlaneGradient(const Vector<Real, 3> &p,
                                                        const Vector<Real, 3> &p1,
                                                        const Vector<Real, 3> &p2,
                                                        const Vector<Real, 3> &p3) {
  Vector<Real, 12> local_grad;
  autogen::point_plane_distance_gradient(p(0),
                                         p(1),
                                         p(2),
                                         p1(0),
                                         p1(1),
                                         p1(2),
                                         p2(0),
                                         p2(1),
                                         p2(2),
                                         p3(0),
                                         p3(1),
                                         p3(2),
                                         local_grad.data());
  return local_grad;
}
inline Matrix<Real, 12, 12> localDistancePointPlaneHessian(const Vector<Real, 3> &p,
                                                           const Vector<Real, 3> &p1,
                                                           const Vector<Real, 3> &p2,
                                                           const Vector<Real, 3> &p3) {
  Matrix<Real, 12, 12> local_hessian;
  autogen::point_plane_distance_hessian(p(0),
                                        p(1),
                                        p(2),
                                        p1(0),
                                        p1(1),
                                        p1(2),
                                        p2(0),
                                        p2(1),
                                        p2(2),
                                        p3(0),
                                        p3(1),
                                        p3(2),
                                        local_hessian.data());
  return local_hessian;
}
inline Real distanceLineLine(const Vector<Real, 3> &l1,
                             const Vector<Real, 3> &l2,
                             const Vector<Real, 3> &m1,
                             const Vector<Real, 3> &m2) {
  Vector<Real, 3> normal = (l2 - l1).cross(m2 - m1);
  if (normal.norm() == 0.0)
    return distancePointLine(l1, m1, m2);
  Real line_to_line = std::abs((l1 - m1).dot(normal));
  return line_to_line / normal.norm();
}
inline Vector<Real, 12> localDistanceLineLineGradient(const Vector<Real, 3> &l1,
                                                      const Vector<Real, 3> &l2,
                                                      const Vector<Real, 3> &m1,
                                                      const Vector<Real, 3> &m2) {
  Vector<Real, 12> local_grad;
  autogen::line_line_distance_gradient(l1(0),
                                       l1(1),
                                       l1(2),
                                       l2(0),
                                       l2(1),
                                       l2(2),
                                       m1(0),
                                       m1(1),
                                       m1(2),
                                       m2(0),
                                       m2(1),
                                       m2(2),
                                       local_grad.data());
  return local_grad;
}
inline Matrix<Real, 12, 12> localDistanceLineLineHessian(const Vector<Real, 3> &l1,
                                                         const Vector<Real, 3> &l2,
                                                         const Vector<Real, 3> &m1,
                                                         const Vector<Real, 3> &m2) {
  Matrix<Real, 12, 12> local_hessian;
  autogen::line_line_distance_hessian(l1(0),
                                      l1(1),
                                      l1(2),
                                      l2(0),
                                      l2(1),
                                      l2(2),
                                      m1(0),
                                      m1(1),
                                      m1(2),
                                      m2(0),
                                      m2(1),
                                      m2(2),
                                      local_hessian.data());
  return local_hessian;
}

enum class EdgeEdgeDistanceType : uint8_t {
  A_C,
  A_D,
  B_C,
  B_D,
  AB_C,
  AB_D,
  A_CD,
  B_CD,
  AB_CD
};

enum class PointTriangleDistanceType : uint8_t {
  P_A,
  P_B,
  P_C,
  P_AB,
  P_BC,
  P_CA,
  P_ABC
};

// modified from ipc-toolkit
inline EdgeEdgeDistanceType decideEdgeEdgeParallelDistanceType(
    const Eigen::Vector3d& ea0,
    const Eigen::Vector3d& ea1,
    const Eigen::Vector3d& eb0,
    const Eigen::Vector3d& eb1)
{
  const Eigen::Vector3d ea = ea1 - ea0;
  const double alpha = (eb0 - ea0).dot(ea) / ea.squaredNorm();
  const double beta = (eb1 - ea0).dot(ea) / ea.squaredNorm();

  uint8_t eac; // 0: EA0, 1: EA1, 2: EA
  uint8_t ebc; // 0: EB0, 1: EB1, 2: EB
  if (alpha < 0) {
    eac = (0 <= beta && beta <= 1) ? 2 : 0;
    ebc = (beta <= alpha) ? 0 : (beta <= 1 ? 1 : 2);
  } else if (alpha > 1) {
    eac = (0 <= beta && beta <= 1) ? 2 : 1;
    ebc = (beta >= alpha) ? 0 : (0 <= beta ? 1 : 2);
  } else {
    eac = 2;
    ebc = 0;
  }

  // f(0, 0) = 0000 = 0 -> A_C
  // f(0, 1) = 0001 = 1 -> A_D
  // f(1, 0) = 0010 = 2 -> B_C
  // f(1, 1) = 0011 = 3 -> B_D
  // f(2, 0) = 0100 = 4 -> AB_C
  // f(2, 1) = 0101 = 5 -> AB_D
  // f(0, 2) = 0110 = 6 -> A_CD
  // f(1, 2) = 0111 = 7 -> B_CD
  // f(2, 2) = 1000 = 8 -> AB_CD

  assert(eac != 2 || ebc != 2); // This case results in a degenerate line-line
  return EdgeEdgeDistanceType(ebc < 2 ? (eac << 1 | ebc) : (6 + eac));
}
// modified from ipc-toolkit
inline EdgeEdgeDistanceType decideEdgeEdgeDistanceType(const Vector<Real, 3> &ea0,
                                                       const Vector<Real, 3> &ea1,
                                                       const Vector<Real, 3> &eb0,
                                                       const Vector<Real, 3> &eb1) {
  constexpr double PARALLEL_THRESHOLD = 1.0e-20;

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
    return EdgeEdgeDistanceType::A_C;
  } else if (a == 0.0) {
    return EdgeEdgeDistanceType::A_CD;
  } else if (c == 0.0) {
    return EdgeEdgeDistanceType::AB_C;
  }

  // Special handling for parallel edges
  Real parallel_tolerance = PARALLEL_THRESHOLD * std::max(1.0, a * c);
  if (u.cross(v).squaredNorm() < parallel_tolerance) {
    return decideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);
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
      return EdgeEdgeDistanceType::A_C;
    } else if (-d >= a) {
      return EdgeEdgeDistanceType::B_C;
    } else {
      return EdgeEdgeDistanceType::AB_C;
    }
  } else if (tN >= tD) { // tc > 1 ⟹ the t=1 edge is visible
    // recompute sc for this edge
    if ((-d + b) <= 0.0) {
      return EdgeEdgeDistanceType::A_D;
    } else if ((-d + b) >= a) {
      return EdgeEdgeDistanceType::B_D;
    } else {
      return EdgeEdgeDistanceType::AB_D;
    }
  }

  return default_case;

}

inline PointTriangleDistanceType decidePointTriangleDistanceType(
    const Eigen::Ref<const Eigen::Vector3d> &p,
    const Eigen::Ref<const Eigen::Vector3d> &t0,
    const Eigen::Ref<const Eigen::Vector3d> &t1,
    const Eigen::Ref<const Eigen::Vector3d> &t2) {
  const Eigen::Vector3d normal = (t1 - t0).cross(t2 - t0);

  Eigen::Matrix<Real, 2, 3> basis, param;

  basis.row(0) = t1 - t0;
  basis.row(1) = basis.row(0).cross(normal);
  param.col(0) = (basis * basis.transpose()).ldlt().solve(basis * (p - t0));
  if (param(0, 0) > 0.0 && param(0, 0) < 1.0 && param(1, 0) >= 0.0) {
    return PointTriangleDistanceType::P_AB;
  }

  basis.row(0) = t2 - t1;
  basis.row(1) = basis.row(0).cross(normal);
  param.col(1) = (basis * basis.transpose()).ldlt().solve(basis * (p - t1));
  if (param(0, 1) > 0.0 && param(0, 1) < 1.0 && param(1, 1) >= 0.0) {
    return PointTriangleDistanceType::P_BC;
  }

  basis.row(0) = t0 - t2;
  basis.row(1) = basis.row(0).cross(normal);
  param.col(2) = (basis * basis.transpose()).ldlt().solve(basis * (p - t2));
  if (param(0, 2) > 0.0 && param(0, 2) < 1.0 && param(1, 2) >= 0.0) {
    return PointTriangleDistanceType::P_CA;
  }

  if (param(0, 0) <= 0.0 && param(0, 2) >= 1.0) {
    // vertex 0 is the closest
    return PointTriangleDistanceType::P_A;
  } else if (param(0, 1) <= 0.0 && param(0, 0) >= 1.0) {
    // vertex 1 is the closest
    return PointTriangleDistanceType::P_B;
  } else if (param(0, 2) <= 0.0 && param(0, 1) >= 1.0) {
    // vertex 2 is the closest
    return PointTriangleDistanceType::P_C;
  } else {
    return PointTriangleDistanceType::P_ABC;
  }
}

Real distancePointTriangle(const Vector<Real, 3> &p,
                           const Vector<Real, 3> &a,
                           const Vector<Real, 3> &b,
                           const Vector<Real, 3> &c);

Vector<Real, 12> localDistancePointTriangleGradient(const Vector<Real, 3> &p,
                                                    const Vector<Real, 3> &a,
                                                    const Vector<Real, 3> &b,
                                                    const Vector<Real, 3> &c);

Matrix<Real, 12, 12> localDistancePointTriangleHessian(const Vector<Real, 3> &p,
                                                       const Vector<Real, 3> &a,
                                                       const Vector<Real, 3> &b,
                                                       const Vector<Real, 3> &c);

Real distanceEdgeEdge(const Vector<Real, 3> &ea0,
                      const Vector<Real, 3> &ea1,
                      const Vector<Real, 3> &eb0,
                      const Vector<Real, 3> &eb1);

Vector<Real, 12> localDistanceEdgeEdgeGradient(const Vector<Real, 3> &a,
                                               const Vector<Real, 3> &b,
                                               const Vector<Real, 3> &c,
                                               const Vector<Real, 3> &d);

Matrix<Real, 12, 12> localDistanceEdgeEdgeHessian(const Vector<Real, 3> &a,
                                                  const Vector<Real, 3> &b,
                                                  const Vector<Real, 3> &c,
                                                  const Vector<Real, 3> &d);
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
