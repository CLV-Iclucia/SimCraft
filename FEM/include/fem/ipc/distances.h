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
  return (p - p1).dot((p2 - p1).cross(p3 - p1)) / (p2 - p1).cross(p3 - p1).norm();
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
  Real line_to_line = (l1 - m1).dot(normal);
  return line_to_line * line_to_line / normal.squaredNorm();
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
  A_CD,
  B_CD,
  AB_C,
  AB_D,
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
    return edge_edge_parallel_distance_type(ea0, ea1, eb0, eb1);
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

PointTriangleDistanceType point_triangle_distance_type(
    const Eigen::Ref<const Eigen::Vector3d>& p,
    const Eigen::Ref<const Eigen::Vector3d>& t0,
    const Eigen::Ref<const Eigen::Vector3d>& t1,
    const Eigen::Ref<const Eigen::Vector3d>& t2)
{
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

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
