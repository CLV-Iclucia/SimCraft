//
// Created by creeper on 6/2/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
#include <cstdint>
#include <fem/types.h>
#include <fem/ipc/external/distances.h>
#include <Maths/block-types.h>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>

namespace sim::fem::ipc {

using maths::LocalGrad;
using maths::LocalHessian;
using maths::localGradFromFlat;
using maths::localHessianFromFlat;

// ============================================================================
// 距离函数（标量返回值）— glm::dvec3 接口
// ============================================================================

inline Real distanceSqrPointPoint(const glm::dvec3 &p1,
                                  const glm::dvec3 &p2) {
  auto d = p1 - p2;
  return glm::dot(d, d);
}

inline Real distanceSqrPointLine(const glm::dvec3 &p,
                                 const glm::dvec3 &l1,
                                 const glm::dvec3 &l2) {
  auto e = l2 - l1;
  Real e_len2 = glm::dot(e, e);
  assert(e_len2 != 0.0);
  auto c = glm::cross(e, p - l1);
  return glm::dot(c, c) / e_len2;
}

inline Real distanceSqrPointPlane(const glm::dvec3 &p,
                                  const glm::dvec3 &p1,
                                  const glm::dvec3 &p2,
                                  const glm::dvec3 &p3) {
  auto normal = glm::cross(p2 - p1, p3 - p1);
  Real n_len2 = glm::dot(normal, normal);
  assert(n_len2 != 0.0);
  Real d = glm::dot(p - p1, normal);
  return d * d / n_len2;
}

inline Real distanceSqrLineLine(const glm::dvec3 &l1,
                                const glm::dvec3 &l2,
                                const glm::dvec3 &m1,
                                const glm::dvec3 &m2) {
  auto normal = glm::cross(l2 - l1, m2 - m1);
  Real n_len2 = glm::dot(normal, normal);
  if (n_len2 == 0.0)
    return distanceSqrPointLine(l1, m1, m2);
  Real line_to_line = glm::dot(l1 - m1, normal);
  line_to_line *= line_to_line;
  return line_to_line / n_len2;
}

// ============================================================================
// 梯度函数 — 返回 LocalGrad<N>
// ============================================================================

/// Point-Point (2 vertices): grad = [2(p1-p2), -2(p1-p2)]
inline LocalGrad<2> localDistanceSqrPointPointGradient(
    const glm::dvec3 &p1, const glm::dvec3 &p2) {
  auto d = 2.0 * (p1 - p2);
  return LocalGrad<2>({d, -d});
}

/// Point-Line (3 vertices): p, l1, l2
inline LocalGrad<3> localDistanceSqrPointLineGradient(
    const glm::dvec3 &p, const glm::dvec3 &l1, const glm::dvec3 &l2) {
  double buf[9];
  autogen::point_line_distance_gradient_3D(
      p.x, p.y, p.z, l1.x, l1.y, l1.z, l2.x, l2.y, l2.z, buf);
  return localGradFromFlat<3>(buf);
}

/// Point-Plane (4 vertices): p, p1, p2, p3
inline LocalGrad<4> localDistanceSqrPointPlaneGradient(
    const glm::dvec3 &p, const glm::dvec3 &p1,
    const glm::dvec3 &p2, const glm::dvec3 &p3) {
  double buf[12];
  autogen::point_plane_distance_gradient(
      p.x, p.y, p.z, p1.x, p1.y, p1.z,
      p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, buf);
  return localGradFromFlat<4>(buf);
}

/// Line-Line (4 vertices): l1, l2, m1, m2
inline LocalGrad<4> localDistanceSqrLineLineGradient(
    const glm::dvec3 &l1, const glm::dvec3 &l2,
    const glm::dvec3 &m1, const glm::dvec3 &m2) {
  double buf[12];
  autogen::line_line_distance_gradient(
      l1.x, l1.y, l1.z, l2.x, l2.y, l2.z,
      m1.x, m1.y, m1.z, m2.x, m2.y, m2.z, buf);
  return localGradFromFlat<4>(buf);
}

// ============================================================================
// Hessian 函数 — 返回 LocalHessian<N>
// ============================================================================

/// Point-Point (2×2 blocks): H = [[2I, -2I], [-2I, 2I]]
inline LocalHessian<2> localDistanceSqrPointPointHessian(
    const glm::dvec3 &p1, const glm::dvec3 &p2) {
  auto I = glm::dmat3(2.0);   // 2 * Identity
  auto negI = glm::dmat3(-2.0);
  LocalHessian<2> H{};
  H[0][0] = I;
  H[0][1] = negI;
  H[1][0] = negI;
  H[1][1] = I;
  return H;
}

/// Point-Line (3×3 blocks)
inline LocalHessian<3> localDistanceSqrPointLineHessian(
    const glm::dvec3 &p, const glm::dvec3 &l1, const glm::dvec3 &l2) {
  double buf[81];  // 9×9 flat
  autogen::point_line_distance_hessian_3D(
      p.x, p.y, p.z, l1.x, l1.y, l1.z, l2.x, l2.y, l2.z, buf);
  return localHessianFromFlat<3>(buf);
}

/// Point-Plane (4×4 blocks)
inline LocalHessian<4> localDistanceSqrPointPlaneHessian(
    const glm::dvec3 &p, const glm::dvec3 &p1,
    const glm::dvec3 &p2, const glm::dvec3 &p3) {
  double buf[144];  // 12×12 flat
  autogen::point_plane_distance_hessian(
      p.x, p.y, p.z, p1.x, p1.y, p1.z,
      p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, buf);
  return localHessianFromFlat<4>(buf);
}

/// Line-Line (4×4 blocks)
inline LocalHessian<4> localDistanceSqrLineLineHessian(
    const glm::dvec3 &l1, const glm::dvec3 &l2,
    const glm::dvec3 &m1, const glm::dvec3 &m2) {
  double buf[144];  // 12×12 flat
  autogen::line_line_distance_hessian(
      l1.x, l1.y, l1.z, l2.x, l2.y, l2.z,
      m1.x, m1.y, m1.z, m2.x, m2.y, m2.z, buf);
  return localHessianFromFlat<4>(buf);
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
  AB_CD,
  Unknown
};

enum class PointTriangleDistanceType : uint8_t {
  P_A,
  P_B,
  P_C,
  P_AB,
  P_BC,
  P_CA,
  P_ABC,
  Unknown
};

// ============================================================================
// decideEdgeEdgeParallelDistanceType — glm::dvec3 版本
// ============================================================================
inline EdgeEdgeDistanceType decideEdgeEdgeParallelDistanceType(
    const glm::dvec3 &ea0, const glm::dvec3 &ea1,
    const glm::dvec3 &eb0, const glm::dvec3 &eb1) {
  const glm::dvec3 ea = ea1 - ea0;
  const double alpha = glm::dot(eb0 - ea0, ea) / glm::dot(ea, ea);
  const double beta  = glm::dot(eb1 - ea0, ea) / glm::dot(ea, ea);

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

  assert(eac != 2 || ebc != 2);
  return EdgeEdgeDistanceType(ebc < 2 ? (eac << 1 | ebc) : (6 + eac));
}

// ============================================================================
// decideEdgeEdgeDistanceType — glm::dvec3 版本
// ============================================================================
inline EdgeEdgeDistanceType decideEdgeEdgeDistanceType(
    const glm::dvec3 &ea0, const glm::dvec3 &ea1,
    const glm::dvec3 &eb0, const glm::dvec3 &eb1) {
  constexpr double PARALLEL_THRESHOLD = 1.0e-20;

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
    return EdgeEdgeDistanceType::A_C;
  } else if (a == 0.0) {
    return EdgeEdgeDistanceType::A_CD;
  } else if (c == 0.0) {
    return EdgeEdgeDistanceType::AB_C;
  }

  Real parallel_tolerance = PARALLEL_THRESHOLD * std::max(1.0, a * c);
  if (glm::dot(glm::cross(u, v), glm::cross(u, v)) < parallel_tolerance) {
    return decideEdgeEdgeParallelDistanceType(ea0, ea1, eb0, eb1);
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
      return EdgeEdgeDistanceType::A_C;
    } else if (-d >= a) {
      return EdgeEdgeDistanceType::B_C;
    } else {
      return EdgeEdgeDistanceType::AB_C;
    }
  } else if (tN >= tD) {
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

// ============================================================================
// decidePointTriangleDistanceType — glm::dvec3 版本（手写 2×2 求解）
// ============================================================================
inline PointTriangleDistanceType
decidePointTriangleDistanceType(const glm::dvec3 &p,
                                const glm::dvec3 &t0,
                                const glm::dvec3 &t1,
                                const glm::dvec3 &t2) {
  auto normal = glm::cross(t1 - t0, t2 - t0);

  // 对每条边做投影判断 (内联 2×2 线性求解)
  auto edgeTest = [&](const glm::dvec3 &from, const glm::dvec3 &to,
                      const glm::dvec3 &point) -> std::pair<double, double> {
    auto e = to - from;
    auto n = glm::cross(e, normal);
    // [e; n] * [s; t]^T = rhs (投影到 edge 切线+法线)
    // 2×2 系统: [[e·e, e·n],[n·e, n·n]] * [s,t] = [e·rhs, n·rhs]
    double a00 = glm::dot(e, e), a01 = glm::dot(e, n);
    double a10 = a01,            a11 = glm::dot(n, n);
    double b0 = glm::dot(e, point - from), b1 = glm::dot(n, point - from);
    double det = a00 * a11 - a01 * a10;
    return {(a11*b0 - a01*b1) / det, (a00*b1 - a10*b0) / det};
  };

  auto [s0, t0_val] = edgeTest(t0, t1, p);
  if (s0 > 0.0 && s0 < 1.0 && t0_val >= 0.0) return PointTriangleDistanceType::P_AB;

  auto [s1, t1_val] = edgeTest(t1, t2, p);
  if (s1 > 0.0 && s1 < 1.0 && t1_val >= 0.0) return PointTriangleDistanceType::P_BC;

  auto [s2, t2_val] = edgeTest(t2, t0, p);
  if (s2 > 0.0 && s2 < 1.0 && t2_val >= 0.0) return PointTriangleDistanceType::P_CA;

  if (s0 <= 0.0 && s2 >= 1.0) return PointTriangleDistanceType::P_A;
  if (s1 <= 0.0 && s0 >= 1.0) return PointTriangleDistanceType::P_B;
  if (s2 <= 0.0 && s1 >= 1.0) return PointTriangleDistanceType::P_C;

  return PointTriangleDistanceType::P_ABC;
}

// ============================================================================
// 剩余函数声明 — glm::dvec3 接口
// ============================================================================

Real distanceSqrPointTriangle(const glm::dvec3 &p,
                              const glm::dvec3 &a,
                              const glm::dvec3 &b,
                              const glm::dvec3 &c);

LocalGrad<4> localDistancePointTriangleGradient(const glm::dvec3 &p,
                                                    const glm::dvec3 &a,
                                                    const glm::dvec3 &b,
                                                    const glm::dvec3 &c);

LocalHessian<4> localDistancePointTriangleHessian(
    const glm::dvec3 &p, const glm::dvec3 &a,
    const glm::dvec3 &b, const glm::dvec3 &c);

Real distanceSqrEdgeEdge(const glm::dvec3 &ea0, const glm::dvec3 &ea1,
                         const glm::dvec3 &eb0,
                         const glm::dvec3 &eb1);

LocalGrad<4> localDistanceSqrEdgeEdgeGradient(const glm::dvec3 &a,
                                                  const glm::dvec3 &b,
                                                  const glm::dvec3 &c,
                                                  const glm::dvec3 &d);

LocalHessian<4> localDistanceSqrEdgeEdgeHessian(const glm::dvec3 &a,
                                                     const glm::dvec3 &b,
                                                     const glm::dvec3 &c,
                                                     const glm::dvec3 &d);

} // namespace sim::fem::ipc
#endif // SIMCRAFT_FEM_INCLUDE_FEM_IPC_DISTANCES_H_
