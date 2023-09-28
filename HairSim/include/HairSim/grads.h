//
// Created by creeper on 23-8-10.
//

#ifndef SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_GRADS_H_
#define SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_GRADS_H_

#include <HairSim/hair.h>
#include <HairSim/math-utils.h>
#include <cassert>
#include <cmath>
namespace hairsim {
inline Vec3d p_m_p_e(const Hair &hair, Index i, Index j) {
  assert(abs(i - j) <= 1);
  if (j == i - 1 || j == i)
    return 0.5 * hair.curvatureBinormal(i) / hair.edgeLength(j);
  else
    return {};
}
inline Vec3d p_invlen_p_e(const Hair &hair, Index i, Index j) {
  assert(abs(i - j) <= 1);
  if (j == i)
    return -hair.edge(j) / cube(hair.edgeLength(j));
  else
    return {};
}
inline Vec3d p_kappa1_p_e(const Hair &hair, Index i, Index j, Index k) {
  assert(j == i + 1 || j == i);
  assert(j == k + 1 || j == k);
  Real sgn = j == k ? -1.0 : 1.0;
  Vec3d t_v = j == k ? hair.vertexTangent(j - 1) : hair.vertexTangent(j);
  return 1.0 / hair.edgeLength(k) *
         (-hair.kappa1(i, j) * hair.tangentTilde(j) +
          2.0 * sgn * t_v.cross(hair.m2(i)) /
              (1.0 + hair.tangent(j - 1).dot(hair.tangent(j))));
}
inline Vec3d p_kappa2_p_e(const Hair &hair, Index i, Index j, Index k) {
  assert(j == i + 1 || j == i);
  assert(j == k + 1 || j == k);
  Real sgn = j == k ? 1.0 : -1.0;
  Vec3d t_v = j == k ? hair.vertexTangent(j - 1) : hair.vertexTangent(j);
  return 1.0 / hair.edgeLength(k) *
         (-hair.kappa2(i, j) * hair.tangentTilde(j) +
          2.0 * sgn * t_v.cross(hair.m1(i)) /
              (1.0 + hair.tangent(j - 1).dot(hair.tangent(j))));
}
inline Mat43d p_curv_p_e(const Hair &hair, Index i, Index j) {
  return Mat43d(p_kappa1_p_e(hair, i - 1, i, j).transpose(),
                p_kappa1_p_e(hair, i, i, j).transpose(),
                p_kappa2_p_e(hair, i - 1, i, j).transpose(),
                p_kappa2_p_e(hair, i, i, j).transpose());
}
inline Real p_Et_p_m(const Hair &hair, Index i) {
  return hair.G() * hair.area() / (4 * hair.vertexReferenceLength(i)) *
         hair.radius() * hair.radius() * hair.m(i);
}
inline Vec3d p_Et_p_e(const Hair &hair, Index i) {
  return p_Et_p_m(hair, i) * p_m_p_e(hair, i, i) +
         p_Et_p_m(hair, i + 1) * p_m_p_e(hair, i + 1, i);
}
inline Mat3d p_kb_p_e(const Hair &hair, Index i, Index j) {
  assert(j == i - 1 || j == i || j == i + 1);
  if (j == i)
    return -2.0 / hair.edgeLength(i) *
               tensorProduct(
                   hair.curvatureBinormal(i) /
                       (1.0 + hair.tangent(i - 1).dot(hair.tangent(i))),
                   hair.vertexTangent(i)) +
           2.0 * skewt(hair.tangent(i - 1)) /
               (hair.edgeLength(i) *
                (1.0 + hair.tangent(i - 1).dot(hair.tangent(i))));
  else if (j == i - 1)
    return -2.0 / hair.edgeLength(i) *
               tensorProduct(hair.curvatureBinormal(i),
                             hair.tangent(i - 1) + hair.tangent(i)) /
               (hair.edgeLength(i - 1) *
                (1.0 + hair.tangent(i - 1).dot(hair.tangent(i)))) +
           2.0 * skewt(hair.tangent(i)) /
               (hair.edgeLength(i - 1) * hair.edgeLength(i) +
                hair.edge(i - 1).dot(hair.edge(i)));
  else if (j == i + 1)
    return -p_kb_p_e(hair, i, i) - p_kb_p_e(hair, i, i - 1);
}
inline Mat3d p_m_p_e_p_e(const Hair &hair, Index i, Index j, Index k) {
  return 0.5 * p_kb_p_e(hair, i, j) / hair.edgeLength(k) +
         0.5 *
             tensorProduct(hair.curvatureBinormal(i), p_invlen_p_e(hair, k, j));
}
} // namespace hairsim

// #endif // SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_GRADS_H_