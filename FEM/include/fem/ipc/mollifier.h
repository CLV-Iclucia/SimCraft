//
// Created by creeper on 9/7/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_MOLLIFIER_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_MOLLIFIER_H_
#include <fem/types.h>
#include <fem/ipc/external/mollifier.h>
namespace sim::fem::ipc {
inline Vector<Real, 12> edgeEdgeCrossSquareNormGradient(
    const Vector<Real, 3> &ea0,
    const Vector<Real, 3> &ea1,
    const Vector<Real, 3> &eb0,
    const Vector<Real, 3> &eb1) {
  Vector<Real, 12> g;
  autogen::edge_edge_cross_squarednorm_gradient(
      ea0[0], ea0[1], ea0[2],
      ea1[0], ea1[1], ea1[2],
      eb0[0], eb0[1], eb0[2],
      eb1[0], eb1[1], eb1[2],
      g.data());
  return g;
}

inline Matrix<Real, 12, 12> edgeEdgeCrossSquaredNormHessian(
    const Vector<Real, 3> &ea0,
    const Vector<Real, 3> &ea1,
    const Vector<Real, 3> &eb0,
    const Vector<Real, 3> &eb1) {
  Matrix<Real, 12, 12> h;
  autogen::edge_edge_cross_squarednorm_hessian(
      ea0[0], ea0[1], ea0[2],
      ea1[0], ea1[1], ea1[2],
      eb0[0], eb0[1], eb0[2],
      eb1[0], eb1[1], eb1[2],
      h.data());
  return h;
}

inline Vector<Real, 12> edgeEdgeMollifierThresholdGradient(
    const Vector<Real, 3> &ea0,
    const Vector<Real, 3> &ea1,
    const Vector<Real, 3> &eb0,
    const Vector<Real, 3> &eb1,
    Real scale) {
  Vector<Real, 12> g;
  autogen::edge_edge_mollifier_threshold_gradient(
      ea0[0], ea0[1], ea0[2],
      ea1[0], ea1[1], ea1[2],
      eb0[0], eb0[1], eb0[2],
      eb1[0], eb1[1], eb1[2],
      g.data(), scale);
  return g;
}
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_MOLLIFIER_H_
