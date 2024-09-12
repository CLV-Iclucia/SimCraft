//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_CONSTRAINT_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_CONSTRAINT_H_
#include <fem/types.h>
#include <fem/ipc/distances.h>
namespace maths {
template <typename T>
struct SparseMatrixBuilder;
}
namespace fem {
struct System;
namespace ipc {
struct LogBarrier;
struct VertexTriangleConstraint {
  const System& system;
  // if type is EdgeEdge, ia and ib are the indices of the two edges
  // if type is VertexTriangle, ia is the index of the vertex and ib is the index of the triangle
  int iv, it;
  PointTriangleDistanceType type;
  void updateDistanceType();
  [[nodiscard]] Real distance() const;
  void assembleBarrierGradient(const LogBarrier& barrier, VecXd& global_gradient, Real kappa) const ;
  void assembleBarrierHessian(const LogBarrier& barrier, maths::SparseMatrixBuilder<Real>& global_hessian, Real kappa) const ;
};
struct EdgeEdgeConstraint {
  const System& system;
  int ia, ib;
  EdgeEdgeDistanceType type;
  void updateDistanceType();
  [[nodiscard]] Real distance() const;
  void assembleMollifiedBarrierGradient(const LogBarrier& barrier, VecXd& globalGradient, Real kappa) const ;
  void assembleMollifiedBarrierHessian(const LogBarrier& barrier, maths::SparseMatrixBuilder<Real>& globalHessian, Real kappa) const ;
 private:
  [[nodiscard]] Real epsCross() const;
  [[nodiscard]] Vector<Real, 12> crossedNormGradient() const;
  [[nodiscard]] Matrix<Real, 12, 12> crossedNormHessian() const;
  [[nodiscard]] Real crossSquaredNorm() const;
  [[nodiscard]] Vector<Real, 12> mollifiedBarrierGradient(const LogBarrier& barrier) const;
  [[nodiscard]] Matrix<Real, 12, 12> mollifiedBarrierHessian(const LogBarrier& barrier) const;
  static Real mollifier(Real c, Real e_x) {
    if (c < e_x) return -(c / e_x) * (c / e_x) + 2 * (c / e_x);
    return 1.0;
  }
  static Real mollifierDerivative(Real c, Real e_x) {
    if (c < e_x) return -2 * c / (e_x * e_x) + 2 / e_x;
    return 0.0;
  }
  static Real mollifierSecondDerivative(Real c, Real e_x) {
    if (c < e_x) return -2 / (e_x * e_x);
    return 0.0;
  }
  [[nodiscard]] Real mollifier() const;
  [[nodiscard]] Vector<Real, 12> mollifierGradient() const;
  [[nodiscard]] Matrix<Real, 12, 12> mollifierHessian() const;
};
}
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_CONSTRAINT_H_
