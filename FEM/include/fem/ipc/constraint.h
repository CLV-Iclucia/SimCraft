//
// Created by creeper on 5/28/24.
//

#pragma once
#include <fem/ipc/distances.h>
#include <fem/simplex.h>
#include <fem/types.h>
#include <span>
namespace sim {
namespace maths {
template <typename T> struct SparseMatrixBuilder;
}
namespace fem {
struct System;
namespace ipc {
struct LogBarrier;
struct VertexTriangleConstraint {
  Triangle triangle;
  CSubVector<Real> xv;
  CSubVector<Real> xt;
  int iv;
  PointTriangleDistanceType type{};
  void updateDistanceType();
  [[nodiscard]] Real distanceSqr() const;
  [[nodiscard]] Vector<Real, 12> localBarrierGradient(const LogBarrier &barrier,
                                                      Real kappa) const;
  void assembleBarrierGradient(const LogBarrier &barrier, VecXd &globalGradient,
                               Real kappa) const;
  void assembleBarrierHessian(const LogBarrier &barrier,
                              maths::SparseMatrixBuilder<Real> &globalHessian,
                              Real kappa) const;
};

struct EdgeEdgeConstraint {
  Edge ea{};
  Edge eb{};
  CSubVector<Real> xa;
  CSubVector<Real> Xa;
  CSubVector<Real> xb;
  CSubVector<Real> Xb;
  EdgeEdgeDistanceType type{};
  void updateDistanceType();
  [[nodiscard]] Real distanceSqr() const;
  void assembleMollifiedBarrierGradient(const LogBarrier &barrier,
                                        VecXd &globalGradient,
                                        Real kappa) const;
  void assembleMollifiedBarrierHessian(
      const LogBarrier &barrier,
      maths::SparseMatrixBuilder<Real> &globalHessian, Real kappa) const;
  [[nodiscard]] Real mollifier() const;
  [[nodiscard]] Vector<Real, 12> mollifierGradient() const;
  [[nodiscard]] Matrix<Real, 12, 12> mollifierHessian() const;
  [[nodiscard]] Vector<Real, 12>
  mollifiedBarrierGradient(const LogBarrier &barrier) const;
  [[nodiscard]] Matrix<Real, 12, 12>
  mollifiedBarrierHessian(const LogBarrier &barrier) const;

private:
  [[nodiscard]] Real epsCross() const;
  [[nodiscard]] Vector<Real, 12> crossedNormGradient() const;
  [[nodiscard]] Matrix<Real, 12, 12> crossedNormHessian() const;
  [[nodiscard]] Real crossSquaredNorm() const;

  static Real mollifier(Real c, Real e_x) {
    if (c < e_x)
      return -(c / e_x) * (c / e_x) + 2 * (c / e_x);
    return 1.0;
  }
  static Real mollifierDerivative(Real c, Real e_x) {
    if (c < e_x)
      return -2 * c / (e_x * e_x) + 2 / e_x;
    return 0.0;
  }
  static Real mollifierSecondDerivative(Real c, Real e_x) {
    if (c < e_x)
      return -2 / (e_x * e_x);
    return 0.0;
  }
};
} // namespace ipc
} // namespace fem
}
