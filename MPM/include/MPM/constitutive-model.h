//
// Created by creeper on 23-8-14.
//

#ifndef SIMCRAFT_MPM_INCLUDE_MPM_CONSTITUTIVE_MODEL_H_
#define SIMCRAFT_MPM_INCLUDE_MPM_CONSTITUTIVE_MODEL_H_
#include <Core/tensor.h>
#include <MPM/mpm.h>
#include <cmath>
#include <variant>
namespace mpm {

struct NeoHookean {
  // compute the first Piola-Kirchhoff stress tensor
  template <int Dim> Matrix<Real, Dim> P(const Matrix<Real, Dim> &F) const {
    Matrix<Real, Dim> F_inv = F.inverse();
    return mu * (F - F_inv.transposed()) +
           lambda * std::log(F.determinant()) * F_inv.transposed();
  }
  template <int Dim>
  void P(const Matrix<Real, Dim> &F, Matrix<Real, Dim> &ret) const {
    Matrix<Real, Dim> F_inv = F.inverse();
    ret = mu * (F - F_inv.transposed()) +
          lambda * std::log(F.determinant()) * F_inv.transposed();
  }

  template <int Dim>
  void partialP_partialF(const Matrix<Real, Dim> &F,
                         FourthOrderTensor<Real, Dim> &ret) const {

  }
  Real mu, lambda;
};

template <int Dim> struct EnergyDensityOp {
  const Matrix<Real, Dim> &F;
  Real operator()(const NeoHookean &nh) const {
    return 0.5 * nh.mu * ((F * F.transposed()).trace() - Dim) -
           nh.mu * std::log(F.determinant()) +
           0.5 * nh.lambda * sqr(std::log(F.determinant()));
  }
};

template <int Dim> struct FirstPiolaKirchhoffStressOp {
  const Matrix<Real, Dim> &F;
  Matrix<Real, Dim> operator()(const NeoHookean &nh) const {
    Matrix<Real, Dim> F_inv = F.inverse();
    return nh.mu * (F - F_inv.transposed()) +
           nh.lambda * std::log(F.determinant()) * F_inv.transposed();
  }
};

template <int Dim> struct partialP_partialF_Op {
  const Matrix<Real, Dim> &F;
  FourthOrderTensor<Real, Dim> &ret;
  void operator()(const NeoHookean &nh) const {}
};
using Material = std::variant<NeoHookean>;
template <int Dim>
inline Real energyDensity(const Material &material,
                          const Matrix<Real, Dim> &F) {
  return std::visit(EnergyDensityOp<Dim>{F}, material);
}
template <int Dim>
inline Matrix<Real, Dim> PK1Stress(const Material &material,
                                   const Matrix<Real, Dim> &F) {
  return std::visit(FirstPiolaKirchhoffStressOp<Dim>{F}, material);
}
template <int Dim>
inline void partialP_partialF(const Material &material,
                              const Matrix<Real, Dim> &F,
                              FourthOrderTensor<Real, Dim> &ret) {
  std::visit(partialP_partialF_Op<Dim>{F, ret}, material);
}
} // namespace mpm
// namespace mpm
#endif // SIMCRAFT_MPM_INCLUDE_MPM_CONSTITUTIVE_MODEL_H_
