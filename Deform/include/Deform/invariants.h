//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_DEFORM_INCLUDE_DEFORM_INVARIANTS_H_
#define SIMCRAFT_DEFORM_INCLUDE_DEFORM_INVARIANTS_H_
#include <Deform/types.h>
#include <Deform/polar-decomp.h>
#include <Deform/deformation-gradient.h>
#include <Core/properties.h>
namespace deform {
struct EigenMatrixSigmas : core::Singleton {
  std::array<Matrix<Real, 3>, 3> eigenMatrixSigmas;
  EigenMatrixSigmas() {
    eigenMatrixSigmas[0].setZero();
    eigenMatrixSigmas[0](0, 1) = -1.0;
    eigenMatrixSigmas[0](1, 0) = 1.0;
    eigenMatrixSigmas[1].setZero();
    eigenMatrixSigmas[1](1, 2) = 1.0;
    eigenMatrixSigmas[1](2, 1) = -1.0;
    eigenMatrixSigmas[2].setZero();
    eigenMatrixSigmas[2](0, 2) = 1.0;
    eigenMatrixSigmas[2](2, 0) = -1.0;
  }
  [[nodiscard]] const auto &S(int i) const {
    return eigenMatrixSigmas[i];
  }
};

const EigenMatrixSigmas &eigenMatrixSigmas() {
  static EigenMatrixSigmas instance;
  return instance;
}

template<typename T>
Vector<T, 3> isotropicInvariants(const DeformationGradient<T, 3> &dg) {
  return {dg.S().trace(), (dg.S() * dg.S()).trace(), dg.S().determinant()};
}

template<typename T>
Matrix<T, 3> gradientIi(const DeformationGradient<T, 3> &dg) {
  return dg.R();
}

template<typename T>
Matrix<T, 3> gradientIii(const DeformationGradient<T, 3> &dg) {
  return 2.0 * dg.F();
}

template<typename T>
Matrix<T, 3> gradientIiii(const DeformationGradient<T, 3> &dg) {
  return maths::determinantGradient(dg.F());
}
template<typename T>
FourthOrderTensor<T, 3> hessianIi(const DeformationGradient<T, 3> &dg) {
  std::array<Real, 3> lambda{2.0 / (dg.Sigma(0) + dg.Sigma(1)), 2.0 / (dg.Sigma(1) + dg.Sigma(2)),
                             2.0 / (dg.Sigma(0) + dg.Sigma(2))};
  FourthOrderTensor<T, 3> pRpF;
  for (int i = 0; i < 3; i++) {
    auto Q = dg.U() * eigenMatrixSigmas().S(i) * dg.V().transpose();
    pRpF += 0.5 * lambda[i] * vectorize(Q) * vectorize(Q).transpose();
  }
  return pRpF;
}
template<typename T>
FourthOrderTensor<T, 3> hessianIii() {
  return 2 * FourthOrderTensor<T, 3>::Identity();
}
template<typename T>
FourthOrderTensor<T, 3> hessianIiii(const DeformationGradient<T, 3> &dg) {
  FourthOrderTensor<T, 3> result{};
  result.setZero();
  result.block<3, 3>(0, 0) = skewt(dg.F().col(2));
  result.block<3, 3>(6, 0) = -skewt(dg.F().col(1));
  result.block<3, 3>(0, 3) = -skewt(dg.F().col(2));
  result.block<3, 3>(6, 3) = skewt(dg.F().col(0));
  result.block<3, 3>(0, 6) = skewt(dg.F().col(1));
  result.block<3, 3>(3, 6) = -skewt(dg.F().col(0));
  return result;
}

}
#endif //SIMCRAFT_DEFORM_INCLUDE_DEFORM_INVARIANTS_H_
