//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_DEFORM_INCLUDE_DEFORM_INVARIANTS_H_
#define SIMCRAFT_DEFORM_INCLUDE_DEFORM_INVARIANTS_H_
#include <Deform/types.h>
#include <Deform/polar-decomp.h>
namespace deform {
std::array<Matrix<Real, 3>, 3> eigenMatrixSigmas;
int prepareEigenMatrixSigmas() {
  eigenMatrixSigmas[0].setZero();
  eigenMatrixSigmas[0](0, 1) = -1.0;
  eigenMatrixSigmas[0](1, 0) = 1.0;
  eigenMatrixSigmas[1].setZero();
  eigenMatrixSigmas[1](1, 2) = 1.0;
  eigenMatrixSigmas[1](2, 1) = -1.0;
  eigenMatrixSigmas[2].setZero();
  eigenMatrixSigmas[2](0, 2) = 1.0;
  eigenMatrixSigmas[2](2, 0) = -1.0;
  return 0;
}
// this ensures that the eigenMatrixSigmas are initialized before main
int kEigenInvoker = prepareEigenMatrixSigmas();

template<typename T>
Vector<T, 3> isotropicInvariants(const Matrix<T, 3> &F) {
  Matrix<T, 3> R, S;
  polarDecomposition<T, ComputeFullS>(F, R, S);
  Vector<T, 3> I;
  I(0) = S.trace();
  I(1) = (S * S).trace();
  I(2) = S.determinant();
  return I;
}

template<typename T>
Vector<T, 3> isotropicInvariantsUsingS(const Matrix<T, 3> &S) {
  Vector<T, 3> I;
  I(0) = S.trace();
  I(1) = (S * S).trace();
  I(2) = S.determinant();
  return I;
}
enum InvariantsDerivativeOptions {
  ComputeIiDerivative = 0x01,
  ComputeIiiDerivative = 0x02,
  ComputeIiiiDerivative = 0x04
};

template<typename T, int Options = ComputeIiDerivative | ComputeIiiDerivative | ComputeIiiiDerivative>
void isotropicInvariantsDerivative(const Matrix<T, 3> &F,
                                   Matrix<T, 3> &pIipF,
                                   Matrix<T, 3> &pIiipF,
                                   Matrix<T, 3> &pIiiipF) {
  Matrix<T, 3> R, S;
  polarDecomposition<T, ComputeFullS>(F, R, S);
  if (Options & ComputeIiDerivative)
    pIipF = R;
  if (Options & ComputeIiiDerivative)
    pIiipF = 2.0 * F;

}

template<typename T, int Options = ComputeIiDerivative | ComputeIiiDerivative | ComputeIiiiDerivative>
void isotropicInvariantsDerivative(const Matrix<T, 3> &R,
                                   const Matrix<T, 3> &S,
                                   Matrix<T, 3> &pIipF,
                                   Matrix<T, 3> &pIiipF,
                                   Matrix<T, 3> &pIiiipF) {
  if (Options & ComputeIiDerivative)
    pIipF = R;
  if (Options & ComputeIiiDerivative)
    pIiipF = 2.0 * R * S;

}
template<typename T>
FourthOrderTensor<T, 3> hessianIi(const Matrix<T, 3> &U, const Vector<T, 3> &sigma, const Matrix<T, 3> &V) {
  std::array<Real, 3> lambda{2.0 / (sigma(0) + sigma(1)), 2.0 / (sigma(1) + sigma(2)), 2.0 / (sigma(0) + sigma(2))};
  FourthOrderTensor<T, 3> pRpF;
  for (int i = 0; i < 3; i++) {
    auto Q = U * eigenMatrixSigmas[i] * V.transpose();
    pRpF += 0.5 * lambda[i] * vectorize(Q) * vectorize(Q).transpose();
  }
  return pRpF;
}
template<typename T>
FourthOrderTensor<T, 3> hessianIii() {
  return 2 * FourthOrderTensor<T, 3>::Identity();
}
template<typename T>
FourthOrderTensor<T, 3> hessianIiii(const Matrix<T, 3> &F) {
  FourthOrderTensor<T, 3> result{};
  result.setZero();
  result.block<3, 3>(1, 0) = skewt(F.col(2));
  result.block<3, 3>(2, 0) = -skewt(F.col(1));
  result.block<3, 3>(0, 1) = -skewt(F.col(2));
  result.block<3, 3>(2, 1) = skewt(F.col(0));
  result.block<3, 3>(0, 2) = skewt(F.col(1));
  result.block<3, 3>(1, 2) = -skewt(F.col(0));
  return result;
}
}
#endif //SIMCRAFT_DEFORM_INCLUDE_DEFORM_INVARIANTS_H_
