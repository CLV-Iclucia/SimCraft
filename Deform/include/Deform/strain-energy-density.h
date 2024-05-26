//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_DEFORMABLES_STRAIN_ENERGY_DENSITY_H_
#define SIMCRAFT_DEFORMABLES_STRAIN_ENERGY_DENSITY_H_
#include <Deform/types.h>
#include <Deform/invariants.h>
namespace deform {
template<typename T>
struct StrainEnergyDensity {
  virtual T computeEnergyDensity(const Matrix<T, 3> &F) const = 0;
  virtual Matrix<T, 3> computePKiStrain(const Matrix<T, 3> &F) const = 0;
  virtual FourthOrderTensor<T, 3> computeEnergyHessian(const Matrix<T, 3> &F) const = 0;
};

template<typename T>
struct StableNeoHookean final : StrainEnergyDensity<T> {
  T mu, lambda;
  StableNeoHookean(T mu, T lambda) : mu(mu), lambda(lambda) {}
  T computeEnergyDensity(const Matrix<T, 3> &F) const override {
    auto I = isotropicInvariants(F);
    return 0.5 * mu * (I(0) - 3.0) - mu * (I(2) - 1.0) + 0.5 * lambda * (I(2) - 1.0) * (I(2) - 1.0);
  }
  Matrix<T, 3> computePKiStrain(const Matrix<T, 3> &F) const override {
    Matrix<T, 3> R, S;
    polarDecomposition(F, R, S);
    auto I = isotropicInvariantsUsingS(S);
    Matrix<T, 3> pIipF, pIiipF, pIiiipF;
    isotropicInvariantsDerivative<T, ComputeIiDerivative | ComputeIiiiDerivative>(F, pIipF, pIiipF, pIiiipF);
    return 0.5 * mu * pIipF - mu * pIiiipF + lambda * (I(2) - 1.0) * pIiiipF;
  }
  FourthOrderTensor<T, 3> computeEnergyHessian(const Matrix<T, 3> &F) const override {

  }
};

template<typename T>
struct ARAP final : StrainEnergyDensity<T> {
  T computeEnergyDensity(const Matrix<T, 3> &F) const override {
    Matrix<T, 3> R, S;
    polarDecomposition<T, ComputeFullS>(F, R, S);
    return (F - R).squaredNorm();
  }
  Matrix<T, 3> computePKiStrain(const Matrix<T, 3> &F) const override {
    Matrix<T, 3> R, S;
    polarDecomposition<T, ComputeFullS>(F, R, S);
    return 2.0 * (F - R);
  }
  FourthOrderTensor<T, 3> computeEnergyHessian(const Matrix<T, 3> &F) const override {
    Matrix<T, 3> U, S, V;
    svd3x3(F, U, S, V);
    return hessianIii<T>() - 2 * hessianIi(U, S.asDiagonal(), V);
  }
};
}
#endif //SIMCRAFT_DEFORMABLES_STRAIN_ENERGY_DENSITY_H_
