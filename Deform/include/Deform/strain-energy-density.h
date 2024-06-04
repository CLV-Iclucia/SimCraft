//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_DEFORMABLES_STRAIN_ENERGY_DENSITY_H_
#define SIMCRAFT_DEFORMABLES_STRAIN_ENERGY_DENSITY_H_
#include <Deform/types.h>
#include <Deform/invariants.h>
#include <Deform/deformation-gradient.h>
namespace deform {
template<typename T>
struct StrainEnergyDensity {
  virtual T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const = 0;
  virtual Matrix<T, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const = 0;
  virtual FourthOrderTensor<T, 3> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const = 0;
};

template<typename T>
struct StableNeoHookean final : StrainEnergyDensity<T> {
  T mu, lambda;
  StableNeoHookean(T mu, T lambda) : mu(mu), lambda(lambda) {}
  T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariants(dg.F());
    return 0.5 * mu * (I(0) - 3.0) - mu * (I(2) - 1.0) + 0.5 * lambda * (I(2) - 1.0) * (I(2) - 1.0);
  }
  // pEpIi = 0.5 * mu
  // pEpIiii = -mu + lambda * (I(2) - 1.0)
  // p2EpIi2 = 0
  // p2EpIiii2 = lambda
  Matrix<T, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariantsUsingS(dg.S());
    auto pIipF = gradientIi<T, true>(dg.R());
    auto pIiiipF = gradientIii<T>(dg.F());
    return 0.5 * mu * pIipF - mu * pIiiipF + lambda * (I(2) - 1.0) * pIiiipF;
  }
  FourthOrderTensor<T, 3> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariants(dg.F());
    return lambda * maths::tensorProduct(maths::vectorize(hessianIi<T>(dg)))
        + ((I(2) - 1.0) * lambda - mu) * hessianIiii<T>(dg);
  }
};

template<typename T>
struct ARAP final : StrainEnergyDensity<T> {
  T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const override {
    return (dg.F() - dg.R()).squaredNorm();
  }
  Matrix<T, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const override {
    return 2.0 * (dg.F() - dg.R());
  }
  FourthOrderTensor<T, 3> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    return hessianIii<T>() - 2 * hessianIi(dg);
  }
};
}
#endif //SIMCRAFT_DEFORMABLES_STRAIN_ENERGY_DENSITY_H_
