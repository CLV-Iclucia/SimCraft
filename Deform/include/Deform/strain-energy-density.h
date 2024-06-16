//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_DEFORMABLES_STRAIN_ENERGY_DENSITY_H_
#define SIMCRAFT_DEFORMABLES_STRAIN_ENERGY_DENSITY_H_
#include <Deform/types.h>
#include <Deform/invariants.h>
#include <Deform/deformation-gradient.h>
#include <Maths/svd.h>
namespace deform {
template<typename T>
struct StrainEnergyDensity {
  virtual T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const = 0;
  virtual Matrix<T, 3, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const = 0;
  virtual Matrix<T, 9, 9> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const = 0;
  virtual Matrix<T, 9, 9> filteredEnergyHessian(const DeformationGradient<T, 3> &dg) const = 0;
};

template<typename T>
struct ElasticityParameters {
  T E, nu;
};

template<typename T>
struct StableNeoHookean final : StrainEnergyDensity<T> {
  T mu, lambda;
  StableNeoHookean(T mu, T lambda) : mu(mu), lambda(lambda) {}
  explicit StableNeoHookean(ElasticityParameters<T> params) {
    auto [E, nu] = params;
    auto mu_ = E / (2.0 * (1.0 + nu));
    auto lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    mu = 0.4 / 0.3 * mu_;
    lambda = lambda_ + 0.5 / 0.6 * mu_;
  }
  T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariants(dg);
    return 0.5 * mu * (I(1) - 3.0) - mu * (I(2) - 1.0) + 0.5 * lambda * (I(2) - 1.0) * (I(2) - 1.0);
  }
  // pEpIii = 0.5 * mu
  // pEpIiii = -mu + lambda * (I(2) - 1.0)
  // p2EpIii2 = 0
  // p2EpIiii2 = lambda
  Matrix<T, 3, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariants(dg);
    auto pIiipF = gradientIii<T>(dg);
    auto pIiiipF = gradientIiii<T>(dg);
    return 0.5 * mu * pIiipF - mu * pIiiipF + lambda * (I(2) - 1.0) * pIiiipF;
  }
  Matrix<T, 9, 9> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariants(dg);
    Vector<T, 9> vec_giii = maths::vectorize(gradientIiii<T>(dg));
    return lambda * vec_giii * vec_giii.transpose() + 0.5 * mu * hessianIii<T>()
        + ((I(2) - 1.0) * lambda - mu) * hessianIiii<T>(dg);
  }
  Matrix<T, 9, 9> filteredEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    auto I = isotropicInvariants(dg);
    Vector<T, 9> vec_giii = maths::vectorize(gradientIiii<T>(dg));
    Matrix<T, 9, 9> H = lambda * vec_giii * vec_giii.transpose() + 0.5 * mu * hessianIii<T>()
        + ((I(2) - 1.0) * lambda - mu) * hessianIiii<T>(dg);
    // compute eigenvalues
    Eigen::SelfAdjointEigenSolver<Matrix<T, 9, 9>> es(H);
    Vector<T, 9> eigenvalues = es.eigenvalues();
    Matrix<T, 9, 9> eigenvectors = es.eigenvectors();
    for (int i = 0; i < 9; i++)
      eigenvalues(i) = std::abs(eigenvalues(i));
    return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
  }
};

template<typename T>
struct ARAP final : StrainEnergyDensity<T> {
  T computeEnergyDensity(const DeformationGradient<T, 3> &dg) const override {
    return (dg.F() - dg.R()).squaredNorm();
  }
  Matrix<T, 3, 3> computeEnergyGradient(const DeformationGradient<T, 3> &dg) const override {
    return 2.0 * (dg.F() - dg.R());
  }
  Matrix<T, 9, 9> computeEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    return hessianIii<T>() - 2 * hessianIi(dg);
  }
  Matrix<T, 9, 9> filteredEnergyHessian(const DeformationGradient<T, 3> &dg) const override {
    auto H = hessianIii<T>() - 2 * hessianIi(dg);
    Matrix<T, 9, 9> U, V;
    Vector<T, 9> S;
    maths::SVD(H, U, S, V);
    for (int i = 0; i < 9; i++)
      S(i) = std::abs(S(i));
    return U * S.asDiagonal() * V.transpose();
  }
};
}
#endif //SIMCRAFT_DEFORMABLES_STRAIN_ENERGY_DENSITY_H_
