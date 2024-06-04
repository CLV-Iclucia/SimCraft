//
// Created by creeper on 6/3/24.
//

#ifndef SIMCRAFT_DEFORM_INCLUDE_DEFORM_DEFORMATION_GRADIENT_H_
#define SIMCRAFT_DEFORM_INCLUDE_DEFORM_DEFORMATION_GRADIENT_H_
#include <Deform/types.h>
#include <Eigen/Core>
#include <Maths/tensor.h>
namespace deform {
using maths::submatrix;
template<typename T, int Dim>
struct DeformationGradient {
  DeformationGradient() = default;
  explicit DeformationGradient(const Matrix<T, Dim> &local_X) {
    m_Dm_inverse = local_X.inverse();
    m_F = Matrix<T, Dim>::Identity();
    m_U = Matrix<T, Dim>::Identity();
    m_V = Matrix<T, Dim>::Identity();
    m_Sigma = Vector<T, Dim>::Ones();
  }
  void updateCurrentConfig(const Matrix<T, Dim> &local_x) {
    m_F = local_x * m_Dm_inverse;
    computeSVD();
  }
  maths::ThirdOrderTensor<T, Dim, Dim + 1> gradient() const {
    if constexpr (Dim == 3) {
      maths::ThirdOrderTensor<T, Dim, Dim * (Dim + 1)> result;
      result.setZero();
      const auto &r0 = m_Dm_inverse.row(0);
      const auto &r1 = m_Dm_inverse.row(1);
      const auto &r2 = m_Dm_inverse.row(2);
      auto s0 = m_Dm_inverse.col(0).sum();
      auto s1 = m_Dm_inverse.col(1).sum();
      auto s2 = m_Dm_inverse.col(2).sum();
      submatrix(result, 0)(0, 0) = -s0;
      submatrix(result, 0)(0, 1) = -s1;
      submatrix(result, 0)(0, 2) = -s2;
      submatrix(result, 1)(1, 0) = -s0;
      submatrix(result, 1)(1, 1) = -s1;
      submatrix(result, 1)(1, 2) = -s2;
      submatrix(result, 2)(2, 0) = -s0;
      submatrix(result, 2)(2, 1) = -s1;
      submatrix(result, 2)(2, 2) = -s2;
      submatrix(result, 3).row(0) = r0;
      submatrix(result, 4).row(0) = r1;
      submatrix(result, 5).row(0) = r2;
      submatrix(result, 6).row(1) = r0;
      submatrix(result, 7).row(1) = r1;
      submatrix(result, 8).row(1) = r2;
      submatrix(result, 9).row(2) = r0;
      submatrix(result, 10).row(2) = r1;
      submatrix(result, 11).row(2) = r2;
      return result;
    } else
      core::ERROR("Sorry, gradient of deformation gradient is not implemented for dimension other than 3");
  }
  // read-only
  const Matrix<T, Dim> &F() const {
    return m_F;
  }
  const Matrix<T, Dim> &R() const {
    return m_U * m_V.transpose();
  }
  const Vector<T, Dim> &Sigma() const {
    return m_Sigma;
  }
  [[nodiscard]] Real Sigma(int i) const {
    return m_Sigma(i);
  }
  const Matrix<T, Dim> &S() const {
    return m_U * m_Sigma.asDiagonal() * m_V.transpose();
  }
 private:
  void computeSVD() {
    auto svd = m_F.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    m_U = svd.matrixU();
    m_V = svd.matrixV();
    m_Sigma = svd.singularValues();
  }
  Matrix<Real, Dim> m_F, m_U, m_V, m_Dm_inverse;
  Vector<Real, Dim> m_Sigma;
};

}
#endif //SIMCRAFT_DEFORM_INCLUDE_DEFORM_DEFORMATION_GRADIENT_H_
