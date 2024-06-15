//
// Created by creeper on 5/23/24.
//
#include <Deform/types.h>
#include <Deform/invariants.h>
#include <Deform/strain-energy-density.h>
#include <Maths/svd.h>
#include <Maths/tensor.h>
#include <iostream>
#include <format>
using namespace deform;

template<typename T, int N, int M>
std::string toString(const Eigen::Matrix<T, N, M> &A) {
  std::string result;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j)
      result += std::format("{:f} ", A(i, j));
    result += '\n';
  }
  return result;
}

template<int Dim>
bool checkSPD(const Eigen::Matrix<Real, Dim, Dim> &A) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, Dim, Dim>> es(A);
  return es.eigenvalues().minCoeff() >= 0.0;
}
template<int Dim>
bool checkSymmetric(const Eigen::Matrix<Real, Dim, Dim> &A) {
  return (A - A.transpose()).norm() < 1e-6;
}
Vector<Real, 12> tetEnergyGradient(const StableNeoHookean<Real> &energy,
                                   const DeformationGradient<Real, 3> &dg) {
  auto gradient_F = energy.computeEnergyGradient(dg);
  auto p_F_p_x = dg.gradient();
  Vector<Real, 12> gradient_x = p_F_p_x.transpose() * maths::vectorize(gradient_F);
  return gradient_x;
}

Matrix<Real, 12, 12> tetEnergyHessian(const StableNeoHookean<Real> &energy,
                                      const DeformationGradient<Real, 3> &dg) {
  auto hessian_F = energy.computeEnergyHessian(dg);
  auto res = maths::SVD(hessian_F);
  auto p_F_p_x = dg.gradient();
  auto hessian_x = p_F_p_x.transpose() * hessian_F * p_F_p_x;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, 12, 12>> es(hessian_x);

  Eigen::MatrixXd DiagEval = es.eigenvalues().real().asDiagonal();
  Eigen::MatrixXd Evec = es.eigenvectors().real();

  for (int i = 0; i < 12; ++i)
    if (es.eigenvalues()[i] < 0.0)
      DiagEval(i,i) = std::abs(es.eigenvalues()[i]);

  return Evec * DiagEval * Evec.transpose();
}

Matrix<Real, 3, 3> tetDs(const Vector<Real, 12> &x) {
  Matrix<Real, 3, 3> Ds;
  Ds.col(0) = x.segment<3>(3) - x.segment<3>(0);
  Ds.col(1) = x.segment<3>(6) - x.segment<3>(0);
  Ds.col(2) = x.segment<3>(9) - x.segment<3>(0);
  return Ds;
}

void tetElasticityOptimize(const Vector<Real, 3> &a,
                           const Vector<Real, 3> &b,
                           const Vector<Real, 3> &c,
                           const Vector<Real, 3> &d) {
  DeformationGradient<Real, 3> dg(Matrix<Real, 3, 3>::Identity());
  StableNeoHookean<Real> energy(0.1, 1.0);
  Vector<Real, 12> x;
  x << a, b, c, d;
  dg.updateCurrentConfig(tetDs(x));
  // one step
  Real E_prev = energy.computeEnergyDensity(dg);

  std::cout << dg.Sigma(0) << ", " << dg.Sigma(1) << ", " << dg.Sigma(2) << '\n';

  std::cout << "----------------------------\n";
  while (true) {
    auto gradient_x = tetEnergyGradient(energy, dg);
    auto hessian_x = tetEnergyHessian(energy, dg);
    Vector<Real, 12> p = -hessian_x.ldlt().solve(gradient_x);
    Real alpha = 1.0;
    Real E;
    Vector<Real, 12> x_new;
    Real g_dot_p = gradient_x.dot(p);
    assert(g_dot_p < 0.0);
    do {
      x_new = x + alpha * p;
      dg.updateCurrentConfig(tetDs(x_new));
      E = energy.computeEnergyDensity(dg);
      if (alpha < 1e-8) break;
      alpha *= 0.5;
    } while (E > E_prev + 1e-8 * g_dot_p);
    x = x_new;
    E_prev = E;
    std::cout << dg.Sigma(0) << ", " << dg.Sigma(1) << ", " << dg.Sigma(2) << '\n';
    std::cout << "----------------------------\n";
    getchar();
  }
}
int main() {
  Vector<Real, 3> a(0, 0, 0);
  Vector<Real, 3> b(-2, 0, 0);
  Vector<Real, 3> c(0, 10, 0);
  Vector<Real, 3> d(0, 0, 2);
  tetElasticityOptimize(a, b, c, d);
}