#include <Deform/types.h>
#include <Deform/invariants.h>
#include <Deform/strain-energy-density.h>
#include <fem/tet-mesh.h>
#include <ogl-render/camera-controller.h>
#include <ogl-render/window.h>
#include <Maths/svd.h>
#include <Maths/tensor.h>
#include <iostream>
#include <format>
using namespace deform;
using namespace fem;

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

void ipcImplicitEuler(TetMesh& mesh) {
  DeformationGradient<Real, 3> dg(Matrix<Real, 3, 3>::Identity());
  StableNeoHookean<Real> energy(ElasticityParameters<Real>{1e4, 0.49});

  dg.updateCurrentConfig(tetDs(x));
  // one step
  Real E_prev = energy.computeEnergyDensity(dg);

  std::cout << dg.Sigma(0) << ", " << dg.Sigma(1) << ", " << dg.Sigma(2) << '\n';

  std::cout << "----------------------------\n";
  int iter = 0;
  while (true) {
    auto gradient_x = tetEnergyGradient(energy, dg);
    auto hessian_x = tetEnergyHessian(energy, dg);
    Vector<Real, 12> p;
    if (iter > 10)
      p = -hessian_x.ldlt().solve(gradient_x);
    else p = -gradient_x;
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
    } while (E > E_prev + 1e-3 * alpha * g_dot_p);
    x = x_new;
    E_prev = E;
    iter++;
    std::cout << dg.Sigma(0) << ", " << dg.Sigma(1) << ", " << dg.Sigma(2) << '\n';
    std::cout << "----------------------------\n";
    getchar();
  }
}
int main() {
  auto bunny = fem::readTetMeshFromTOBJ(ASSETS_DIR"/bunny.tobj");
  if (bunny == nullptr) {
    core::ERROR("Failed to load bunny.tobj");
    std::exit(-1);
  }

}