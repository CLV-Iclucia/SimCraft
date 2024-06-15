//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_EQUATIONS_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_EQUATIONS_H_
#include <Maths/types.h>
namespace maths {
struct QuadraticPolynomial {
  Real a, b, c;
};
bool quadraticSolve(const QuadraticPolynomial &poly, Real &x1, Real &x2);
struct CubicPolynomial {
  Real a, b, c, d;
};
struct CubicEquationRoots {
  std::array<Real, 3> roots{};
  int num_roots{};
  CubicEquationRoots() = default;
  explicit CubicEquationRoots(Real x) : num_roots(1) {
    roots[0] = x;
  }
  void addRoot(Real x) {
    roots[num_roots++] = x;
  }
};
CubicEquationRoots clampedCubicSolve(const CubicPolynomial &poly, Real l, Real r, Real tolerance);
struct BinaryLinearSystem {
  Real a00, a01, a10, a11, b0, b1;
};
inline bool binaryLinearSolve(const BinaryLinearSystem &sys, Real &x, Real &y) {
  Real det = sys.a00 * sys.a11 - sys.a01 * sys.a10;
  if (det == 0.0) return false;
  x = (sys.b0 * sys.a11 - sys.b1 * sys.a01) / det;
  y = (sys.a00 * sys.b1 - sys.a10 * sys.b0) / det;
  return true;
}
}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_EQUATIONS_H_
