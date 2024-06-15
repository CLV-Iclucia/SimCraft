#include <Maths/equations.h>
#include <Core/utils.h>
#include <iostream>

namespace maths {

// the methods come from "A Fast & Robust Solution for Cubic & Higher-Order Polynomials" by Cem Yuksel, SIGGRAPH 2022
bool quadraticSolve(const QuadraticPolynomial &poly, Real &x1, Real &x2) {
  const auto &[a, b, c] = poly;
  Real d = b * b - 4 * a * c;
  if (d < 0)
    return false;
  int sign = b < 0 ? -1 : 1;
  Real q = -0.5 * (b + sign * std::sqrt(d));
  x1 = q / a;
  x2 = c / q;
  return true;
}

static Real evalQuadratic(const QuadraticPolynomial &poly, Real x) {
  const auto &[a, b, c] = poly;
  return (a * x + b) * x + c;
}

static Real evalCubic(const CubicPolynomial &poly, Real x) {
  const auto &[a, b, c, d] = poly;
  return ((a * x + b) * x + c) * x + d;
}

static Real robustCubicNewton(const CubicPolynomial &poly, Real x, Real l,
                              Real r, Real tolerance) {
  const auto &[a, b, c, d] = poly;
  while (true) {
    Real f = evalCubic(poly, x);
    Real df = evalQuadratic({.a = 3 * a, .b = 2 * b, .c = c}, x);
    if (df == 0.0) [[unlikely]]
      return x;
    Real dx = -f / df;
    x = core::clamp(x + dx, l, r);
    if (std::abs(dx) < tolerance)
      return x;
  }
}

// returns all the real root within the interval [l, r]
CubicEquationRoots clampedCubicSolve(const CubicPolynomial &poly, Real l, Real r,
                       Real tolerance) {
  const auto &[a, b, c, d] = poly;
  if (a == 0.0) {
    Real x1, x2;
    if (!quadraticSolve({.a = b, .b = c, .c = d}, x1, x2))
      return {};
    if (x1 > x2)
      std::swap(x1, x2);
    CubicEquationRoots roots;
    if (l <= x1 && x1 <= r)
      roots.addRoot(x1);
    if (l <= x2 && x2 <= r)
      roots.addRoot(x2);
    return roots;
  }
  Real x1, x2;
  Real fl = evalCubic(poly, l);
  Real fr = evalCubic(poly, r);
  Real xc = -b / (3.0 * a);
  Real fxc = evalCubic(poly, xc);
  if (!quadraticSolve({.a = 3 * a, .b = 2 * b, .c = c}, x1, x2)) {
    if (fl * fr > 0.0)
      return {};
    if (xc > l && xc < r) {
      if (fl * fxc <= 0.0)
        return CubicEquationRoots(robustCubicNewton(poly, l, l, xc, tolerance));
      return CubicEquationRoots(robustCubicNewton(poly, r, xc, r, tolerance));
    }
    return CubicEquationRoots(robustCubicNewton(poly, (l + r) * 0.5, l, r, tolerance));
  }
  if (x1 > x2)
    std::swap(x1, x2);
  Real fx1 = evalCubic(poly, x1);
  CubicEquationRoots roots;
  if (l < x1) {
    Real bound = std::min(r, x1);
    Real fbound = bound == r ? fr : fx1;
    if (fl * fbound <= 0.0)
      roots.addRoot(robustCubicNewton(poly, l, l, bound, tolerance));
  }
  Real fx2 = evalCubic(poly, x2);
  Real lbound = std::max(l, x1);
  Real flbound = lbound == l ? fl : fx1;
  Real rbound = std::min(r, x2);
  Real frbound = rbound == r ? fr : fx2;
  if (lbound < rbound && flbound * frbound <= 0.0) {
    Real root;
    if (xc > lbound && xc < rbound) {
      if (flbound * fxc <= 0.0)
        root = robustCubicNewton(poly, (lbound + xc) * 0.5, lbound, xc, tolerance);
      else root = robustCubicNewton(poly, (rbound + xc) * 0.5, xc, rbound, tolerance);
    } else
      root = robustCubicNewton(poly, (lbound + rbound) * 0.5, lbound,
                             rbound, tolerance);
    roots.addRoot(root);
  }
  if (x2 < r) {
    Real bound = std::max(l, x2);
    Real fbound = bound == l ? fl : fx2;
    if (fbound * fr <= 0.0)
      roots.addRoot(robustCubicNewton(poly, r, bound, r, tolerance));
  }
  return roots;
}
}