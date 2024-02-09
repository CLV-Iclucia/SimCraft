#include <Core/utils.h>

#include <iostream>

namespace core {
// the methods come from "A Fast & Robust Solution for Cubic & Higher-Order Polynomials" by Cem Yuksel, SIGGRAPH 2022

bool quadraticSolve(Real a, Real b, Real c, Real& x1, Real& x2) {
  Real d = b * b - 4 * a * c;
  if (d < 0)
    return false;
  int sign = b < 0 ? -1 : 1;
  Real q = -0.5 * (b + sign * std::sqrt(d));
  x1 = q / a;
  x2 = c / q;
  return true;
}

static Real evalQuadratic(Real a, Real b, Real c, Real x) {
  return (a * x + b) * x + c;
}

static Real evalCubic(Real a, Real b, Real c, Real d, Real x) {
  return ((a * x + b) * x + c) * x + d;
}

static Real robustCubicNewton(Real a, Real b, Real c, Real d, Real x, Real l,
                              Real r, Real tolerance) {
  while (true) {
    Real f = evalCubic(a, b, c, d, x);
    Real df = evalQuadratic(3 * a, 2 * b, c, x);
    Real dx = -f / df;
    x = clamp(x + dx, l, r);
    if (std::abs(dx) < tolerance)
      return x;
  }
}

// returns the smallest real root within the interval [l, r]
// if no real root is found, returns NaN
Real cubicSolve(Real a, Real b, Real c, Real d, Real l, Real r,
                Real tolerance) {
  if (a == 0.0) {
    Real x1, x2;
    if (!quadraticSolve(b, c, d, x1, x2))
      return std::numeric_limits<Real>::quiet_NaN();
    if (x1 > x2)
      std::swap(x1, x2);
    if (l <= x1 && x1 <= r)
      return x1;
    if (l <= x2 && x2 <= r)
      return x2;
    return std::numeric_limits<Real>::quiet_NaN();
  }
  Real x1, x2;
  Real fl = evalCubic(a, b, c, d, l);
  Real fr = evalCubic(a, b, c, d, r);
  Real xc = -b / (3.0 * a);
  Real fxc = evalCubic(a, b, c, d, xc);
  if (!quadraticSolve(3 * a, 2 * b, c, x1, x2)) {
    if (fl * fr > 0.0)
      return std::numeric_limits<Real>::quiet_NaN();
    if (xc > l && xc < r) {
      if (fl * fxc <= 0.0)
        return robustCubicNewton(a, b, c, d, l, l, xc, tolerance);
      return robustCubicNewton(a, b, c, d, r, xc, r, tolerance);
    }
    return robustCubicNewton(a, b, c, d, (l + r) * 0.5, l, r, tolerance);
  }
  if (x1 > x2)
    std::swap(x1, x2);
  Real fx1 = evalCubic(a, b, c, d, x1);
  if (l < x1) {
    Real bound = std::min(r, x1);
    Real fbound = bound == r ? fr : fx1;
    if (fl * fbound <= 0.0)
      return robustCubicNewton(a, b, c, d, l, l, bound, tolerance);
  }
  Real fx2 = evalCubic(a, b, c, d, x2);
  Real lbound = std::max(l, x1);
  Real flbound = lbound == l ? fl : fx1;
  Real rbound = std::min(r, x2);
  Real frbound = rbound == r ? fr : fx2;
  if (lbound < rbound && flbound * frbound <= 0.0) {
    if (xc > lbound && xc < rbound) {
      if (flbound * fxc <= 0.0)
        return robustCubicNewton(a, b, c, d, (lbound + xc) * 0.5, lbound, xc, tolerance);
      return robustCubicNewton(a, b, c, d, (rbound + xc) * 0.5, xc, rbound, tolerance);
    }
    return robustCubicNewton(a, b, c, d, (lbound + rbound) * 0.5, lbound,
                             rbound, tolerance);
  }
  if (x2 < r) {
    Real bound = std::max(l, x2);
    Real fbound = bound == l ? fl : fx2;
    if (fbound * fr <= 0.0)
      return robustCubicNewton(a, b, c, d, r, bound, r, tolerance);
  }
  return std::numeric_limits<Real>::quiet_NaN();
}
}