//
// Created by creeper on 9/16/24.
//
#include <fem/ipc/distances.h>
#include <gtest/gtest.h>
using namespace fem;
using namespace fem::ipc;

Vector<Real, 6> numericalPointDistanceGradient(const Vector<Real, 3> &a, const Vector<Real, 3> &b) {
  Real h = 1e-6;
  Vector<Real, 6> g;
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> a1 = a;
    Vector<Real, 3> a2 = a;
    a1[i] += h;
    a2[i] -= h;
    g[i] = (distanceSqrPointPoint(a1, b) - distanceSqrPointPoint(a2, b)) / (2 * h);
  }
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> b1 = b;
    Vector<Real, 3> b2 = b;
    b1[i] += h;
    b2[i] -= h;
    g[i + 3] = (distanceSqrPointPoint(a, b1) - distanceSqrPointPoint(a, b2)) / (2 * h);
  }
  return g;
}

Vector<Real, 9> numericalPointLineDistanceGradient(const Vector<Real, 3> &a,
                                                   const Vector<Real, 3> &b,
                                                   const Vector<Real, 3> &c) {
  Real h = 1e-6;
  Vector<Real, 9> g;
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> a1 = a;
    Vector<Real, 3> a2 = a;
    a1[i] += h;
    a2[i] -= h;
    g[i] = (distanceSqrPointLine(a1, b, c) - distanceSqrPointLine(a2, b, c)) / (2 * h);
  }
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> b1 = b;
    Vector<Real, 3> b2 = b;
    b1[i] += h;
    b2[i] -= h;
    g[i + 3] = (distanceSqrPointLine(a, b1, c) - distanceSqrPointLine(a, b2, c)) / (2 * h);
  }
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> c1 = c;
    Vector<Real, 3> c2 = c;
    c1[i] += h;
    c2[i] -= h;
    g[i + 6] = (distanceSqrPointLine(a, b, c1) - distanceSqrPointLine(a, b, c2)) / (2 * h);
  }
  return g;
}

Vector<Real, 12> numericalPointPlaneDistanceGradient(const Vector<Real, 3> &a,
                                                     const Vector<Real, 3> &b,
                                                     const Vector<Real, 3> &c,
                                                     const Vector<Real, 3> &d) {
  Real h = 1e-6;
  Vector<Real, 12> g;
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> a1 = a;
    Vector<Real, 3> a2 = a;
    a1[i] += h;
    a2[i] -= h;
    g[i] = (distanceSqrPointPlane(a1, b, c, d) - distanceSqrPointPlane(a2, b, c, d)) / (2 * h);
  }
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> b1 = b;
    Vector<Real, 3> b2 = b;
    b1[i] += h;
    b2[i] -= h;
    g[i + 3] = (distanceSqrPointPlane(a, b1, c, d) - distanceSqrPointPlane(a, b2, c, d)) / (2 * h);
  }
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> c1 = c;
    Vector<Real, 3> c2 = c;
    c1[i] += h;
    c2[i] -= h;
    g[i + 6] = (distanceSqrPointPlane(a, b, c1, d) - distanceSqrPointPlane(a, b, c2, d)) / (2 * h);
  }
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> d1 = d;
    Vector<Real, 3> d2 = d;
    d1[i] += h;
    d2[i] -= h;
    g[i + 9] = (distanceSqrPointPlane(a, b, c, d1) - distanceSqrPointPlane(a, b, c, d2)) / (2 * h);
  }
  return g;
}

Vector<Real, 12> numericalLineLineDistanceGradient(const Vector<Real, 3> &a,
                                                   const Vector<Real, 3> &b,
                                                   const Vector<Real, 3> &c,
                                                   const Vector<Real, 3> &d) {
  Real h = 1e-6;
  Vector<Real, 12> g;
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> a1 = a;
    Vector<Real, 3> a2 = a;
    a1[i] += h;
    a2[i] -= h;
    g[i] = (distanceSqrLineLine(a1, b, c, d) - distanceSqrLineLine(a2, b, c, d)) / (2 * h);
  }
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> b1 = b;
    Vector<Real, 3> b2 = b;
    b1[i] += h;
    b2[i] -= h;
    g[i + 3] = (distanceSqrLineLine(a, b1, c, d) - distanceSqrLineLine(a, b2, c, d)) / (2 * h);
  }
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> c1 = c;
    Vector<Real, 3> c2 = c;
    c1[i] += h;
    c2[i] -= h;
    g[i + 6] = (distanceSqrLineLine(a, b, c1, d) - distanceSqrLineLine(a, b, c2, d)) / (2 * h);
  }
  for (int i = 0; i < 3; i++) {
    Vector<Real, 3> d1 = d;
    Vector<Real, 3> d2 = d;
    d1[i] += h;
    d2[i] -= h;
    g[i + 9] = (distanceSqrLineLine(a, b, c, d1) - distanceSqrLineLine(a, b, c, d2)) / (2 * h);
  }
  return g;
}

TEST(PointPointDistanceGradient, Test) {
  for (int i = 0; i < 100; i++) {
    Vector<Real, 3> a = Vector<Real, 3>::Random();
    Vector<Real, 3> b = Vector<Real, 3>::Random();
    Vector<Real, 6> numGrad = numericalPointDistanceGradient(a, b);
    Vector<Real, 6> symGrad = ipc::localDistanceSqrPointPointGradient(a, b);
    ASSERT_LT((numGrad - symGrad).lpNorm<Eigen::Infinity>(), 1e-6);
  }
}

TEST(PointLineDistanceGradient, Test) {
  for (int i = 0; i < 100; i++) {
    Vector<Real, 3> a = Vector<Real, 3>::Random();
    Vector<Real, 3> b = Vector<Real, 3>::Random();
    Vector<Real, 3> c = Vector<Real, 3>::Random();
    Vector<Real, 9> numGrad = numericalPointLineDistanceGradient(a, b, c);
    Vector<Real, 9> symGrad = ipc::localDistanceSqrPointLineGradient(a, b, c);
    ASSERT_LT((numGrad - symGrad).lpNorm<Eigen::Infinity>(), 1e-6);
  }
}

TEST(PointPlaneDistanceGradient, Test) {
  for (int i = 0; i < 100; i++) {
    Vector<Real, 3> a = Vector<Real, 3>::Random();
    Vector<Real, 3> b = Vector<Real, 3>::Random();
    Vector<Real, 3> c = Vector<Real, 3>::Random();
    Vector<Real, 3> d = Vector<Real, 3>::Random();
    Vector<Real, 12> numGrad = numericalPointPlaneDistanceGradient(a, b, c, d);
    Vector<Real, 12> symGrad = ipc::localDistanceSqrPointPlaneGradient(a, b, c, d);
    ASSERT_LT((numGrad - symGrad).lpNorm<Eigen::Infinity>(), 1e-6);
  }
}

TEST(LineLineDistanceGradient, Test) {
  for (int i = 0; i < 100; i++) {
    Vector<Real, 3> a = Vector<Real, 3>::Random();
    Vector<Real, 3> b = Vector<Real, 3>::Random();
    Vector<Real, 3> c = Vector<Real, 3>::Random();
    Vector<Real, 3> d = Vector<Real, 3>::Random();
    Vector<Real, 12> numGrad = numericalLineLineDistanceGradient(a, b, c, d);
    Vector<Real, 12> symGrad = ipc::localDistanceSqrLineLineGradient(a, b, c, d);
    ASSERT_LT((numGrad - symGrad).lpNorm<Eigen::Infinity>(), 1e-6);
  }
}

int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}