//
// Created by creeper on 5/24/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_COLLISION_RESOLVER_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_COLLISION_RESOLVER_H_
#include <memory>
#include <Spatify/lbvh.h>
#include <fem/types.h>
#include <fem/tet-mesh.h>
#include <fem/system.h>
#include <Core/utils.h>
#include <Maths/equations.h>
#include <Maths/inclusion-root-finder.h>

namespace fem::ipc {

constexpr Real kTCCDEpsilonEE = 6.217248937900877e-15;
constexpr Real kTCCDEpsilonVT = 6.661338147750939e-15;

using fem::System;

struct CCDQuery {
  Vector<Real, 3> x1;
  Vector<Real, 3> x2;
  Vector<Real, 3> x3;
  Vector<Real, 3> x4;
  Vector<Real, 3> u1;
  Vector<Real, 3> u2;
  Vector<Real, 3> u3;
  Vector<Real, 3> u4;
};

struct Trajectory {
  const System &system;
  const VecXd &p;
  Real toi = 1.0;
  int idx{};
};

inline BBox<Real, 3> computeTrajectoryBBox(const Trajectory &trajectory) {
  const auto &[system, p, toi, idx] = trajectory;
  BBox<Real, 3> box;
  for (int i = 0; i < 3; i++) {
    auto pos = system.currentPos(system.surfaces()(i, idx));
    auto u = p.segment<3>(3 * system.surfaces()(i, idx)) * toi;
    auto posNext = pos + u;
    box.expand({pos(0), pos(1), pos(2)}).expand({posNext(0), posNext(1), posNext(2)});
  }
  return box;
}

struct SystemTriangleTrajectoryAccessor {
  const System &system;
  const VecXd &p;
  Real toi = 1.0;
  using CoordType = Real;

  [[nodiscard]]
  size_t size() const {
    return system.numTriangles();
  }
  [[nodiscard]] BBox<Real, 3> bbox(int idx) const {
    return computeTrajectoryBBox({.system = system, .p = p, .toi = toi, .idx = idx});
  }
};
// reserved for TCCD implementation
struct VertexTriangleQuery {
  const std::array<Vector<Real, 3>, 4> &x;
  const std::array<Vector<Real, 3>, 4> &p;
  Vector<Real, 3> operator()(Real t, Real a, Real b) const {
    return (1.0 - a - b) * (x[1] + p[1] * t) + a * (x[2] + p[2] * t) + b * (x[3] + p[3] * t) - x[0] - p[0] * t;
  }
  [[nodiscard]] maths::MultiInterval<3> cubeEpsilon(const maths::MultiInterval<3> &I) const {
    std::array<Real, 8> vx{};
    std::array<Real, 8> vy{};
    std::array<Real, 8> vz{};
    for (int i = 0; i < 8; i++) {
      int m = i & 1;
      int n = (i >> 1) & 1;
      int l = (i >> 2) & 1;
      Real tm = I[0].bounds[m];
      Real an = I[1].bounds[n];
      Real bl = I[2].bounds[l];
      auto v = this->operator()(tm, an, bl);
      vx[i] = v(0);
      vy[i] = v(1);
      vz[i] = v(2);
    }
    Real gamma_x = std::max(*std::max_element(vx.begin(), vx.end()), 1.0);
    Real gamma_y = std::max(*std::max_element(vy.begin(), vy.end()), 1.0);
    Real gamma_z = std::max(*std::max_element(vz.begin(), vz.end()), 1.0);
    Real ex = gamma_x * gamma_x * gamma_x * kTCCDEpsilonVT;
    Real ey = gamma_y * gamma_y * gamma_y * kTCCDEpsilonVT;
    Real ez = gamma_z * gamma_z * gamma_z * kTCCDEpsilonVT;
    return {maths::Interval{.bounds = {-ex, ex}},
            maths::Interval{.bounds = {-ey, ey}},
            maths::Interval{.bounds = {-ez, ez}}};
  }
};

struct EdgeEdgeQuery {
  const std::array<Vector<Real, 3>, 4> &x;
  const std::array<Vector<Real, 3>, 4> &p;
  Vector<Real, 3> operator()(Real t, Real a, Real b) const {
    return (1.0 - a) * (x[0] + p[0] * t) + a * (x[1] + p[1] * t) + b * (x[2] + p[2] * t)
        + (1.0 - b) * (x[3] + p[3] * t);
  }
  [[nodiscard]] maths::MultiInterval<3> cubeEpsilon(const maths::MultiInterval<3> &I) const {
    std::array<Real, 8> vx{};
    std::array<Real, 8> vy{};
    std::array<Real, 8> vz{};
    for (int i = 0; i < 8; i++) {
      int m = i & 1;
      int n = (i >> 1) & 1;
      int l = (i >> 2) & 1;
      Real tm = I[0].bounds[m];
      Real an = I[1].bounds[n];
      Real bl = I[2].bounds[l];
      auto v = this->operator()(tm, an, bl);
      vx[i] = v(0);
      vy[i] = v(1);
      vz[i] = v(2);
    }
    Real gamma_x = std::max(*std::max_element(vx.begin(), vx.end()), 1.0);
    Real gamma_y = std::max(*std::max_element(vy.begin(), vy.end()), 1.0);
    Real gamma_z = std::max(*std::max_element(vz.begin(), vz.end()), 1.0);
    Real ex = gamma_x * gamma_x * gamma_x * kTCCDEpsilonEE;
    Real ey = gamma_y * gamma_y * gamma_y * kTCCDEpsilonEE;
    Real ez = gamma_z * gamma_z * gamma_z * kTCCDEpsilonEE;
    return {maths::Interval{.bounds = {-ex, ex}},
            maths::Interval{.bounds = {-ey, ey}},
            maths::Interval{.bounds = {-ez, ez}}};
  }
};

struct Contact {
  Vector<Real, 3> pos;
  Real t{1.0};
};

enum class CCDMode {
  EE,
  VT
};

struct ACCDOptions {
  CCDMode mode{CCDMode::EE};
  const CCDQuery &query;
  Real toi{1.0};
  std::optional<Real> reservedDistance{};
};

maths::CubicEquationRoots solveCoplanarTime(const CCDQuery &query, Real toi);
std::optional<Contact> eeCCD(const CCDQuery &query, Real toi);
std::optional<Contact> vtCCD(const CCDQuery &query, Real toi);

struct CollisionDetector {
  explicit CollisionDetector() : m_bvh(std::make_unique<spatify::LBVH<Real>>()) {}
  std::optional<Real> detect(const System &system, const VecXd &p);
  std::optional<Real> runACCD(const ACCDOptions &options);
  [[nodiscard]] const spatify::LBVH<Real> &bvh() const {
    return *m_bvh;
  }

 private:
  struct TrianglePairCCDQuery {
    const System &system;
    const VecXd &p;
    int ta, tb;
    Real toi;
  };
  std::optional<Real> trianglePairCCD(const TrianglePairCCDQuery &query);
  std::optional<Real> runACCD(CCDMode mode,
                              std::array<Vector<Real, 3>, 4> &x,
                              std::array<Vector<Real, 3>, 4> &p,
                              Real toi) const;
  std::optional<Real> runACCDReserved(CCDMode mode,
                                      std::array<Vector<Real, 3>, 4> &x,
                                      std::array<Vector<Real, 3>, 4> &p,
                                      Real toi,
                                      Real reservedDistance) const;
  Real s{0.1};
  std::unique_ptr<spatify::LBVH<Real>> m_bvh{};
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_COLLISION_RESOLVER_H_
