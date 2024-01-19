// add header guard, use full path
#ifndef FLUIDSIM_COMMON_ADVECT_SOLVER_H_
#define FLUIDSIM_COMMON_ADVECT_SOLVER_H_

#include <FluidSim/fluid-sim.h>
#include <Spatify/grids.h>
#include <Core/utils.h>
#include <Spatify/arrays.h>

namespace fluid {
using spatify::FaceCentredGrid;
using spatify::Array3D;
Vec2d sampleVelocity(const Vec2d& p,
                     const FaceCentredGrid<Real, Real, 2, 0>& u_grid,
                     const FaceCentredGrid<Real, Real, 2, 1>& v_grid);
Vec3d sampleVelocity(const Vec3d& p,
                     const FaceCentredGrid<Real, Real, 3, 0>& u_grid,
                     const FaceCentredGrid<Real, Real, 3, 1>& v_grid,
                     const FaceCentredGrid<Real, Real, 3, 2>& w_grid);
Real fractionInside(Real lu, Real ru, Real rd, Real ld);
template <typename T>
inline T clamp(T x, T min, T max) {
  return x < min ? min : (x > max ? max : x);
}

template <typename T>
inline bool approx(T x, T y) {
  if constexpr (std::is_same_v<T, Real>)
    return std::abs(x - y) < 1e-6;
  else if (std::is_same_v<T, float>)
    return std::abs(x - y) < 1e-3;
  else return x == y;
}

inline bool notNan(Real x) {
  return x == x;
}

using core::sqr;
using core::normalize;
using core::dot;

Real dotProduct(const spatify::Array3D<Real>& a,
                const spatify::Array3D<Real>& b,
                const spatify::Array3D<uint8_t>& active);
void saxpy(spatify::Array3D<Real>& a, const spatify::Array3D<Real>& b, Real x,
           const spatify::Array3D<uint8_t>& active);
void scaleAndAdd(spatify::Array3D<Real>& a, const spatify::Array3D<Real>& b,
                 Real x, const spatify::Array3D<uint8_t>& active);

inline Real LinfNorm(spatify::Array3D<Real>& a, const spatify::Array3D<uint8_t>& active) {
  Real maxv = 0;
  a.forEach([&](int i, int j, int k) {
    if (!active(i, j, k)) return;
    maxv = std::max(maxv, std::abs(a(i, j, k)));
  });
  return maxv;
}
}

#endif