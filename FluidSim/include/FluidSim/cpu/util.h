// add header guard, use full path
#ifndef FLUIDSIM_COMMON_ADVECT_SOLVER_H_
#define FLUIDSIM_COMMON_ADVECT_SOLVER_H_

#include <FluidSim/fluid-sim.h>
#include <Spatify/grids.h>
#include <Core/utils.h>
#include <Spatify/arrays.h>

namespace fluid {
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
using core::sqr;
using core::normalize;
using core::dot;

Real dotProduct(const spatify::Array3D<Real>& a,
                const spatify::Array3D<Real>& b);
void saxpy(spatify::Array3D<Real>& a, const spatify::Array3D<Real>& b, Real x);
void scaleAndAdd(spatify::Array3D<Real>& a, const spatify::Array3D<Real>& b,
                 Real x);

inline Real LinfNorm(spatify::Array3D<Real>& a) {
  Real maxv = 0;
  a.forEach([&](int i, int j, int k) {
    maxv = std::max(maxv, std::abs(a(i, j, k)));
  });
  return maxv;
}
}

#endif