#include <FluidSim/common/fluid-simulator.h>
#include <FluidSim/common/project-solver.h>
#include <FluidSim/common/util.h>

#include "FluidSim/common/advect-solver.h"


namespace fluid {
void FvmSolver2D::buildSystem(const spatify::GhostArray2D<Marker, 1>& markers,
                              const FaceCentredGrid<Real, Real, 2, 0>* ug,
                              const FaceCentredGrid<Real, Real, 2, 1>* vg,
                              const SDF<2>& collider_sdf,
                              Real density, Real dt) {
  Real h{ug->gridSpacing().x};
  for (int x = 0; x < rhs.width(); x++)
    for (int y = 0; y < rhs.height(); y++) {
      if (markers(x, y) == Marker::Solid || markers(x, y) == Marker::Air)
        return;
      active_cells.emplace_back(x, y);
      rhs(x, y) =
          -(ug->at(x + 1, y) - ug->at(x, y)) * density * h -
          (vg->at(x, y + 1) - vg->at(x, y)) * density * h;
      if (markers(x - 1, y) == Marker::Solid)
        rhs(x, y) -= ug->at(x, y) / h;
      if (markers(x + 1, y) == Marker::Solid)
        rhs(x, y) += ug->at(x + 1, y) / h;
      if (markers(x, y - 1) == Marker::Solid)
        rhs(x, y) -= vg->at(x, y) / h;
      if (markers(x, y + 1) == Marker::Solid)
        rhs(x, y) += vg->at(x, y + 1) / h;
      active_cells.emplace_back(x, y);
      rhs(x, y) /= dt;
    }
}

static const std::array<Vec3i, 6> neighbours = {
    Vec3i(-1, 0, 0), Vec3i(1, 0, 0), Vec3i(0, -1, 0),
    Vec3i(0, 1, 0), Vec3i(0, 0, -1), Vec3i(0, 0, 1),
};

enum Directions {
  Left,
  Right,
  Up,
  Down,
  Front,
  Back
};

void applyCompressedMatrix(const Array3D<Real>& Adiag,
                           const std::array<Array3D<Real>, 6>&
                           Aneighbour, const Array3D<Real>& x,
                           Array3D<Real>& b) {
  b.parallelForEach([&](int i, int j, int k) {
    Real t = Adiag(i, j, k) * x(i, j, k);
    if (i > 0) t += Aneighbour[Left](i, j, k) * x(i - 1, j, k);
    if (j > 0) t += Aneighbour[Down](i, j, k) * x(i, j - 1, k);
    if (k > 0) t += Aneighbour[Back](i, j, k) * x(i, j, k - 1);
    if (i < x.width() - 1) t += Aneighbour[Right](i, j, k) * x(i + 1, j, k);
    if (j < x.height() - 1) t += Aneighbour[Up](i, j, k) * x(i, j + 1, k);
    if (k < x.depth() - 1) t += Aneighbour[Front](i, j, k) * x(i, j, k + 1);
    b(i, j, k) = t;
  });
}

void FvmSolver3D::buildSystem(const spatify::GhostArray3D<Marker, 1>& markers,
                              const FaceCentredGrid<Real, Real, 3, 0>* ug,
                              const FaceCentredGrid<Real, Real, 3, 1>* vg,
                              const FaceCentredGrid<Real, Real, 3, 2>* wg,
                              const SDF<3>* fluid_sdf,
                              const SDF<3>& collider_sdf,
                              Real density, Real dt) {
  Real h = ug->gridSpacing().x;
  uWeights.parallelForEach([&](int i, int j, int k) {
    Vec3d p = ug->indexToCoord(i, j, k);
    Real bu = collider_sdf.eval(p + Vec3d(0.0, 1, -1) * 0.5 * h);
    Real bd = collider_sdf.eval(p + Vec3d(0.0, -1, -1) * 0.5 * h);
    Real fd = collider_sdf.eval(p + Vec3d(0.0, -1, 1) * 0.5 * h);
    Real fu = collider_sdf.eval(p + Vec3d(0.0, 1, 1) * 0.5 * h);
    uWeights(i, j, k) = clamp(1.0 - fractionInside(bu, bd, fd, fu), 0.0, 1.0);
  });
  vWeights.parallelForEach([&](int i, int j, int k) {
    Vec3d p = vg->indexToCoord(i, j, k);
    Real lb = collider_sdf.eval(p + Vec3d(-1, 0.0, -1) * 0.5 * h);
    Real rb = collider_sdf.eval(p + Vec3d(1, 0.0, -1) * 0.5 * h);
    Real rf = collider_sdf.eval(p + Vec3d(1, 0.0, 1) * 0.5 * h);
    Real lf = collider_sdf.eval(p + Vec3d(-1, 0.0, 1) * 0.5 * h);
    vWeights(i, j, k) = clamp(1.0 - fractionInside(lb, rb, rf, lf), 0.0, 1.0);
  });
  wWeights.parallelForEach([&](int i, int j, int k) {
    Vec3d p = wg->indexToCoord(i, j, k);
    Real ld = collider_sdf.eval(p + Vec3d(-1, -1, 0.0) * 0.5 * h);
    Real lu = collider_sdf.eval(p + Vec3d(-1, 1, 0.0) * 0.5 * h);
    Real ru = collider_sdf.eval(p + Vec3d(1, 1, 0.0) * 0.5 * h);
    Real rd = collider_sdf.eval(p + Vec3d(1, -1, 0.0) * 0.5 * h);
    wWeights(i, j, k) = clamp(1.0 - fractionInside(ld, lu, ru, rd), 0.0, 1.0);
  });
  Adiag.parallelForEach([&](int i, int j, int k) {
    if (markers(i, j, k) != Marker::Fluid) return;
    if (markers(i - 1, j, k) != Marker::Fluid) {
    }
  });
}

void ModifiedIncompleteCholesky3D::precond(const Array3D<Real>& Adiag,
                                           const std::array<Array3D<Real>, 6>&
                                           Aneighbour, const Array3D<Real>& r,
                                           Array3D<Real>& z) {
  E.parallelForEach([&](int i, int j, int k) {
    Real e = Adiag(i, j, k);
    if (i > 0) e -= sqr(Aneighbour[Left](i, j, k) * E(i - 1, j, k));
    if (j > 0) e -= sqr(Aneighbour[Down](i, j, k) * E(i, j - 1, k));
    if (k > 0) e -= sqr(Aneighbour[Back](i, j, k) * E(i, j, k - 1));
    if (i > 0 && j < r.height() - 1 && k < r.depth() - 1)
      e -= tau * Aneighbour[Left](i, j, k) * (
            Aneighbour[Left](i - 1, j + 1, k) + Aneighbour[Left](
                i - 1, j, k + 1)) /
          sqr(E(i - 1, j, k));
    if (i < r.width() - 1 && j > 0 && k < r.depth() - 1)
      e -= tau * Aneighbour[Down](i, j, k) * (
            Aneighbour[Down](i + 1, j - 1, k) + Aneighbour[Down](
                i, j - 1, k + 1)) /
          sqr(E(i, j - 1, k));
    if (i < r.width() - 1 && j < r.height() - 1 && k > 0)
      e -= tau * Aneighbour[Back](i, j, k) * (
            Aneighbour[Back](i + 1, j, k - 1) + Aneighbour[Back](
                i, j + 1, k - 1)) /
          sqr(E(i, j, k - 1));
    if (e < sigma * Adiag(i, j, k)) e = Adiag(i, j, k);
    E(i, j, k) = 1.0 / std::sqrt(e);
  });
  // solve Lq = r
  q.forEach([&](int i, int j, int k) {
    Real t = r(i, j, k);
    if (i > 0) t -= Aneighbour[Left](i, j, k) * E(i - 1, j, k) * q(i - 1, j, k);
    if (j > 0) t -= Aneighbour[Down](i, j, k) * E(i, j - 1, k) * q(i, j - 1, k);
    if (k > 0) t -= Aneighbour[Back](i, j, k) * E(i, j, k - 1) * q(i, j, k - 1);
    q(i, j, k) = t * E(i, j, k);
  });
  // solve L^Tz = q
  z.forEachReversed([&](int i, int j, int k) {
    Real t = q(i, j, k);
    if (i < r.width() - 1)
      t -= Aneighbour[Left](i + 1, j, k) * E(i, j, k) * z(i + 1, j, k);
    if (j < r.height() - 1)
      t -= Aneighbour[Down](i, j + 1, k) * E(i, j, k) * z(i, j + 1, k);
    if (k < r.depth() - 1)
      t -= Aneighbour[Back](i, j, k + 1) * E(i, j, k) * z(i, j, k + 1);
    z(i, j, k) = t * E(i, j, k);
  });
}

Real CgSolver3D::solve(const Array3D<Real>& Adiag,
                       const std::array<Array3D<Real>, 6>& Aneighbour,
                       const Array3D<Real>& rhs,
                       Array3D<Real>& pg) {
  pg.fill(0);
  r.copyFrom(rhs);
  if (preconditioner == nullptr)
    z.copyFrom(r);
  else
    preconditioner->precond(Adiag, Aneighbour, rhs, z);
  s.copyFrom(z);
  Real sigma = dotProduct(z, r);
  Real residual{};
  for (int i = 0; i < max_iterations; i++) {
    applyCompressedMatrix(Adiag, Aneighbour, s, z);
    Real alpha = sigma / dotProduct(s, z);
    saxpy(pg, s, alpha);
    saxpy(r, z, -alpha);
    residual = LinfNorm(r);
    if (residual < tolerance) break;
    if (preconditioner)
      preconditioner->precond(Adiag, Aneighbour, r, z);
    Real sigma_new = dotProduct(z, r);
    Real beta = sigma_new / sigma;
    scaleAndAdd(s, z, beta);
    sigma = sigma_new;
  }
  return residual;
}
}