#include <FluidSim/cpu/fluid-simulator.h>
#include <FluidSim/cpu/project-solver.h>
#include <FluidSim/cpu/advect-solver.h>
#include <FluidSim/cpu/util.h>

namespace fluid {
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

void FvmSolver3D::buildSystem(const GhostArray3D<Marker, 1>& markers,
                              const FaceCentredGrid<Real, Real, 3, 0>& ug,
                              const FaceCentredGrid<Real, Real, 3, 1>& vg,
                              const FaceCentredGrid<Real, Real, 3, 2>& wg,
                              const SDF<3>& fluid_sdf,
                              const SDF<3>& collider_sdf,
                              Real dt) {
  Real h = ug.gridSpacing().x;
  Adiag.fill(0);
  Aneighbour[Left].fill(0);
  Aneighbour[Right].fill(0);
  Aneighbour[Up].fill(0);
  Aneighbour[Down].fill(0);
  Aneighbour[Front].fill(0);
  Aneighbour[Back].fill(0);
  uWeights.parallelForEach([&](int i, int j, int k) {
    if (markers(i - 1, j, k) != Marker::Fluid && markers(i, j, k) !=
        Marker::Fluid)
      return;
    Vec3d p = ug.indexToCoord(i, j, k);
    Real bu = collider_sdf.eval(p + Vec3d(0.0, 1, -1) * 0.5 * h);
    Real bd = collider_sdf.eval(p + Vec3d(0.0, -1, -1) * 0.5 * h);
    Real fd = collider_sdf.eval(p + Vec3d(0.0, -1, 1) * 0.5 * h);
    Real fu = collider_sdf.eval(p + Vec3d(0.0, 1, 1) * 0.5 * h);
    uWeights(i, j, k) = clamp(1.0 - fractionInside(bu, bd, fd, fu), 0.0, 1.0);
  });
  vWeights.parallelForEach([&](int i, int j, int k) {
    if (markers(i, j - 1, k) != Marker::Fluid && markers(i, j, k) !=
        Marker::Fluid)
      return;
    Vec3d p = vg.indexToCoord(i, j, k);
    Real lb = collider_sdf.eval(p + Vec3d(-1, 0.0, -1) * 0.5 * h);
    Real rb = collider_sdf.eval(p + Vec3d(1, 0.0, -1) * 0.5 * h);
    Real rf = collider_sdf.eval(p + Vec3d(1, 0.0, 1) * 0.5 * h);
    Real lf = collider_sdf.eval(p + Vec3d(-1, 0.0, 1) * 0.5 * h);
    vWeights(i, j, k) = clamp(1.0 - fractionInside(lb, rb, rf, lf), 0.0, 1.0);
  });
  wWeights.parallelForEach([&](int i, int j, int k) {
    if (markers(i, j - 1, k) != Marker::Fluid && markers(i, j, k) !=
        Marker::Fluid)
      return;
    Vec3d p = wg.indexToCoord(i, j, k);
    Real ld = collider_sdf.eval(p + Vec3d(-1, -1, 0.0) * 0.5 * h);
    Real lu = collider_sdf.eval(p + Vec3d(-1, 1, 0.0) * 0.5 * h);
    Real ru = collider_sdf.eval(p + Vec3d(1, 1, 0.0) * 0.5 * h);
    Real rd = collider_sdf.eval(p + Vec3d(1, -1, 0.0) * 0.5 * h);
    wWeights(i, j, k) = clamp(1.0 - fractionInside(ld, lu, ru, rd), 0.0, 1.0);
  });
  Adiag.parallelForEach([&](int i, int j, int k) {
    if (markers(i, j, k) != Marker::Fluid) return;
    // if it is fluid, then += uWeights *
    Real fsd = fluid_sdf.grid(i, j, k);
    Real factor = dt / h;
    // left
    if (fluid_sdf.grid(i - 1, j, k) > 0.0) {
      Real theta = std::max(fsd / (fsd - fluid_sdf.grid(i - 1, j, k)), 0.01);
      Adiag(i, j, k) += uWeights(i, j, k) * factor / theta;
    } else {
      Adiag(i, j, k) += uWeights(i, j, k) * factor;
      Aneighbour[Left](i, j, k) = -uWeights(i, j, k) * factor;
    }
    rhs(i, j, k) += uWeights(i, j, k) * ug(i, j, k);

    // right
    if (fluid_sdf.grid(i + 1, j, k) > 0.0) {
      Real theta = std::max(fsd / (fsd - fluid_sdf.grid(i + 1, j, k)), 0.01);
      Adiag(i, j, k) += uWeights(i + 1, j, k) * factor / theta;
    } else {
      Adiag(i, j, k) += uWeights(i + 1, j, k) * factor;
      Aneighbour[Right](i, j, k) = -uWeights(i + 1, j, k) * factor;
    }
    rhs(i, j, k) -= uWeights(i + 1, j, k) * ug(i + 1, j, k);

    // down
    if (fluid_sdf.grid(i, j - 1, k) > 0.0) {
      Real theta = std::max(fsd / (fsd - fluid_sdf.grid(i, j - 1, k)), 0.01);
      Adiag(i, j, k) += vWeights(i, j, k) * factor / theta;
    } else {
      Adiag(i, j, k) += vWeights(i, j, k) * factor;
      Aneighbour[Down](i, j, k) = -vWeights(i, j, k) * factor;
    }
    rhs(i, j, k) += vWeights(i, j, k) * vg(i, j, k);

    // up
    if (fluid_sdf.grid(i, j + 1, k) > 0.0) {
      Real theta = std::max(fsd / (fsd - fluid_sdf.grid(i, j + 1, k)), 0.01);
      Adiag(i, j, k) += vWeights(i, j + 1, k) * factor / theta;
    } else {
      Adiag(i, j, k) += vWeights(i, j + 1, k) * factor;
      Aneighbour[Up](i, j, k) = -vWeights(i, j + 1, k) * factor;
    }
    rhs(i, j, k) -= vWeights(i, j + 1, k) * vg(i, j + 1, k);

    // back
    if (fluid_sdf.grid(i, j, k - 1) > 0.0) {
      Real theta = std::max(fsd / (fsd - fluid_sdf.grid(i, j, k - 1)), 0.01);
      Adiag(i, j, k) += wWeights(i, j, k) * factor / theta;
    } else {
      Adiag(i, j, k) += wWeights(i, j, k) * factor;
      Aneighbour[Back](i, j, k) = -wWeights(i, j, k) * factor;
    }
    rhs(i, j, k) += wWeights(i, j, k) * wg(i, j, k);

    // front
    if (markers(i, j, k + 1) != Marker::Fluid) {
      Real theta = std::max(fsd / (fsd - fluid_sdf.grid(i, j, k + 1)), 0.01);
      Adiag(i, j, k) += wWeights(i, j, k + 1) * factor / theta;
    } else {
      Adiag(i, j, k) += wWeights(i, j, k + 1) * factor;
      Aneighbour[Front](i, j, k) = -wWeights(i, j, k + 1) * factor;
    }
    rhs(i, j, k) -= wWeights(i, j, k + 1) * wg(i, j, k + 1);
  });
}

void ModifiedIncompleteCholesky3D::precond(
    const GhostArray3D<Marker, 1>& markers,
    const Array3D<Real>& Adiag,
    const std::array<Array3D<Real>, 6>&
    Aneighbour,
    const Array3D<Real>& r,
    Array3D<Real>& z) {
  E.parallelForEach([&](int i, int j, int k) {
    if (markers(i, j, k) != Marker::Fluid) return;
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

Real CgSolver3D::solve(const GhostArray3D<Marker, 1>& markers,
                       const Array3D<Real>& Adiag,
                       const std::array<Array3D<Real>, 6>& Aneighbour,
                       const Array3D<Real>& rhs, Array3D<Real>& pg) {
  pg.fill(0);
  r.copyFrom(rhs);
  if (preconditioner)
    preconditioner->precond(markers, Adiag, Aneighbour, rhs, z);
  else
    z.copyFrom(r);
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
      preconditioner->precond(markers, Adiag, Aneighbour, r, z);
    Real sigma_new = dotProduct(z, r);
    Real beta = sigma_new / sigma;
    scaleAndAdd(s, z, beta);
    sigma = sigma_new;
  }
  return residual;
}

void FvmSolver3D::project(const GhostArray3D<Marker, 1>& markers,
                          FaceCentredGrid<Real, Real, 3, 0>& ug,
                          FaceCentredGrid<Real, Real, 3, 1>& vg,
                          FaceCentredGrid<Real, Real, 3, 2>& wg,
                          const Array3D<Real>& pg,
                          const SDF<3>& fluid_sdf,
                          const SDF<3>& collider_sdf,
                          Real dt) {
  Real h = ug.gridSpacing().x;
  ug.parallelForEach([&](int i, int j, int k) {
    if (uWeights(i, j, k) <= 0.0) return;
    Real theta{1.0};
    if (i == 0 || i == ug.width() - 1) {
      ug(i, j, k) = 0.0;
      return;
    }
    Real sd_left = fluid_sdf.grid(i - 1, j, k);
    Real sd_right = fluid_sdf.grid(i, j, k);
    if (sd_left >= 0.0 && sd_right >= 0.0) return;
    if (sd_left < 0.0 && sd_right < 0.0) {
      ug(i, j, k) -= (pg(i, j, k) - pg(i - 1, j, k)) * dt / h;
      return;
    }
    theta = std::max(sd_left / (sd_left - sd_right), 0.01);
    ug(i, j, k) -= pg(i - 1, j, k) * dt / h / theta;
  });
  vg.parallelForEach([&](int i, int j, int k) {
    if (vWeights(i, j, k) <= 0.0) return;
    Real theta{1.0};
    if (j == 0 || j == vg.height() - 1) {
      vg(i, j, k) = 0.0;
      return;
    }
    Real sd_down = fluid_sdf.grid(i - 1, j, k);
    Real sd_up = fluid_sdf.grid(i, j, k);
    if (sd_down >= 0.0 && sd_up >= 0.0) return;
    if (sd_down < 0.0 && sd_up < 0.0) {
      vg(i, j, k) -= (pg(i, j, k) - pg(i, j - 1, k)) * dt / h;
      return;
    }
    theta = std::max(sd_down / (sd_down - sd_up), 0.01);
    vg(i, j, k) -= pg(i, j - 1, k) * dt / h / theta;
  });
  wg.parallelForEach([&](int i, int j, int k) {
    if (wWeights(i, j, k) <= 0.0) return;
    Real theta{1.0};
    if (k == 0 || k == wg.depth() - 1) {
      wg(i, j, k) = 0.0;
      return;
    }
    Real sd_back = fluid_sdf.grid(i, j, k - 1);
    Real sd_front = fluid_sdf.grid(i, j, k);
    if (sd_back >= 0.0 && sd_front >= 0.0) return;
    if (sd_back < 0.0 && sd_front < 0.0) {
      wg(i, j, k) -= (pg(i, j, k) - pg(i - 1, j, k)) * dt / h;
      return;
    }
    theta = std::max(sd_back / (sd_back - sd_front), 0.01);
    wg(i, j, k) -= pg(i, j, k - 1) * dt / h / theta;
  });
}
}