#include <FluidSim/cpu/fluid-simulator.h>
#include <FluidSim/cpu/project-solver.h>
#include <FluidSim/cpu/advect-solver.h>
#include <FluidSim/cpu/util.h>

namespace fluid {
static const std::array neighbours = {
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
                           const Array3D<uint8_t>& active,
                           Array3D<Real>& b) {
  b.parallelForEach([&](int i, int j, int k) {
    if (!active(i, j, k) || Adiag(i, j, k) == 0.0) return;
    Real t = Adiag(i, j, k) * x(i, j, k);
    if (i > 0 && active(i - 1, j, k))
      t += Aneighbour[Left](i, j, k) * x(i - 1, j, k);
    if (j > 0 && active(i, j - 1, k))
      t += Aneighbour[Down](i, j, k) * x(i, j - 1, k);
    if (k > 0 && active(i, j, k - 1))
      t += Aneighbour[Back](i, j, k) * x(i, j, k - 1);
    if (i < x.width() - 1 && active(i + 1, j, k))
      t += Aneighbour[Right](i, j, k) * x(i + 1, j, k);
    if (j < x.height() - 1 && active(i, j + 1, k))
      t += Aneighbour[Up](i, j, k) * x(i, j + 1, k);
    if (k < x.depth() - 1 && active(i, j, k + 1))
      t += Aneighbour[Front](i, j, k) * x(i, j, k + 1);
    b(i, j, k) = t;
    assert(!std::isnan(t));
  });
}

void FvmSolver3D::checkSystemSymmetry() const {
  Adiag.forEach([&](int i, int j, int k) {
    if (i == 0 || j == 0 || k == 0 || i == Adiag.width() - 1 || j == Adiag.
        height() - 1 || k == Adiag.depth() - 1)
      return;
    if (!active(i, j, k) || Adiag(i, j, k) == 0.0) return;
    if (i > 0 && active(i - 1, j, k))
      assert(approx(Aneighbour[Left](i, j, k), Aneighbour[Right](i - 1, j, k)));
    if (j > 0 && active(i, j - 1, k))
      assert(approx(Aneighbour[Down](i, j, k), Aneighbour[Up](i, j - 1, k)));
    if (k > 0 && active(i, j, k - 1))
      assert(approx(Aneighbour[Back](i, j, k), Aneighbour[Front](i, j, k - 1)));
    if (i < Adiag.width() - 1 && active(i + 1, j, k))
      assert(approx(Aneighbour[Right](i, j, k), Aneighbour[Left](i + 1, j, k)));
    if (j < Adiag.height() - 1 && active(i, j + 1, k))
      assert(approx(Aneighbour[Up](i, j, k), Aneighbour[Down](i, j + 1, k)));
    if (k < Adiag.depth() - 1 && active(i, j, k + 1))
      assert(approx(Aneighbour[Front](i, j, k), Aneighbour[Back](i, j, k + 1)));
  });
}

void FvmSolver3D::buildSystem(const FaceCentredGrid<Real, Real, 3, 0>& ug,
                              const FaceCentredGrid<Real, Real, 3, 1>& vg,
                              const FaceCentredGrid<Real, Real, 3, 2>& wg,
                              const SDF<3>& fluidSdf,
                              const SDF<3>& colliderSdf, Real dt) {
  Real h = ug.gridSpacing().x;
  Adiag.fill(0);
  Aneighbour[Left].fill(0);
  Aneighbour[Right].fill(0);
  Aneighbour[Up].fill(0);
  Aneighbour[Down].fill(0);
  Aneighbour[Front].fill(0);
  Aneighbour[Back].fill(0);
  uWeights.fill(0);
  vWeights.fill(0);
  wWeights.fill(0);
  uWeights.parallelForEach([&](int i, int j, int k) {
    if (i == 0) return;
    if (fluidSdf(i - 1, j, k) > 0.0 && fluidSdf(i, j, k) > 0.0)
      return;
    Vec3d p = ug.indexToCoord(i, j, k);
    Real bu = colliderSdf.eval(p + Vec3d(0.0, 1, -1) * 0.5 * h);
    Real bd = colliderSdf.eval(p + Vec3d(0.0, -1, -1) * 0.5 * h);
    Real fd = colliderSdf.eval(p + Vec3d(0.0, -1, 1) * 0.5 * h);
    Real fu = colliderSdf.eval(p + Vec3d(0.0, 1, 1) * 0.5 * h);
    Real frac = fractionInside(bu, bd, fd, fu);
    assert(frac >= 0.0 && frac <= 1.0);
    uWeights(i, j, k) = 1.0 - frac;
    assert(notNan(uWeights(i, j, k)));
  });
  vWeights.parallelForEach([&](int i, int j, int k) {
    if (j == 0) return;
    if (fluidSdf(i, j - 1, k) > 0.0 && fluidSdf(i, j, k) > 0.0)
      return;
    Vec3d p = vg.indexToCoord(i, j, k);
    Real lb = colliderSdf.eval(p + Vec3d(-1, 0.0, -1) * 0.5 * h);
    Real rb = colliderSdf.eval(p + Vec3d(1, 0.0, -1) * 0.5 * h);
    Real rf = colliderSdf.eval(p + Vec3d(1, 0.0, 1) * 0.5 * h);
    Real lf = colliderSdf.eval(p + Vec3d(-1, 0.0, 1) * 0.5 * h);
    Real frac = fractionInside(lb, rb, rf, lf);
    assert(frac >= 0.0 && frac <= 1.0);
    vWeights(i, j, k) = 1.0 - frac;
    assert(notNan(vWeights(i, j, k)));
  });
  wWeights.parallelForEach([&](int i, int j, int k) {
    if (k == 0) return;
    if (fluidSdf(i, j, k - 1) > 0.0 && fluidSdf(i, j, k) > 0.0)
      return;
    Vec3d p = wg.indexToCoord(i, j, k);
    Real ld = colliderSdf.eval(p + Vec3d(-1, -1, 0.0) * 0.5 * h);
    Real lu = colliderSdf.eval(p + Vec3d(-1, 1, 0.0) * 0.5 * h);
    Real ru = colliderSdf.eval(p + Vec3d(1, 1, 0.0) * 0.5 * h);
    Real rd = colliderSdf.eval(p + Vec3d(1, -1, 0.0) * 0.5 * h);
    Real frac = fractionInside(ld, lu, ru, rd);
    assert(frac >= 0.0 && frac <= 1.0);
    wWeights(i, j, k) = 1.0 - frac;
    assert(notNan(wWeights(i, j, k)));
  });

  Adiag.forEach([&](int i, int j, int k) {
    if (i == 0 || j == 0 || k == 0 || i == Adiag.width() - 1 || j == Adiag.
        height() - 1 || k == Adiag.depth()
        - 1) {
      active(i, j, k) = false;
      return;
    }
    if (uWeights(i, j, k) == 0.0 && uWeights(i + 1, j, k) == 0.0 &&
        vWeights(i, j, k) == 0.0 && vWeights(i, j + 1, k) == 0.0 &&
        wWeights(i, j, k) == 0.0 && wWeights(i, j, k + 1) == 0.0)
      return;
    if (fluidSdf(i, j, k) > 0.0) {
      active(i, j, k) = false;
      return;
    }
    active(i, j, k) = true;
    Real signed_dist = fluidSdf(i, j, k);
    Real factor = dt / h;
    assert(notNan(ug(i, j, k)));
    assert(notNan(vg(i, j, k)));
    assert(notNan(wg(i, j, k)));

    // left
    if (fluidSdf(i - 1, j, k) > 0.0) {
      Real theta = std::max(
          signed_dist / (signed_dist - fluidSdf(i - 1, j, k)), 0.01);
      Adiag(i, j, k) += uWeights(i, j, k) * factor / theta;
    } else {
      Adiag(i, j, k) += uWeights(i, j, k) * factor;
      Aneighbour[Left](i, j, k) -= uWeights(i, j, k) * factor;
    }
    rhs(i, j, k) += uWeights(i, j, k) * ug(i, j, k);

    // right
    if (fluidSdf(i + 1, j, k) > 0.0) {
      Real theta = std::max(
          signed_dist / (signed_dist - fluidSdf(i + 1, j, k)), 0.01);
      Adiag(i, j, k) += uWeights(i + 1, j, k) * factor / theta;
    } else {
      Adiag(i, j, k) += uWeights(i + 1, j, k) * factor;
      Aneighbour[Right](i, j, k) -= uWeights(i + 1, j, k) * factor;
    }
    rhs(i, j, k) -= uWeights(i + 1, j, k) * ug(i + 1, j, k);

    // down
    if (fluidSdf(i, j - 1, k) > 0.0) {
      Real theta = std::max(
          signed_dist / (signed_dist - fluidSdf(i, j - 1, k)), 0.01);
      Adiag(i, j, k) += vWeights(i, j, k) * factor / theta;
    } else {
      Adiag(i, j, k) += vWeights(i, j, k) * factor;
      Aneighbour[Down](i, j, k) -= vWeights(i, j, k) * factor;
    }
    rhs(i, j, k) += vWeights(i, j, k) * vg(i, j, k);
    // up
    if (fluidSdf(i, j + 1, k) > 0.0) {
      Real theta = std::max(
          signed_dist / (signed_dist - fluidSdf(i, j + 1, k)), 0.01);
      Adiag(i, j, k) += vWeights(i, j + 1, k) * factor / theta;
    } else {
      Adiag(i, j, k) += vWeights(i, j + 1, k) * factor;
      Aneighbour[Up](i, j, k) -= vWeights(i, j + 1, k) * factor;
    }
    rhs(i, j, k) -= vWeights(i, j + 1, k) * vg(i, j + 1, k);
    // back
    if (fluidSdf(i, j, k - 1) > 0.0) {
      Real theta = std::max(
          signed_dist / (signed_dist - fluidSdf(i, j, k - 1)), 0.01);
      Adiag(i, j, k) += wWeights(i, j, k) * factor / theta;
    } else {
      Adiag(i, j, k) += wWeights(i, j, k) * factor;
      Aneighbour[Back](i, j, k) -= wWeights(i, j, k) * factor;
    }
    rhs(i, j, k) += wWeights(i, j, k) * wg(i, j, k);

    // front
    if (fluidSdf(i, j, k + 1) > 0.0) {
      Real theta = std::max(
          signed_dist / (signed_dist - fluidSdf(i, j, k + 1)), 0.01);
      Adiag(i, j, k) += wWeights(i, j, k + 1) * factor / theta;
    } else {
      Adiag(i, j, k) += wWeights(i, j, k + 1) * factor;
      Aneighbour[Front](i, j, k) -= wWeights(i, j, k + 1) * factor;
    }
    rhs(i, j, k) -= wWeights(i, j, k + 1) * wg(i, j, k + 1);
    assert(notNan(rhs(i, j, k)));
  });
  checkSystemSymmetry();
}

void ModifiedIncompleteCholesky3D::precond(
    const Array3D<Real>& Adiag,
    const std::array<Array3D<Real>, 6>& Aneighbour,
    const Array3D<Real>& r,
    const Array3D<uint8_t>& active,
    Array3D<Real>& z) {
  // before this we have guaranteed that A is symmetric
  E.forEach([&](int i, int j, int k) {
    if (!active(i, j, k)) return;
    Real e = Adiag(i, j, k);
    assert(notNan(e));
    if (i > 0 && active(i - 1, j, k)) {
      e -= sqr(Aneighbour[Left](i, j, k) * E(i - 1, j, k));
      if (j < E.height() - 1 && active(i - 1, j + 1, k))
        e -= tau * Aneighbour[Left](i, j, k) * Aneighbour[Left](i - 1, j + 1, k)
            / sqr(E(i - 1, j, k));
      if (k < E.depth() - 1 && active(i - 1, j, k + 1))
        e -= tau * Aneighbour[Left](i, j, k) * Aneighbour[Left](i - 1, j, k + 1)
            / sqr(E(i - 1, j, k));
    }
    if (j > 0 && active(i, j - 1, k)) {
      e -= sqr(Aneighbour[Down](i, j, k) * E(i, j - 1, k));
      if (i < E.width() - 1 && active(i + 1, j - 1, k))
        e -= tau * Aneighbour[Down](i, j, k) * Aneighbour[Down](i + 1, j - 1, k)
            / sqr(E(i, j - 1, k));
      if (k < E.depth() - 1 && active(i, j - 1, k + 1))
        e -= tau * Aneighbour[Down](i, j, k) * Aneighbour[Down](i, j - 1, k + 1)
            / sqr(E(i, j - 1, k));
    }
    if (k > 0 && active(i, j, k - 1)) {
      e -= sqr(Aneighbour[Back](i, j, k) * E(i, j, k - 1));
      if (i < E.width() - 1 && active(i + 1, j, k - 1))
        e -= tau * Aneighbour[Back](i, j, k) * Aneighbour[Back](i + 1, j, k - 1)
            / sqr(E(i, j, k - 1));
      if (j < E.height() - 1 && active(i, j + 1, k - 1))
        e -= tau * Aneighbour[Back](i, j, k) * Aneighbour[Back](i, j + 1, k - 1)
            / sqr(E(i, j, k - 1));
    }
    if (e < sigma * Adiag(i, j, k)) e = Adiag(i, j, k);
    assert(e > 0.0);
    E(i, j, k) = 1.0 / std::sqrt(e);
  });
  // solve Lq = r
  q.forEach([&](int i, int j, int k) {
    if (!active(i, j, k)) return;
    Real t = r(i, j, k);
    assert(notNan(t));
    if (i > 0 && active(i - 1, j, k))
      t -= Aneighbour[Left](i, j, k) * E(i - 1, j, k) * q(i - 1, j, k);
    if (j > 0 && active(i, j - 1, k))
      t -= Aneighbour[Down](i, j, k) * E(i, j - 1, k) * q(i, j - 1, k);
    if (k > 0 && active(i, j, k - 1))
      t -= Aneighbour[Back](i, j, k) * E(i, j, k - 1) * q(i, j, k - 1);
    assert(notNan(E(i, j, k)));
    q(i, j, k) = t * E(i, j, k);
  });
  // solve L^Tz = q
  assert(
      z.width() == q.width() && z.height() == q.height() && z.depth() == q.depth
      ());
  z.forEachReversed([&](int i, int j, int k) {
    if (!active(i, j, k)) return;
    Real t = q(i, j, k);
    if (i < r.width() - 1 && active(i + 1, j, k))
      t -= Aneighbour[Left](i + 1, j, k) * E(i, j, k) * z(i + 1, j, k);
    if (j < r.height() - 1 && active(i, j + 1, k))
      t -= Aneighbour[Down](i, j + 1, k) * E(i, j, k) * z(i, j + 1, k);
    if (k < r.depth() - 1 && active(i, j, k + 1))
      t -= Aneighbour[Back](i, j, k + 1) * E(i, j, k) * z(i, j, k + 1);
    assert(notNan(E(i, j, k)));
    z(i, j, k) = t * E(i, j, k);
    assert(notNan(z(i, j, k)));
  });
}

Real CgSolver3D::solve(const Array3D<Real>& Adiag,
                       const std::array<Array3D<Real>, 6>& Aneighbour,
                       const Array3D<Real>& rhs,
                       const Array3D<uint8_t>& active,
                       Array3D<Real>& pg) {
  pg.fill(0);
  r.copyFrom(rhs);
  Real residual = LinfNorm(r, active);
  if (residual < tolerance)
    return residual;
  if (preconditioner)
    preconditioner->precond(Adiag, Aneighbour, rhs, active, z);
  else
    z.copyFrom(r);
  s.copyFrom(z);
  Real sigma = dotProduct(z, r, active);
  for (int i = 0; i < max_iterations; i++) {
    applyCompressedMatrix(Adiag, Aneighbour, s, active, z);
    Real sdotz = dotProduct(s, z, active);
    assert(sdotz != 0);
    Real alpha = sigma / sdotz;
    saxpy(pg, s, alpha, active);
    saxpy(r, z, -alpha, active);
    residual = LinfNorm(r, active);
    if (residual < tolerance) break;
    if (preconditioner)
      preconditioner->precond(Adiag, Aneighbour, r, active, z);
    Real sigma_new = dotProduct(z, r, active);
    assert(sigma != 0);
    Real beta = sigma_new / sigma;
    scaleAndAdd(s, z, beta, active);
    sigma = sigma_new;
  }
  return residual;
}

void FvmSolver3D::project(FaceCentredGrid<Real, Real, 3, 0>& ug,
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
    assert(notNan(pg(i, j, k)));
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
    assert(notNan(pg(i, j, k)));
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
    assert(notNan(pg(i, j, k)));
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