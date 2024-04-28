//
// Created by creeper on 13/1/33.
//
#include <Core/animation.h>
#include <Core/rand-gen.h>
#include <FluidSim/cpu/fluid-simulator.h>
#include <FluidSim/cpu/advect-solver.h>
#include <FluidSim/cpu/project-solver.h>
#include <FluidSim/cpu/util.h>
#include <cassert>

namespace fluid::cpu {
void FluidSimulator::applyForce(Real dt) const {
  vg->forEach([this, dt](int i, int j, int k) {
    if (vValid->at(i, j, k) && j > 0 && j < vg->height() - 1)
      vg->at(i, j, k) -= 9.8 * dt;
  });
}

void FluidSimulator::clear() {
  uValid->fill(0);
  vValid->fill(0);
  wValid->fill(0);
  uValidBuf->fill(0);
  vValidBuf->fill(0);
  wValidBuf->fill(0);
  vg->fill(0);
  ug->fill(0);
  wg->fill(0);
  ubuf->fill(0);
  vbuf->fill(0);
  wbuf->fill(0);
  uw.fill(0);
  vw.fill(0);
  ww.fill(0);
  pg.fill(0);
}

void FluidSimulator::smoothFluidSurface(int iters) {
  for (int iter = 0; iter < iters; iter++) {
    fluidSurfaceBuf->grid.forEach([&](int i, int j, int k) {
      Real sum = 0;
      int count = 0;
      if (i > 0) {
        sum += fluidSurface->grid(i - 1, j, k);
        count++;
      }
      if (i < fluidSurface->width() - 1) {
        sum += fluidSurface->grid(i + 1, j, k);
        count++;
      }
      if (j > 0) {
        sum += fluidSurface->grid(i, j - 1, k);
        count++;
      }
      if (j < fluidSurface->height() - 1) {
        sum += fluidSurface->grid(i, j + 1, k);
        count++;
      }
      if (k > 0) {
        sum += fluidSurface->grid(i, j, k - 1);
        count++;
      }
      if (k < fluidSurface->depth() - 1) {
        sum += fluidSurface->grid(i, j, k + 1);
        count++;
      }
      fluidSurfaceBuf->grid(i, j, k) = sum / count;
      if (fluidSurface->grid(i, j, k) < fluidSurfaceBuf->grid(i, j, k))
        fluidSurfaceBuf->grid(i, j, k) = fluidSurface->grid(i, j, k);
    });
    std::swap(fluidSurface, fluidSurfaceBuf);
  }
}

void FluidSimulator::applyCollider() const {
  ug->parallelForEach([&](int i, int j, int k) {
    if (i == 0 || i == ug->width() - 1) {
      ug->at(i, j, k) = 0.0;
      return;
    }
    Vec3d pos{ug->indexToCoord(i, j, k)};
    if (colliderSdf->eval(pos) < 0.0) {
      // calc the normal, and project out the x component
      Vec3d normal{normalize(colliderSdf->grad(pos))};
      ug->at(i, j, k) -= ug->at(i, j, k) * normal.x;
    }
  });
  vg->parallelForEach([&](int i, int j, int k) {
    if (j == 0 || j == vg->height() - 1) {
      vg->at(i, j, k) = 0.0;
      return;
    }
    Vec3d pos = vg->indexToCoord(i, j, k);
    if (colliderSdf->eval(pos) < 0.0) {
      Vec3d normal{normalize(colliderSdf->grad(pos))};
      vg->at(i, j, k) -= vg->at(i, j, k) * normal.y;
    }
  });
  wg->parallelForEach([&](int i, int j, int k) {
    if (k == 0 || k == wg->depth() - 1) {
      wg->at(i, j, k) = 0.0;
      return;
    }
    Vec3d pos{wg->indexToCoord(i, j, k)};
    if (colliderSdf->eval(pos) < 0.0) {
      Vec3d normal{normalize(colliderSdf->grad(pos))};
      wg->at(i, j, k) -= wg->at(i, j, k) * normal.z;
    }
  });
}

void FluidSimulator::applyDirichletBoundary() const {
  for (int j = 0; j < ug->height(); j++) {
    for (int k = 0; k < ug->depth(); k++) {
      ug->at(0, j, k) = 0.0;
      uValid->at(0, j, k) = 1;
      ug->at(ug->width() - 1, j, k) = 0.0;
      uValid->at(ug->width() - 1, j, k) = 1;
    }
  }
  for (int i = 0; i < vg->width(); i++) {
    for (int k = 0; k < vg->depth(); k++) {
      vg->at(i, 0, k) = 0.0;
      vValid->at(i, 0, k) = 1;
      vg->at(i, vg->height() - 1, k) = 0.0;
      vValid->at(i, vg->height() - 1, k) = 1;
    }
  }
  for (int i = 0; i < wg->width(); i++) {
    for (int j = 0; j < wg->height(); j++) {
      wg->at(i, j, 0) = 0.0;
      wValid->at(i, j, 0) = 1;
      wg->at(i, j, wg->depth() - 1) = 0.0;
      wValid->at(i, j, wg->depth() - 1) = 1;
    }
  }
}

void FluidSimulator::substep(Real dt) {
  clear();
  std::cout << "Solving advection... ";
  advector->advect(std::span(positions()), *ug, *vg, *wg, *colliderSdf, dt);
  std::cout << "Done." << std::endl;
  std::cout << "Reconstructing surface... ";
  fluidSurfaceReconstructor->reconstruct(
      m_particles.positions, 0.6 * ug->gridSpacing().x,
      *fluidSurface, *sdfValid);
  std::cout << "Done." << std::endl;
  std::cout << "Smoothing surface... ";
  extrapolateFluidSdf(10);
  smoothFluidSurface(5);
  std::cout << "Done." << std::endl;
  std::cout << "Solving P2G... ";
  advector->solveP2G(std::span(positions()), *ug, *vg, *wg,
                     *colliderSdf, uw, vw, ww, *uValid, *vValid, *wValid, dt);
  applyDirichletBoundary();
  std::cout << "Done." << std::endl;
  std::cout << "Extrapolating velocities... ";
  extrapolate(ug, ubuf, uValid, uValidBuf, 10);
  extrapolate(vg, vbuf, vValid, vValidBuf, 10);
  extrapolate(wg, wbuf, wValid, wValidBuf, 10);
  std::cout << "Done." << std::endl;
  applyForce(dt);
  std::cout << "Building linear system... ";
  projector->buildSystem(*ug, *vg, *wg, *fluidSurface, *colliderSdf, dt);
  std::cout << "Done." << std::endl;
  std::cout << "Solving linear system... ";
  if (Real residual{projector->solvePressure(*fluidSurface, pg)};
    residual > 1e-4)
    std::cerr << "Warning: projection residual is " << residual << std::endl;
  else std::cout << "Projection residual is " << residual << std::endl;
  std::cout << "Done." << std::endl;
  std::cout << "Doing projection and applying collider... ";
  projector->project(*ug, *vg, *wg, pg, *fluidSurface, *colliderSdf, dt);
  applyCollider();
  std::cout << "Done." << std::endl;
  std::cout << "Solving G2P... ";
  advector->solveG2P(std::span(positions()), *ug, *vg, *wg,
                     *colliderSdf, dt);
  std::cout << "Done" << std::endl;
}

Real FluidSimulator::CFL() const {
  Real h{ug->gridSpacing().x};
  Real cfl{h / 1e-6};
  ug->forEach([&cfl, h, this](int x, int y, int z) {
    assert(notNan(ug->at(x, y ,z)));
    if (ug->at(x, y, z) != 0.0)
      cfl = std::min(cfl, h / abs(ug->at(x, y, z)));
  });
  vg->forEach([&cfl, h, this](int x, int y, int z) {
    assert(notNan(vg->at(x, y, z)));
    if (vg->at(x, y, z) != 0.0)
      cfl = std::min(cfl, h / abs(vg->at(x, y, z)));
  });
  wg->forEach([&cfl, h, this](int x, int y, int z) {
    assert(notNan(wg->at(x, y, z)));
    if (wg->at(x, y, z) != 0.0)
      cfl = std::min(cfl, h / abs(wg->at(x, y, z)));
  });
//  return 10.0 * std::max(1e-4, cfl);
return 1.0;
}

void FluidSimulator::step(core::Frame& frame) {
  Real t = 0;
  std::cout << std::format("********* Frame {} *********", frame.idx) <<
      std::endl;
  int substep_cnt = 0;
  while (t < frame.dt) {
    Real cfl = CFL();
    Real dt = std::min(cfl, frame.dt - t);
    substep_cnt++;
    std::cout << std::format("<<<<< Substep {}, dt = {} >>>>>", substep_cnt, dt)
        << std::endl;
    substep(dt);
    t += dt;
  }
  frame.onAdvance();
}
} // namespace fluid