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

namespace fluid {
void HybridFluidSimulator3D::applyForce(Real dt) const {
  vg->forEach([this, dt](int i, int j, int k) {
    vg->at(i, j, k) -= 9.8 * dt;
  });
}

void HybridFluidSimulator3D::clear() {
  ug->fill(0);
  ubuf->fill(0);
  vg->fill(0);
  vbuf->fill(0);
  wg->fill(0);
  wbuf->fill(0);
  pg.fill(0);
}

void HybridFluidSimulator3D::smoothFluidSurface(int iters) {
  for (int i = 0; i < iters; i++) {
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

void HybridFluidSimulator3D::applyCollider() const {
  ug->parallelForEach([&](int i, int j, int k) {
    Vec3d pos{ug->indexToCoord(i, j, k)};
    if (colliderSdf->eval(pos) < 0.0) {
      // calc the normal, and project out the x component
      Vec3d normal{normalize(colliderSdf->grad(pos))};
      ug->at(i, j, k) -= ug->at(i, j, k) * normal.x;
    }
  });
  vg->parallelForEach([&](int i, int j, int k) {
    Vec3d pos = vg->indexToCoord(i, j, k);
    if (colliderSdf->eval(pos) < 0.0) {
      Vec3d normal{normalize(colliderSdf->grad(pos))};
      vg->at(i, j, k) -= vg->at(i, j, k) * normal.y;
    }
  });
  wg->parallelForEach([&](int i, int j, int k) {
    Vec3d pos{wg->indexToCoord(i, j, k)};
    if (colliderSdf->eval(pos) < 0.0) {
      Vec3d normal{normalize(colliderSdf->grad(pos))};
      wg->at(i, j, k) -= wg->at(i, j, k) * normal.z;
    }
  });
}

void HybridFluidSimulator3D::substep(Real dt) {
  clear();
  std::cout << "Solving advection... ";
  advector->advect(std::span(positions()), *ug, *vg, *wg, *colliderSdf, dt);
  std::cout << "Done." << std::endl;
  std::cout << "Reconstructing surface... ";
  fluidSurfaceReconstructor->reconstruct(
      m_particles.positions, 1.2 * ug->gridSpacing().x / std::sqrt(2.0),
      *fluidSurface, *sdfValid);
  std::cout << "Done." << std::endl;
  std::cout << "Smoothing surface... ";
  extrapolateFluidSdf(10);
  smoothFluidSurface(5);
  std::cout << "Done." << std::endl;
  std::cout << "Solving P2G... ";
  advector->solveP2G(std::span(positions()), *ug, *vg, *wg,
                     *colliderSdf, *uValid, *vValid, *wValid, dt);
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

Real HybridFluidSimulator3D::CFL() const {
  Real h{ug->gridSpacing().x};
  Real cfl{h / 1e-6};
  ug->forEach([&cfl, h, this](int x, int y, int z) {
    assert(notNan(ug->at(x, y ,z)));
    if (ug->at(x, y, z) != 0.0)
      cfl = std::max(cfl, h / abs(ug->at(x, y, z)));
  });
  vg->forEach([&cfl, h, this](int x, int y, int z) {
    assert(notNan(vg->at(x, y, z)));
    if (vg->at(x, y, z) != 0.0)
      cfl = std::max(cfl, h / abs(vg->at(x, y, z)));
  });
  wg->forEach([&cfl, h, this](int x, int y, int z) {
    assert(notNan(wg->at(x, y, z)));
    if (wg->at(x, y, z) != 0.0)
      cfl = std::max(cfl, h / abs(wg->at(x, y, z)));
  });
  return cfl;
}

void HybridFluidSimulator3D::step(core::Frame& frame) {
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