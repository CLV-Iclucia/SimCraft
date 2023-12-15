//
// Created by creeper on 13/1/33.
//
#include <Core/animation.h>
#include <Core/rand-gen.h>
#include <FluidSim/common/fluid-simulator.h>
#include <FluidSim/common/advect-solver.h>
#include <FluidSim/common/project-solver.h>
#include <FluidSim/common/util.h>
#include <cassert>

namespace fluid {
void HybridFluidSimulator3D::applyForce(Real dt) const {
  vg->forInside([this, dt](const Vec3i& idx) {
    vg->at(idx) -= 9.8 * dt;
  });
}

void HybridFluidSimulator3D::markCell() {
  markers.fill(Marker::Air, Marker::Solid);
  markers.forEach([&](int i, int j, int k) {
    if (fluidSurface->grid(i, j, k) < 0.0)
      markers(i, j, k) = Marker::Fluid;
    else if (colliderSdf.grid(i, j, k) < 0.0)
      markers(i, j, k) = Marker::Solid;
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
    fluidSurfaceBuf->grid.parallelForEach([&](int i, int j, int k) {
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
      if (fluidSurface->grid(i, j, k) > fluidSurfaceBuf->grid(i, j, k))
        fluidSurfaceBuf->grid(i, j, k) = fluidSurface->grid(i, j, k);
    });
    std::swap(fluidSurface, fluidSurfaceBuf);
  }
}

void HybridFluidSimulator3D::applyCollider() const {
  ug->parallelForEach([&](int i, int j, int k) {
    Vec3d pos{ug->indexToCoord(Vec3i(i, j, k))};
    if (colliderSdf.eval(pos) < 0.0) {
      // calc the normal, and project out the x component
      Vec3d normal{normalize(colliderSdf.grad(pos))};
      ug->at(i, j, k) -= ug->at(i, j, k) * normal.x;
    }
  });
  vg->parallelForEach([&](int i, int j, int k) {
    Vec3d pos = vg->indexToCoord(Vec3i(i, j, k));
    if (colliderSdf.eval(pos) < 0.0) {
      Vec3d normal{normalize(colliderSdf.grad(pos))};
      vg->at(i, j, k) -= vg->at(i, j, k) * normal.y;
    }
  });
  wg->parallelForEach([&](int i, int j, int k) {
    Vec3d pos{wg->indexToCoord(Vec3i(i, j, k))};
    if (colliderSdf.eval(pos) < 0.0) {
      Vec3d normal{normalize(colliderSdf.grad(pos))};
      wg->at(i, j, k) -= wg->at(i, j, k) * normal.z;
    }
  });
}

void HybridFluidSimulator3D::substep(Real dt) {
  clear();
  advector->advect(std::span(positions()), ug.get(), vg.get(), wg.get(),
                   colliderSdf, dt);
  fluidSurfaceReconstructor->reconstruct(
      nParticles, m_particles.positions.data(),
      1.2 * ug->gridSpacing().x / std::sqrt(2.0), fluidSurface.get());
  smoothFluidSurface(5);
  advector->solveP2G(std::span(positions()), ug.get(), vg.get(), wg.get(),
                     colliderSdf, uValid.get(), vValid.get(), wValid.get(), dt);
  extrapolate(ug, ubuf, uValid, uValidBuf, 10);
  extrapolate(vg, vbuf, vValid, vValidBuf, 10);
  extrapolate(wg, wbuf, wValid, wValidBuf, 10);
  applyForce(dt);
  projector->buildSystem(markers, ug.get(), vg.get(), wg.get(),
                         fluidSurface.get(), colliderSdf, density, dt);
  Real residual{projector->solvePressure(pg)};
  if (residual > 1e-4)
    std::cerr << "Warning: projection residual is " << residual << std::endl;
  projector->project(markers, ug.get(), vg.get(), wg.get(), fluidSurface.get(),
                     colliderSdf, density, dt);
  applyCollider();
  advector->solveG2P(std::span(positions()), ug.get(), vg.get(), wg.get(),
                     colliderSdf, dt);
}

Real HybridFluidSimulator3D::CFL() const {
  Real cfl{0.0};
  Real h{ug->gridSpacing().x};
  Real min_speed{1e-6};
  ug->forEach([&cfl, h, this](int x, int y, int z) {
    if (ug->at(x, y, z) != 0.0)
      cfl = std::max(cfl, h / abs(ug->at(x, y, z)));
  });
  vg->forEach([&cfl, h, this](int x, int y, int z) {
    if (vg->at(x, y, z) != 0.0)
      cfl = std::max(cfl, h / abs(vg->at(x, y, z)));
  });
  wg->forEach([&cfl, h, this](int x, int y, int z) {
    if (wg->at(x, y, z) != 0.0)
      cfl = std::max(cfl, h / abs(wg->at(x, y, z)));
  });
  return std::min(cfl, h / min_speed);
}

void HybridFluidSimulator3D::step(core::Frame& frame) {
  Real t = 0;
  while (t < frame.dt) {
    Real cfl = CFL();
    Real dt = std::min(cfl, frame.dt - t);
    substep(dt);
    t += dt;
  }
  frame.onAdvance();
}
} // namespace fluid