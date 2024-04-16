#include <Core/animation.h>
#include <Core/rand-gen.h>
#include <FluidSim/cuda/fluid-simulator.h>
#include <FluidSim/cuda/advect-solver.h>
#include <FluidSim/cuda/project-solver.h>
#include <FluidSim/cuda/utils.h>
#include <FluidSim/cuda/vec-op.cuh>
#include <cassert>

namespace fluid::cuda {
void FluidSimulator::applyForce(Real dt) const {
  vg->forEach([this, dt](int i, int j, int k) {
    if (vValid.read(i, j, k) && j > 0 && j < vg->height() - 1)
      vg.read(i, j, k) -= 9.8 * dt;
  });
}

void FluidSimulator::clear() const {
  uValid->zero();
  vValid->zero();
  wValid->zero();
  uValidBuf->zero();
  vValidBuf->zero();
  wValidBuf->zero();
  v->zero();
  u->zero();
  w->zero();
  uBuf->zero();
  vBuf->zero();
  wBuf->zero();
  uw->zero();
  vw->zero();
  ww->zero();
  p->zero();
}

static CUDA_GLOBAL void kernelSmooth(CudaSurfaceAccessor<Real> grid,
                                     CudaSurfaceAccessor<Real> buf,
                                     CudaSurfaceAccessor<uint8_t> valid,
                                     CudaSurfaceAccessor<uint8_t> validBuf,
                                     int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (valid.read(i, j, k))
    return;
  Real sum = 0;
  int count = 0;
  if (i > 0) {
    sum += grid.read(i - 1, j, k);
    count++;
  }
  if (i < resolution.x - 1) {
    sum += grid.read(i + 1, j, k);
    count++;
  }
  if (j > 0) {
    sum += grid.read(i, j - 1, k);
    count++;
  }
  if (j < resolution.y - 1) {
    sum += grid.read(i, j + 1, k);
    count++;
  }
  if (k > 0) {
    sum += grid.read(i, j, k - 1);
    count++;
  }
  if (k < resolution.z - 1) {
    sum += grid.read(i, j, k + 1);
    count++;
  }
  Real val = fmin(grid.read(i, j, k), sum / count);
  buf.write(val, i, j, k);
}

void FluidSimulator::smoothFluidSurface(int iters) {
  for (int iter = 0; iter < iters; iter++) {
    cudaSafeCheck(kernelSmooth<<<>>>());
    std::swap(fluidSurface, fluidSurfaceBuf);
  }
}
static CUDA_GLOBAL void kernelApplyCollider(CudaSurfaceAccessor<Real> ug,
                                            CudaSurfaceAccessor<Real> vg,
                                            CudaSurfaceAccessor<Real> wg,
                                            CudaTextureAccessor<Real> colliderSdf,
                                            int3 resolution,
                                            Real h) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (i == 0 || i == resolution.x) {
    ug.write(0.0, i, j, k);
    return;
  }
  auto pos =
      make_float3(static_cast<float>(i * h), static_cast<float>((j + 0.5) * h), static_cast<float>((k + 0.5) * h));
  double3 normal{normalize(grad(colliderSdf, pos, resolution, h))};
  if (colliderSdf.sample(pos) < 0.0) {
    // calc the normal, and project out the x component
    Real val = ug.read(i, j, k);
    ug.write(val - val * normal.x, i, j, k);
  }
  if (j == 0 || j == resolution.y) {
    vg.write(0.0, i, j, k);
    return;
  }
  pos = make_float3(static_cast<float>((i + 0.5) * h), static_cast<float>(j * h), static_cast<float>((k + 0.5) * h));
  if (colliderSdf.sample(pos) < 0.0) {
    Real val = vg.read(i, j, k);
    vg.write(val - val * normal.y, i, j, k);
  }
  if (k == 0 || k == resolution.z) {
    wg.write(0.0, i, j, k);
    return;
  }
  pos = make_float3(static_cast<float>((i + 0.5) * h), static_cast<float>((j + 0.5) * h), static_cast<float>(k * h));
  if (colliderSdf.sample(pos) < 0.0) {
    Real val = wg.read(i, j, k);
    wg.write(val - val * normal.z, i, j, k);
  }
}
void FluidSimulator::applyCollider() const {

}

void FluidSimulator::applyDirichletBoundary() const {

}

// here, resolution is the resolution of the grid
// resolution is not necceasarily the same as the resolution of the fluid surface
static void extrapolate(std::unique_ptr<CudaSurface<Real>> &grid,
                        std::unique_ptr<CudaSurface<Real>> &buf,
                        std::unique_ptr<CudaSurface<uint8_t>> &valid,
                        std::unique_ptr<CudaSurface<uint8_t>> &validBuf,
                        int3 resolution,
                        int iters) {
  for (int iter = 0; iter < iters; iter++) {
    cudaSafeCheck(kernelSmooth<<<>>>(grid->accessor(), buf->accessor(),
                                     valid->accessor(), validBuf->accessor(),
                                     grid->resolution()));
    std::swap(grid, buf);
    std::swap(valid, validBuf);
  }
}

void FluidSimulator::extrapolateFluidSdf(int iters) const {
  for (int iter = 0; iter < iters; iter++) {
    cudaSafeCheck(kernelSmooth<<<>>>());
    std::swap(sdfValid, sdfValidBuf);
  }
}

void FluidSimulator::substep(Real dt) {
  clear();
  std::cout << "Solving advection... ";
  advectionSolver->advect(*advectionSolver->particles, *u, *v, *w, resolution, h, dt);
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
                     *colliderSdf, uw, vw, ww, *uValid, *vValid, *wValid, dt);
  applyDirichletBoundary();
  std::cout << "Done." << std::endl;
  std::cout << "Extrapolating velocities... ";
  extrapolate(u, uBuf, uValid, uValidBuf, 10);
  extrapolate(v, vBuf, vValid, vValidBuf, 10);
  extrapolate(w, wBuf, wValid, wValidBuf, 10);
  std::cout << "Done." << std::endl;
  applyForce(dt);
  std::cout << "Building linear system... ";
  projector->buildSystem(*u, *v, *w, *fluidSurface, *colliderSdf, dt);
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

}

void FluidSimulator::step(core::Frame &frame) {
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
}