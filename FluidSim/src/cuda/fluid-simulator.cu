#include <Core/animation.h>
#include <Core/rand-gen.h>
#include <FluidSim/cuda/fluid-simulator.h>
#include <FluidSim/cuda/advect-solver.h>
#include <FluidSim/cuda/project-solver.h>
#include <FluidSim/cuda/utils.h>
#include <FluidSim/cuda/vec-op.cuh>

namespace fluid::cuda {
static CUDA_GLOBAL void kernelApplyForce(CudaSurfaceAccessor<float> u,
                                         CudaSurfaceAccessor<float> v,
                                         CudaSurfaceAccessor<float> w,
                                         double3 acceleration,
                                         int3 resolution,
                                         Real dt) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  u.write(u.read(i, j, k) + acceleration.x * dt, i, j, k);
  v.write(v.read(i, j, k) + acceleration.y * dt, i, j, k);
  w.write(w.read(i, j, k) + acceleration.z * dt, i, j, k);
}
void FluidSimulator::applyForce(Real dt) const {
  cudaSafeCheck(kernelApplyForce<<<
  LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(u->surfaceAccessor(),
                                                                 v->surfaceAccessor(),
                                                                 w->surfaceAccessor(),
                                                                 make_double3(0, -9.8, 0),
                                                                 resolution,
                                                                 dt));
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

static CUDA_GLOBAL void kernelSmooth(CudaSurfaceAccessor<float> grid,
                                     CudaSurfaceAccessor<float> buf,
                                     CudaSurfaceAccessor<uint8_t> valid,
                                     CudaSurfaceAccessor<uint8_t> validBuf,
                                     int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (valid.read(i, j, k))
    return;
  float sum = 0;
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
  float val = fmin(grid.read(i, j, k), sum / count);
  buf.write(val, i, j, k);
}

void FluidSimulator::smoothFluidSurface(int iters) {
  for (int iter = 0; iter < iters; iter++) {
    cudaSafeCheck(kernelSmooth<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
        fluidSurface->surfaceAccessor(), fluidSurfaceBuf->surfaceAccessor(),
        sdfValid->surfaceAccessor(), sdfValidBuf->surfaceAccessor(), resolution));
    std::swap(fluidSurface, fluidSurfaceBuf);
  }
}
static CUDA_GLOBAL void kernelApplyCollider(CudaSurfaceAccessor<float> ug,
                                            CudaSurfaceAccessor<float> vg,
                                            CudaSurfaceAccessor<float> wg,
                                            CudaTextureAccessor<float> colliderSdf,
                                            int3 resolution,
                                            float h) {
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
    float val = ug.read(i, j, k);
    ug.write(val - val * normal.x, i, j, k);
  }
  if (j == 0 || j == resolution.y) {
    vg.write(0.0, i, j, k);
    return;
  }
  pos = make_float3(static_cast<float>((i + 0.5) * h), static_cast<float>(j * h), static_cast<float>((k + 0.5) * h));
  if (colliderSdf.sample(pos) < 0.0) {
    float val = vg.read(i, j, k);
    vg.write(val - val * normal.y, i, j, k);
  }
  if (k == 0 || k == resolution.z) {
    wg.write(0.0, i, j, k);
    return;
  }
  pos = make_float3(static_cast<float>((i + 0.5) * h), static_cast<float>((j + 0.5) * h), static_cast<float>(k * h));
  if (colliderSdf.sample(pos) < 0.0) {
    float val = wg.read(i, j, k);
    wg.write(val - val * normal.z, i, j, k);
  }
}
void FluidSimulator::applyCollider() const {

}

void FluidSimulator::applyDirichletBoundary() const {

}

// here, resolution is the resolution of the grid
// resolution is not necceasarily the same as the resolution of the fluid surface
static void extrapolate(std::unique_ptr<CudaSurface<float>> &grid,
                        std::unique_ptr<CudaSurface<float>> &buf,
                        std::unique_ptr<CudaSurface<uint8_t>> &valid,
                        std::unique_ptr<CudaSurface<uint8_t>> &validBuf,
                        int3 resolution,
                        int iters) {
//  for (int iter = 0; iter < iters; iter++) {
//    cudaSafeCheck(kernelExtrapolate<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
//        grid->surfaceAccessor(), buf->surfaceAccessor(), valid->surfaceAccessor(),
//        validBuf->surfaceAccessor(), resolution));
//    std::swap(grid, buf);
//    std::swap(valid, validBuf);
//  }
}

void FluidSimulator::extrapolateFluidSdf(int iters) {
  for (int iter = 0; iter < iters; iter++) {
    cudaSafeCheck(kernelSmooth<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
        fluidSurface->surfaceAccessor(), fluidSurfaceBuf->surfaceAccessor(),
        sdfValid->surfaceAccessor(), sdfValidBuf->surfaceAccessor(), resolution));
    std::swap(sdfValid, sdfValidBuf);
  }
}

void FluidSimulator::substep(Real dt) {
  clear();
  std::cout << "Solving advection... ";
  advectionSolver->advect(*particles, *u, *v, *w, resolution, h, dt);
  std::cout << "Done." << std::endl;
  std::cout << "Reconstructing surface... ";

  std::cout << "Done." << std::endl;
  std::cout << "Smoothing surface... ";
  extrapolateFluidSdf(10);
  smoothFluidSurface(5);
  std::cout << "Done." << std::endl;
  std::cout << "Solving P2G... ";
  advectionSolver->solveP2G(*particles, *u, *v, *w,
                            *colliderSdf, *uw, *vw, *ww, *uValid, *vValid, *wValid, resolution, h, dt);
  applyDirichletBoundary();
  std::cout << "Done." << std::endl;
  std::cout << "Extrapolating velocities... ";
//  extrapolate(u, uBuf, uValid, uValidBuf, resolution, 5);
//  extrapolate(v, vBuf, vValid, vValidBuf, resolution, 5);
//  extrapolate(w, wBuf, wValid, wValidBuf, resolution, 5);
  std::cout << "Done." << std::endl;
  applyForce(dt);
  std::cout << "Building linear system... ";
  projectionSolver->buildSystem(*u, *v, *w, *fluidSurface, *colliderSdf, resolution, h, dt);
  std::cout << "Done." << std::endl;
  std::cout << "Solving linear system... ";
  if (float residual{projectionSolver->solvePressure(*fluidSurface, *p, resolution, h, dt)};
      residual > 1e-4)
    std::cerr << "Warning: projection residual is " << residual << std::endl;
  else std::cout << "Projection residual is " << residual << std::endl;
  std::cout << "Done." << std::endl;
  std::cout << "Doing projection and applying collider... ";
  projectionSolver->project(*u, *v, *w, *fluidSurface, *colliderSdf, *p, resolution, h, dt);
  applyCollider();
  std::cout << "Done." << std::endl;
  std::cout << "Solving G2P... ";
  advectionSolver->solveG2P(*particles, *u, *v, *w, resolution, h, dt);
  std::cout << "Done" << std::endl;
}

Real FluidSimulator::CFL() const {
return 0.0;
}
}