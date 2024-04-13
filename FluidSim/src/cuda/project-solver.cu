//
// Created by creeper on 24-3-26.
//
#include <FluidSim/cuda/project-solver.h>
#include <FluidSim/cuda/gpu-arrays.h>
#include <FluidSim/cuda/vec-op.cuh>
#include <cub/cub.cuh>
#include <format>

namespace fluid::cuda {

static CUDA_DEVICE CUDA_FORCEINLINE void cycle(Real &a0, Real &a1, Real &a2, Real &a3) {
  Real tmp = a0;
  a0 = a3;
  a3 = a2;
  a2 = a1;
  a1 = tmp;
}

static CUDA_DEVICE Real fractionInside(Real lu, Real ru, Real rd, Real ld) {
  int n_negs = (lu <= 0.0) + (ru <= 0.0) + (rd <= 0.0) + (ld <= 0.0);
  if (n_negs == 0)
    return 0.0;
  if (n_negs == 4)
    return 1.0;
  while (lu > 0.0)
    cycle(lu, ru, rd, ld);
  // now lu must be negative
  if (n_negs == 1) {
    Real fracu = ru / (ru - lu);
    Real fracl = ld / (ld - lu);
    return fracu * fracl * 0.5;
  }
  if (n_negs == 2) {
    if (ru <= 0.0) {
      Real fracl = ld / (ld - lu);
      Real fracr = rd / (rd - ru);
      return 0.5 * (fracl + fracr);
    }
    if (ld <= 0.0) {
      Real fracu = ru / (ru - lu);
      Real fracd = rd / (rd - ld);
      return 0.5 * (fracu + fracd);
    }
    if (rd <= 0.0) {
      Real fracu = ru / (ru - lu);
      Real fracd = ld / (ld - rd);
      Real fracl = ld / (ld - lu);
      Real fracr = ru / (ru - rd);
      if (lu + ru + rd + ld <= 0.0)
        return 1.0 - 0.5 * ((1.0 - fracu) * (1.0 - fracl) + (1.0 - fracd) * (
            1.0 - fracr));
      return 0.5 * (fracu * fracl + fracd * fracr);
    }
  }
  if (n_negs == 3) {
    while (lu <= 0.0)
      cycle(lu, ru, rd, ld);
    Real fracu = ru / (ru - lu);
    Real fracl = ld / (ld - lu);
    return 1.0 - 0.5 * fracu * fracl;
  }
}

static CUDA_GLOBAL void kernrelApplyCompressedMatrix(
    ConstAccessor<CompressedSystem> A,
    CudaSurfaceAccessor<Real> x,
    CudaSurfaceAccessor<uint8_t> active,
    CudaSurfaceAccessor<Real> b,
    int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (!active.read(i, j, k)) return;
  Real t = A.diag.read(i, j, k) * x.read(i, j, k);
  t += active.read<cudaBoundaryModeZero>(i - 1, j, k) *
      A.neighbour[Left].read(i, j, k) *
      x.read<cudaBoundaryModeZero>(i - 1, j, k);
  t += active.read<cudaBoundaryModeZero>(i, j - 1, k) *
      A.neighbour[Down].read(i, j, k) *
      x.read<cudaBoundaryModeZero>(i, j - 1, k);
  t += active.read<cudaBoundaryModeZero>(i - 1, j, k) *
      A.neighbour[Back].read(i, j, k) *
      x.read<cudaBoundaryModeZero>(i, j, k - 1);
  t += active.read<cudaBoundaryModeZero>(i + 1, j, k) *
      A.neighbour[Right].read(i, j, k) *
      x.read<cudaBoundaryModeZero>(i + 1, j, k);
  t += active.read<cudaBoundaryModeZero>(i, j + 1, k) *
      A.neighbour[Up].read(i, j, k) *
      x.read<cudaBoundaryModeZero>(i, j + 1, k);
  t += active.read<cudaBoundaryModeZero>(i, j, k + 1) *
      A.neighbour[Front].read(i, j, k) *
      x.read<cudaBoundaryModeZero>(i, j, k + 1);
  b.write(t, i, j, k);
}

static void applyCompressedMatrix(const CompressedSystem &sys,
                                  const CudaSurface<Real> &x,
                                  const CudaSurface<uint8_t> &active,
                                  CudaSurface<Real> &b,
                                  int3 resolution) {
  cudaSafeCheck(kernrelApplyCompressedMatrix<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      sys.constAccessor(), x.surfAccessor(), active.surfAccessor(), b.surfAccessor(), resolution));
}

static CUDA_GLOBAL void kernelSaxpy(CudaSurfaceAccessor<Real> x,
                                    CudaSurfaceAccessor<Real> y,
                                    Real alpha,
                                    CudaSurfaceAccessor<uint8_t> active,
                                    int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (!active.read(i, j, k)) return;
  Real val = x.read(i, j, k) + alpha * y.read(i, j, k);
  x.write(val, i, j, k);
}

static CUDA_GLOBAL void kernelScaleAndAdd(CudaSurfaceAccessor<Real> x,
                                          CudaSurfaceAccessor<Real> y,
                                          Real alpha,
                                          CudaSurfaceAccessor<uint8_t> active,
                                          int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (!active.read(i, j, k)) return;
  x.write(x.read(i, j, k) + alpha * y.read(i, j, k), i, j, k);
}

static void scaleAndAdd(CudaSurface<Real> &x,
                        const CudaSurface<Real> &y,
                        Real alpha,
                        const CudaSurface<uint8_t> &active,
                        int3 resolution) {
  cudaSafeCheck(kernelScaleAndAdd<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      x.surfAccessor(), y.surfAccessor(), alpha, active.surfAccessor(), resolution));
}

static CUDA_GLOBAL void kernelDotProduct(CudaSurfaceAccessor<Real> surfaceA,
                                         CudaSurfaceAccessor<Real> surfaceB,
                                         CudaSurfaceAccessor<uint8_t> active,
                                         Accessor<DeviceArray<Real>> result,
                                         int3 dimensions) {
  get_and_restrict_tid_3d(x, y, z, dimensions.x, dimensions.y, dimensions.z);
  auto block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  Real valueA = surfaceA.read(x, y, z);
  Real valueB = surfaceB.read(x, y, z);
  Real local_result = valueA * valueB * active.read(x, y, z);
  using BlockReduce = cub::BlockReduce<Real, kThreadBlockSize3D,
                                       cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                       kThreadBlockSize3D,
                                       kThreadBlockSize3D>;
  CUDA_SHARED
  BlockReduce::TempStorage temp_storage;
  Real block_result = BlockReduce(temp_storage).Sum(local_result,
                                                    blockDim.x * blockDim.y * blockDim.z);
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    result[block_idx] = block_result;
}

static CUDA_GLOBAL void kernelL1Norm(CudaSurfaceAccessor<Real> surface,
                                     CudaSurfaceAccessor<uint8_t> active,
                                     Accessor<DeviceArray<Real>> result,
                                     int3 dimensions) {
  get_and_restrict_tid_3d(x, y, z, dimensions.x, dimensions.y, dimensions.z);
  uint32_t block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  Real local_result = fabs(surface.read(x, y, z)) * active.read(x, y, z);
  using BlockReduce = cub::BlockReduce<Real, kThreadBlockSize3D,
                                       cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                       kThreadBlockSize3D,
                                       kThreadBlockSize3D>;
  CUDA_SHARED
  BlockReduce::TempStorage temp_storage;
  auto block_result = BlockReduce(temp_storage).Sum(local_result,
                                                    blockDim.x * blockDim.y * blockDim.z);
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    result[block_idx] = block_result;
}

static void saxpy(CudaSurface<Real> &x,
                  const CudaSurface<Real> &y,
                  Real alpha,
                  const CudaSurface<uint8_t> &active,
                  int3 resolution) {
  cudaSafeCheck(kernelSaxpy<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      x.surfAccessor(), y.surfAccessor(), alpha, active.surfAccessor(), resolution));
}

Real CgSolver::dotProduct(const CudaSurface<Real> &surfaceA,
                       const CudaSurface<Real> &surfaceB,
                       const CudaSurface<uint8_t> &active,
                       int3 resolution) const {
  int block_num = (resolution.x + kThreadBlockSize3D - 1) / kThreadBlockSize3D
      * (resolution.y + kThreadBlockSize3D - 1) / kThreadBlockSize3D
      * (resolution.z + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  cudaSafeCheck(kernelDotProduct<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      surfaceA.surfAccessor(), surfaceB.surfAccessor(), active.surfAccessor(), device_reduce_buffer->accessor(), resolution));
  device_reduce_buffer->copyTo(host_reduce_buffer);
  Real sum = 0.0;
  for (int i = 0; i < block_num; i++)
    sum += host_reduce_buffer[i];
  return sum;
}

static CUDA_GLOBAL void kernelComputeAreaWeights(
    CudaSurfaceAccessor<Real> uWeights,
    CudaSurfaceAccessor<Real> vWeights,
    CudaSurfaceAccessor<Real> wWeights,
    CudaSurfaceAccessor<Real> fluidSdf,
    CudaTextureAccessor<Real> colliderSdf,
    int3 resolution, Real h) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (i == 0) {
    uWeights.write(0.0, resolution.x, j, k);
    return;
  } else {
    if (fluidSdf.read<cudaBoundaryModeZero>(i - 1, j, k) <= 0.0
        || fluidSdf.read(i, j, k) < 0.0) {
      float3 p = make_float3(static_cast<float>(i) - 0.5f, static_cast<float>(j), static_cast<float>(k));
      Real bu = colliderSdf.sample(p + make_float3(0.0f, 0.5f, -0.5f));
      Real bd = colliderSdf.sample(p + make_float3(0.0f, -0.5f, -0.5f));
      Real fd = colliderSdf.sample(p + make_float3(0.0f, -0.5f, 0.5f));
      Real fu = colliderSdf.sample(p + make_float3(0.0f, 0.5f, 0.5f));
      Real frac = fractionInside(bu, bd, fd, fu);
      uWeights.write(1.0 - frac, i, j, k);
    }
  }
  if (j == 0) {
    vWeights.write(0.0, i, resolution.y, k);
    return;
  } else {
    if (fluidSdf.read(i, j - 1, k) < 0.0 || fluidSdf.read(i, j, k) < 0.0) {
      float3 p = make_float3(i, j - 0.5, k);
      Real lb = colliderSdf.sample(p + make_float3(-0.5f, 0.0f, -0.5f));
      Real rb = colliderSdf.sample(p + make_float3(0.5f, 0.0f, -0.5f));
      Real rf = colliderSdf.sample(p + make_float3(0.5f, 0.0f, 0.5f));
      Real lf = colliderSdf.sample(p + make_float3(-0.5f, 0.0f, 0.5f));
      Real frac = fractionInside(lb, rb, rf, lf);
      assert(frac >= 0.0 && frac <= 1.0);
      vWeights.write(1.0 - frac, i, j, k);
      assert(notNan(vWeights.read(i, j, k)));
    }
  }
  if (k == 0) {
    wWeights.write(0.0, i, j, resolution.z);
  } else {
    if (fluidSdf.read<cudaBoundaryModeZero>(i, j, k - 1) < 0.0 || fluidSdf.read(i, j, k) < 0.0) {
      float3 p = make_float3(i, j, k - 0.5);
      Real ld = colliderSdf.sample(p + make_float3(-0.5f, -0.5f, 0.f));
      Real lu = colliderSdf.sample(p + make_float3(-0.5f, 0.5f, 0.f));
      Real ru = colliderSdf.sample(p + make_float3(0.5f, 0.5f, 0.f));
      Real rd = colliderSdf.sample(p + make_float3(0.5f, -0.5f, 0.f));
      Real frac = fractionInside(ld, lu, ru, rd);
      assert(frac >= 0.0 && frac <= 1.0);
      wWeights.write(1.0 - frac, i, j, k);
      assert(notNan(wWeights.read(i, j, k)));
    }
  }

}

static CUDA_GLOBAL void kernelComputeMatrix(
    Accessor<CompressedSystem> A,
    CudaSurfaceAccessor<Real> uWeights,
    CudaSurfaceAccessor<Real> vWeights,
    CudaSurfaceAccessor<Real> wWeights,
    CudaSurfaceAccessor<Real> rhs,
    CudaSurfaceAccessor<uint8_t> active,
    CudaSurfaceAccessor<Real> ug,
    CudaSurfaceAccessor<Real> vg,
    CudaSurfaceAccessor<Real> wg,
    CudaSurfaceAccessor<Real> fluidSdf,
    Real h,
    Real dt,
    int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (uWeights.read<cudaBoundaryModeZero>(i, j, k) == 0.0 &&
      uWeights.read<cudaBoundaryModeZero>(i + 1, j, k) == 0.0 &&
      vWeights.read<cudaBoundaryModeZero>(i, j, k) == 0.0 &&
      vWeights.read<cudaBoundaryModeZero>(i, j + 1, k) == 0.0 &&
      wWeights.read<cudaBoundaryModeZero>(i, j, k) == 0.0 &&
      wWeights.read<cudaBoundaryModeZero>(i, j, k + 1) == 0.0)
    return;
  if (fluidSdf.read(i, j, k) > 0.0)
    return;
  active.write(1, i, j, k);
  Real signed_dist = fluidSdf.read(i, j, k);
  Real factor = dt / h;

  // left
  Real a_diag = 0;
  Real b = 0;
  if (i > 0) {
    auto sdf_l = fluidSdf.read<cudaBoundaryModeZero>(i - 1, j, k);
    if (sdf_l > 0.0) {
      Real theta = fmin(sdf_l / (sdf_l - signed_dist), 0.99);
      a_diag += uWeights.read(i, j, k) * factor / (1.0 - theta);
    } else {
      a_diag += uWeights.read(i, j, k) * factor;
      A.neighbour[Left].write(-uWeights.read(i, j, k) * factor, i, j, k);
    }
    b += uWeights.read(i, j, k) * ug.read(i, j, k);
  }

  // right
  if (i < resolution.x - 1) {
    auto sdf_r = fluidSdf.read(i + 1, j, k);
    if (sdf_r > 0.0) {
      Real theta = fmax(
          signed_dist / (signed_dist - sdf_r), 0.01);
      a_diag += uWeights.read(i + 1, j, k) * factor / theta;
    } else {
      if (i < resolution.x - 1) {
        a_diag += uWeights.read(i + 1, j, k) * factor;
        A.neighbour[Right].write(-uWeights.read(i + 1, j, k) * factor, i, j, k);
      }
    }
    b -= uWeights.read(i + 1, j, k) * ug.read(i + 1, j, k);
  }

  // down
  if (j > 0) {
    auto sdf_d = fluidSdf.read<cudaBoundaryModeZero>(i, j - 1, k);
    if (sdf_d > 0.0) {
      Real theta = fmin(sdf_d / (sdf_d - signed_dist), 0.99);
      a_diag += vWeights.read(i, j, k) * factor / (1.0 - theta);
    } else {
      a_diag += vWeights.read(i, j, k) * factor;
      A.neighbour[Down].write(-vWeights.read(i, j, k) * factor, i, j, k);
    }
    b += vWeights.read(i, j, k) * vg.read(i, j, k);
  }
  // up
  if (j < resolution.y - 1) {
    auto sdf_u = fluidSdf.read<cudaBoundaryModeZero>(i, j + 1, k);
    if (sdf_u > 0.0) {
      Real theta = fmax(signed_dist / (signed_dist - sdf_u), 0.01);
      a_diag += vWeights.read(i, j + 1, k) * factor / theta;
    } else {
      a_diag += vWeights.read(i, j + 1, k) * factor;
      A.neighbour[Up].write(-vWeights.read(i, j + 1, k) * factor, i, j, k);
    }
    b -= vWeights.read(i, j + 1, k) * vg.read(i, j + 1, k);
  }

  // back
  if (k > 0) {
    auto sdf_b = fluidSdf.read<cudaBoundaryModeZero>(i, j, k - 1);
    if (sdf_b > 0.0) {
      Real theta = fmin(sdf_b / (sdf_b - signed_dist), 0.99);
      a_diag += wWeights.read(i, j, k) * factor / (1.0 - theta);
    } else {
      a_diag += wWeights.read(i, j, k) * factor;
      A.neighbour[Back].write(-wWeights.read(i, j, k) * factor, i, j, k);
    }
    b += wWeights.read(i, j, k) * wg.read(i, j, k);
  }

  // front
  if (k < resolution.z - 1) {
    auto sdf_f = fluidSdf.read<cudaBoundaryModeZero>(i, j, k + 1);
    if (sdf_f > 0.0) {
      Real theta = fmax(signed_dist / (signed_dist - sdf_f), 0.01);
      a_diag += wWeights.read(i, j, k + 1) * factor / theta;
    } else {
      a_diag += wWeights.read(i, j, k + 1) * factor;
      A.neighbour[Front].write(-wWeights.read(i, j, k + 1) * factor, i, j, k);
    }
    b -= wWeights.read(i, j, k + 1) * wg.read(i, j, k + 1);
  }
  assert(a_diag > 0.0);
  A.diag.write(a_diag, i, j, k);
  rhs.write(b, i, j, k);
}

Real CgSolver::L1Norm(const CudaSurface<Real> &surface,
                      const CudaSurface<uint8_t> &active,
                      int3 resolution) const {
  int block_num = (resolution.x + kThreadBlockSize3D - 1) / kThreadBlockSize3D
      * (resolution.y + kThreadBlockSize3D - 1) / kThreadBlockSize3D
      * (resolution.z + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  cudaSafeCheck(kernelL1Norm<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      surface.surfAccessor(), active.surfAccessor(), device_reduce_buffer->accessor(), resolution));
  device_reduce_buffer->copyTo(host_reduce_buffer);
  Real sum = 0.0;
  for (int i = 0; i < block_num; i++)
    sum += host_reduce_buffer[i];
  return sum;
}

static CUDA_GLOBAL void kernelProjectVelocity(
    CudaSurfaceAccessor<Real> ug,
    CudaSurfaceAccessor<Real> vg,
    CudaSurfaceAccessor<Real> wg,
    CudaSurfaceAccessor<Real> pg,
    CudaSurfaceAccessor<Real> uWeights,
    CudaSurfaceAccessor<Real> vWeights,
    CudaSurfaceAccessor<Real> wWeights,
    CudaSurfaceAccessor<Real> fluidSdf,
    CudaSurfaceAccessor<Real> colliderSdf,
    int3 resolution,
    Real h,
    Real dt) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (uWeights.read(i, j, k) > 0.0) {
    if (i == 0) {
      ug.write(0.0, 0, j, k);
      ug.write(0.0, resolution.x, j, k);
      return;
    }
    Real sd_left = fluidSdf.read<cudaBoundaryModeZero>(i - 1, j, k);
    Real sd_right = fluidSdf.read(i, j, k);
    Real u = ug.read(i, j, k);
    if (sd_left < 0.0 && sd_right < 0.0) {
      ug.write(u - (pg.read(i, j, k) - pg.read(i - 1, j, k)) * dt / h, i, j, k);
      return;
    }
    if (sd_left < 0.0) {
      Real theta = fmax(sd_left / (sd_left - sd_right), 0.01);
      ug.write(u + pg.read(i - 1, j, k) * dt / h / theta, i, j, k);
    } else {
      Real theta = fmin(sd_left / (sd_left - sd_right), 0.99);
      ug.write(u - pg.read(i, j, k) * dt / h / (1.0 - theta), i, j, k);
    }
  }
  if (vWeights.read(i, j, k) > 0.0) {
    if (j == 0) {
      vg.write(0.0, i, 0, k);
      vg.write(0.0, i, resolution.y, k);
      return;
    }
    Real sd_down = fluidSdf.read<cudaBoundaryModeZero>(i, j - 1, k);
    Real sd_up = fluidSdf.read(i, j, k);
    assert(notNan(pg(i, j, k)));
    Real v = vg.read(i, j, k);
    if (sd_down < 0.0 && sd_up < 0.0) {
      vg.write(v - (pg.read(i, j, k) - pg.read(i, j - 1, k)) * dt / h, i, j, k);
      return;
    }
    if (sd_down < 0.0) {
      Real theta = fmax(sd_down / (sd_down - sd_up), 0.01);
      vg.write(v + pg.read(i, j - 1, k) * dt / h / theta, i, j, k);
    } else {
      Real theta = fmin(sd_down / (sd_down - sd_up), 0.99);
      vg.write(v - pg.read(i, j, k) * dt / h / (1.0 - theta), i, j, k);
    }
  }
  if (wWeights.read(i, j, k) > 0.0) {
    if (k == 0) {
      wg.write(0.0, i, j, 0);
      wg.write(0.0, i, j, resolution.z);
      return;
    }
    assert(notNan(pg(i, j, k)));
    Real sd_back = fluidSdf.read(i, j, k - 1);
    Real sd_front = fluidSdf.read(i, j, k);
    Real w = wg.read(i, j, k);
    if (sd_back < 0.0 && sd_front < 0.0) {
      wg.write(w - (pg.read(i, j, k) - pg.read(i, j, k - 1)) * dt / h, i, j, k);
      return;
    }
    if (sd_back < 0.0) {
      Real theta = fmax(sd_back / (sd_back - sd_front), 0.01);
      wg.write(w + pg.read(i, j, k - 1) * dt / h / theta, i, j, k);
    } else {
      Real theta = fmin(sd_back / (sd_back - sd_front), 0.99);
      wg.write(w - pg.read(i, j, k) * dt / h / (1.0 - theta), i, j, k);
    }
  }
}

void FvmSolver::buildSystem(const CudaSurface<Real> &ug,
                            const CudaSurface<Real> &vg,
                            const CudaSurface<Real> &wg,
                            const CudaSurface<Real> &fluidSdf,
                            const CudaSurface<Real> &colliderSdf,
                            int3 resolution,
                            Real h,
                            Real dt) {
  sys->diag->zero();
  sys->neighbour[Left]->zero();
  sys->neighbour[Right]->zero();
  sys->neighbour[Up]->zero();
  sys->neighbour[Down]->zero();
  sys->neighbour[Front]->zero();
  sys->neighbour[Back]->zero();
  sys->rhs->zero();
  uWeights->zero();
  vWeights->zero();
  wWeights->zero();
  active->zero();
}
Real CgSolver::solve(const CompressedSystem &sys,
                     const CudaSurface<uint8_t> &active,
                     CudaSurface<Real> &pg,
                     int3 resolution) {
  pg.zero();
  r->copyFrom(*sys.rhs);
  Real residual = L1Norm(*r, active, resolution);
  if (residual < tolerance) {
    std::cout << "naturally converged" << std::endl;
    return residual;
  }
  if (preconditioner)
    preconditioner->precond(sys, *r, active, *z);
  else
    z->copyFrom(*r);
  s->copyFrom(*z);
  Real sigma = dotProduct(*z, *r, active, resolution);
  int iter = 1;
  for (; iter < max_iterations; iter++) {
    applyCompressedMatrix(sys, *s, active, *z, resolution);
    Real sdotz = dotProduct(*s, *z, active, resolution);
    assert(sdotz != 0);
    Real alpha = sigma / sdotz;
    saxpy(pg, *s, alpha, active, resolution);
    saxpy(*r, *z, -alpha, active, resolution);
    residual = L1Norm(*r, active, resolution);
    if (residual < tolerance) break;
    if (preconditioner)
      preconditioner->precond(sys, *r, active, *z);
    Real sigma_new = dotProduct(*z, *r, active, resolution);
    assert(sigma != 0);
    Real beta = sigma_new / sigma;
    scaleAndAdd(*s, *z, beta, active, resolution);
    sigma = sigma_new;
  }
  std::cout << std::format("PCG iterations: {}", iter) << std::endl;
  return residual;
}

}