//
// Created by creeper on 24-3-26.
//
#include <FluidSim/cuda/project-solver.h>
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <FluidSim/cuda/vec-op.cuh>
#include <cub/cub.cuh>
#include <format>

namespace fluid::cuda {

static CUDA_DEVICE CUDA_FORCEINLINE void cycle(float &a0, float &a1, float &a2, float &a3) {
  float tmp = a0;
  a0 = a3;
  a3 = a2;
  a2 = a1;
  a1 = tmp;
}

static CUDA_DEVICE float fractionInside(float lu, float ru, float rd, float ld) {
  int n_negs = (lu <= 0.0) + (ru <= 0.0) + (rd <= 0.0) + (ld <= 0.0);
  if (n_negs == 0)
    return 0.0;
  if (n_negs == 4)
    return 1.0;
  while (lu > 0.0)
    cycle(lu, ru, rd, ld);
  // now lu must be negative
  if (n_negs == 1) {
    float fracu = ru / (ru - lu);
    float fracl = ld / (ld - lu);
    return fracu * fracl * 0.5;
  }
  if (n_negs == 2) {
    if (ru <= 0.0) {
      float fracl = ld / (ld - lu);
      float fracr = rd / (rd - ru);
      return 0.5 * (fracl + fracr);
    }
    if (ld <= 0.0) {
      float fracu = ru / (ru - lu);
      float fracd = rd / (rd - ld);
      return 0.5 * (fracu + fracd);
    }
    if (rd <= 0.0) {
      float fracu = ru / (ru - lu);
      float fracd = ld / (ld - rd);
      float fracl = ld / (ld - lu);
      float fracr = ru / (ru - rd);
      if (lu + ru + rd + ld <= 0.0)
        return 1.0 - 0.5 * ((1.0 - fracu) * (1.0 - fracl) + (1.0 - fracd) * (
            1.0 - fracr));
      return 0.5 * (fracu * fracl + fracd * fracr);
    }
  }
  if (n_negs == 3) {
    while (lu <= 0.0)
      cycle(lu, ru, rd, ld);
    float fracu = ru / (ru - lu);
    float fracl = ld / (ld - lu);
    return 1.0 - 0.5 * fracu * fracl;
  }
}

static CUDA_GLOBAL void kernrelApplyCompressedMatrix(
    ConstAccessor<CompressedSystem> A,
    CudaSurfaceAccessor<float> x,
    CudaSurfaceAccessor<uint8_t> active,
    CudaSurfaceAccessor<float> b,
    int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (!active.read(i, j, k)) return;
  float t = A.diag.read(i, j, k) * x.read(i, j, k);
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
                                  const CudaSurface<float> &x,
                                  const CudaSurface<uint8_t> &active,
                                  CudaSurface<float> &b,
                                  int3 resolution) {
  cudaSafeCheck(kernrelApplyCompressedMatrix<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      sys.constAccessor(), x.surfaceAccessor(), active.surfaceAccessor(), b.surfaceAccessor(), resolution));
}

static CUDA_GLOBAL void kernelSaxpy(CudaSurfaceAccessor<float> x,
                                    CudaSurfaceAccessor<float> y,
                                    float alpha,
                                    CudaSurfaceAccessor<uint8_t> active,
                                    int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (!active.read(i, j, k)) return;
  float val = x.read(i, j, k) + alpha * y.read(i, j, k);
  x.write(val, i, j, k);
}

static CUDA_GLOBAL void kernelScaleAndAdd(CudaSurfaceAccessor<float> x,
                                          CudaSurfaceAccessor<float> y,
                                          float alpha,
                                          CudaSurfaceAccessor<uint8_t> active,
                                          int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (!active.read(i, j, k)) return;
  x.write(x.read(i, j, k) + alpha * y.read(i, j, k), i, j, k);
}

static void scaleAndAdd(CudaSurface<float> &x,
                        const CudaSurface<float> &y,
                        float alpha,
                        const CudaSurface<uint8_t> &active,
                        int3 resolution) {
  cudaSafeCheck(kernelScaleAndAdd<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      x.surfaceAccessor(), y.surfaceAccessor(), alpha, active.surfaceAccessor(), resolution));
}

static CUDA_GLOBAL void kernelDotProduct(CudaSurfaceAccessor<float> surfaceA,
                                         CudaSurfaceAccessor<float> surfaceB,
                                         CudaSurfaceAccessor<uint8_t> active,
                                         Accessor<DeviceArray<float>> result,
                                         int3 dimensions) {
  get_and_restrict_tid_3d(x, y, z, dimensions.x, dimensions.y, dimensions.z);
  auto block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  float valueA = surfaceA.read(x, y, z);
  float valueB = surfaceB.read(x, y, z);
  float local_result = valueA * valueB * active.read(x, y, z);
  using BlockReduce = cub::BlockReduce<float, kThreadBlockSize3D,
                                       cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                       kThreadBlockSize3D,
                                       kThreadBlockSize3D>;
  CUDA_SHARED
  BlockReduce::TempStorage temp_storage;
  float block_result = BlockReduce(temp_storage).Sum(local_result,
                                                    blockDim.x * blockDim.y * blockDim.z);
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    result[block_idx] = block_result;
}

static CUDA_GLOBAL void kernelL1Norm(CudaSurfaceAccessor<float> surface,
                                     CudaSurfaceAccessor<uint8_t> active,
                                     Accessor<DeviceArray<float>> result,
                                     int3 dimensions) {
  get_and_restrict_tid_3d(x, y, z, dimensions.x, dimensions.y, dimensions.z);
  uint32_t block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  float local_result = fabs(surface.read(x, y, z)) * active.read(x, y, z);
  using BlockReduce = cub::BlockReduce<float, kThreadBlockSize3D,
                                       cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                       kThreadBlockSize3D,
                                       kThreadBlockSize3D>;
  CUDA_SHARED BlockReduce::TempStorage temp_storage;
  auto block_result = BlockReduce(temp_storage).Sum(local_result,
                                                    blockDim.x * blockDim.y * blockDim.z);
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    result[block_idx] = block_result;
}

static void saxpy(CudaSurface<float> &x,
                  const CudaSurface<float> &y,
                  float alpha,
                  const CudaSurface<uint8_t> &active,
                  int3 resolution) {
  cudaSafeCheck(kernelSaxpy<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      x.surfaceAccessor(), y.surfaceAccessor(), alpha, active.surfaceAccessor(), resolution));
}

float CgSolver::dotProduct(const CudaSurface<float> &surfaceA,
                       const CudaSurface<float> &surfaceB,
                       const CudaSurface<uint8_t> &active,
                       int3 resolution) const {
  int block_num = (resolution.x + kThreadBlockSize3D - 1) / kThreadBlockSize3D
      * (resolution.y + kThreadBlockSize3D - 1) / kThreadBlockSize3D
      * (resolution.z + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  cudaSafeCheck(kernelDotProduct<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      surfaceA.surfaceAccessor(), surfaceB.surfaceAccessor(),
      active.surfaceAccessor(), device_reduce_buffer->accessor(), resolution));
  device_reduce_buffer->copyTo(host_reduce_buffer);
  float sum = 0.0;
  for (int i = 0; i < block_num; i++)
    sum += host_reduce_buffer[i];
  return sum;
}

static CUDA_GLOBAL void kernelComputeAreaWeights(
    CudaSurfaceAccessor<float> uWeights,
    CudaSurfaceAccessor<float> vWeights,
    CudaSurfaceAccessor<float> wWeights,
    CudaSurfaceAccessor<float> fluidSdf,
    CudaTextureAccessor<float> colliderSdf,
    int3 resolution, float h) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (i == 0) {
    uWeights.write(0.0, resolution.x, j, k);
    return;
  } else {
    if (fluidSdf.read<cudaBoundaryModeZero>(i - 1, j, k) <= 0.0
        || fluidSdf.read(i, j, k) < 0.0) {
      float3 p = make_float3(static_cast<float>(i) - 0.5f, static_cast<float>(j), static_cast<float>(k));
      float bu = colliderSdf.sample(p + make_float3(0.0f, 0.5f, -0.5f));
      float bd = colliderSdf.sample(p + make_float3(0.0f, -0.5f, -0.5f));
      float fd = colliderSdf.sample(p + make_float3(0.0f, -0.5f, 0.5f));
      float fu = colliderSdf.sample(p + make_float3(0.0f, 0.5f, 0.5f));
      float frac = fractionInside(bu, bd, fd, fu);
      uWeights.write(1.0 - frac, i, j, k);
    }
  }
  if (j == 0) {
    vWeights.write(0.0, i, resolution.y, k);
    return;
  } else {
    if (fluidSdf.read(i, j - 1, k) < 0.0 || fluidSdf.read(i, j, k) < 0.0) {
      float3 p = make_float3(i, j - 0.5, k);
      float lb = colliderSdf.sample(p + make_float3(-0.5f, 0.0f, -0.5f));
      float rb = colliderSdf.sample(p + make_float3(0.5f, 0.0f, -0.5f));
      float rf = colliderSdf.sample(p + make_float3(0.5f, 0.0f, 0.5f));
      float lf = colliderSdf.sample(p + make_float3(-0.5f, 0.0f, 0.5f));
      float frac = fractionInside(lb, rb, rf, lf);
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
      float ld = colliderSdf.sample(p + make_float3(-0.5f, -0.5f, 0.f));
      float lu = colliderSdf.sample(p + make_float3(-0.5f, 0.5f, 0.f));
      float ru = colliderSdf.sample(p + make_float3(0.5f, 0.5f, 0.f));
      float rd = colliderSdf.sample(p + make_float3(0.5f, -0.5f, 0.f));
      float frac = fractionInside(ld, lu, ru, rd);
      assert(frac >= 0.0 && frac <= 1.0);
      wWeights.write(1.0 - frac, i, j, k);
      assert(notNan(wWeights.read(i, j, k)));
    }
  }

}

static CUDA_GLOBAL void kernelComputeMatrix(
    Accessor<CompressedSystem> A,
    CudaSurfaceAccessor<float> uWeights,
    CudaSurfaceAccessor<float> vWeights,
    CudaSurfaceAccessor<float> wWeights,
    CudaSurfaceAccessor<float> rhs,
    CudaSurfaceAccessor<uint8_t> active,
    CudaSurfaceAccessor<float> ug,
    CudaSurfaceAccessor<float> vg,
    CudaSurfaceAccessor<float> wg,
    CudaSurfaceAccessor<float> fluidSdf,
    float h,
    float dt,
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
  float signed_dist = fluidSdf.read(i, j, k);
  float factor = dt / h;

  // left
  float a_diag = 0;
  float b = 0;
  if (i > 0) {
    auto sdf_l = fluidSdf.read<cudaBoundaryModeZero>(i - 1, j, k);
    if (sdf_l > 0.0) {
      float theta = min(sdf_l / (sdf_l - signed_dist), 0.99);
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
      float theta = max(
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
      float theta = min(sdf_d / (sdf_d - signed_dist), 0.99);
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
      float theta = max(signed_dist / (signed_dist - sdf_u), 0.01);
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
      float theta = min(sdf_b / (sdf_b - signed_dist), 0.99);
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
      float theta = max(signed_dist / (signed_dist - sdf_f), 0.01);
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

float CgSolver::L1Norm(const CudaSurface<float> &surface,
                      const CudaSurface<uint8_t> &active,
                      int3 resolution) const {
  int block_num = (resolution.x + kThreadBlockSize3D - 1) / kThreadBlockSize3D
      * (resolution.y + kThreadBlockSize3D - 1) / kThreadBlockSize3D
      * (resolution.z + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  cudaSafeCheck(kernelL1Norm<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      surface.surfaceAccessor(), active.surfaceAccessor(), device_reduce_buffer->accessor(), resolution));
  device_reduce_buffer->copyTo(host_reduce_buffer);
  float sum = 0.0;
  for (int i = 0; i < block_num; i++)
    sum += host_reduce_buffer[i];
  return sum;
}

static CUDA_GLOBAL void kernelProjectVelocity(
    CudaSurfaceAccessor<float> ug,
    CudaSurfaceAccessor<float> vg,
    CudaSurfaceAccessor<float> wg,
    CudaSurfaceAccessor<float> pg,
    CudaSurfaceAccessor<float> uWeights,
    CudaSurfaceAccessor<float> vWeights,
    CudaSurfaceAccessor<float> wWeights,
    CudaSurfaceAccessor<float> fluidSdf,
    CudaSurfaceAccessor<float> colliderSdf,
    int3 resolution,
    float h,
    float dt) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (uWeights.read(i, j, k) > 0.0) {
    if (i == 0) {
      ug.write(0.0, 0, j, k);
      ug.write(0.0, resolution.x, j, k);
      return;
    }
    float sd_left = fluidSdf.read<cudaBoundaryModeZero>(i - 1, j, k);
    float sd_right = fluidSdf.read(i, j, k);
    float u = ug.read(i, j, k);
    if (sd_left < 0.0 && sd_right < 0.0) {
      ug.write(u - (pg.read(i, j, k) - pg.read(i - 1, j, k)) * dt / h, i, j, k);
      return;
    }
    if (sd_left < 0.0) {
      float theta = max(sd_left / (sd_left - sd_right), 0.01);
      ug.write(u + pg.read(i - 1, j, k) * dt / h / theta, i, j, k);
    } else {
      float theta = min(sd_left / (sd_left - sd_right), 0.99);
      ug.write(u - pg.read(i, j, k) * dt / h / (1.0 - theta), i, j, k);
    }
  }
  if (vWeights.read(i, j, k) > 0.0) {
    if (j == 0) {
      vg.write(0.0, i, 0, k);
      vg.write(0.0, i, resolution.y, k);
      return;
    }
    float sd_down = fluidSdf.read<cudaBoundaryModeZero>(i, j - 1, k);
    float sd_up = fluidSdf.read(i, j, k);
    assert(notNan(pg(i, j, k)));
    float v = vg.read(i, j, k);
    if (sd_down < 0.0 && sd_up < 0.0) {
      vg.write(v - (pg.read(i, j, k) - pg.read(i, j - 1, k)) * dt / h, i, j, k);
      return;
    }
    if (sd_down < 0.0) {
      float theta = max(sd_down / (sd_down - sd_up), 0.01);
      vg.write(v + pg.read(i, j - 1, k) * dt / h / theta, i, j, k);
    } else {
      float theta = min(sd_down / (sd_down - sd_up), 0.99);
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
    float sd_back = fluidSdf.read(i, j, k - 1);
    float sd_front = fluidSdf.read(i, j, k);
    float w = wg.read(i, j, k);
    if (sd_back < 0.0 && sd_front < 0.0) {
      wg.write(w - (pg.read(i, j, k) - pg.read(i, j, k - 1)) * dt / h, i, j, k);
      return;
    }
    if (sd_back < 0.0) {
      float theta = max(sd_back / (sd_back - sd_front), 0.01);
      wg.write(w + pg.read(i, j, k - 1) * dt / h / theta, i, j, k);
    } else {
      float theta = min(sd_back / (sd_back - sd_front), 0.99);
      wg.write(w - pg.read(i, j, k) * dt / h / (1.0 - theta), i, j, k);
    }
  }
}

void FvmSolver::buildSystem(const CudaSurface<float> &ug,
                            const CudaSurface<float> &vg,
                            const CudaSurface<float> &wg,
                            const CudaSurface<float> &fluidSdf,
                            const CudaTexture<float> &colliderSdf,
                            int3 resolution,
                            float h,
                            float dt) {
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
  cudaSafeCheck(kernelComputeAreaWeights<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      uWeights->surfaceAccessor(), vWeights->surfaceAccessor(), wWeights->surfaceAccessor(),
      fluidSdf.surfaceAccessor(), colliderSdf.texAccessor(), resolution, h));
    cudaSafeCheck(kernelComputeMatrix<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
        sys->accessor(), uWeights->surfaceAccessor(), vWeights->surfaceAccessor(), wWeights->surfaceAccessor(),
        sys->rhs->surfaceAccessor(), active->surfaceAccessor(),
        ug.surfaceAccessor(), vg.surfaceAccessor(), wg.surfaceAccessor(),
        fluidSdf.surfaceAccessor(), h, dt, resolution));
}
float CgSolver::solve(const CompressedSystem &sys,
                     const CudaSurface<uint8_t> &active,
                     CudaSurface<float> &pg,
                     int3 resolution) {
  pg.zero();
  r->copyFrom(*sys.rhs);
  float residual = L1Norm(*r, active, resolution);
  if (residual < tolerance) {
    std::cout << "naturally converged" << std::endl;
    return residual;
  }
  if (preconditioner)
    preconditioner->precond(sys, *r, active, *z);
  else
    z->copyFrom(*r);
  s->copyFrom(*z);
  float sigma = dotProduct(*z, *r, active, resolution);
  int iter = 1;
  for (; iter < max_iterations; iter++) {
    applyCompressedMatrix(sys, *s, active, *z, resolution);
    float sdotz = dotProduct(*s, *z, active, resolution);
    assert(sdotz != 0);
    float alpha = sigma / sdotz;
    saxpy(pg, *s, alpha, active, resolution);
    saxpy(*r, *z, -alpha, active, resolution);
    residual = L1Norm(*r, active, resolution);
    if (residual < tolerance) break;
    if (preconditioner)
      preconditioner->precond(sys, *r, active, *z);
    float sigma_new = dotProduct(*z, *r, active, resolution);
    assert(sigma != 0);
    float beta = sigma_new / sigma;
    scaleAndAdd(*s, *z, beta, active, resolution);
    sigma = sigma_new;
  }
  std::cout << std::format("PCG iterations: {}", iter) << std::endl;
  return residual;
}

}