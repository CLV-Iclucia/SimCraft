//
// Created by creeper on 24-3-22.
//

#ifndef SIM_CRAFT_CUDA_UTILS_H
#define SIM_CRAFT_CUDA_UTILS_H
#include <Core/cuda-utils.h>
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <cub/cub.cuh>
namespace fluid::cuda {
CUDA_FORCEINLINE CUDA_CALLABLE double3 getCellCentre(int x, int y, int z) {
  return make_double3(x + 0.5, y + 0.5, z + 0.5);
}

CUDA_FORCEINLINE CUDA_CALLABLE double3 getFaceCentre(
    int x, int y, int z, int axis) {
  return make_double3(x + (axis == 0), y + (axis == 1), z + (axis == 2));
}

CUDA_FORCEINLINE CUDA_CALLABLE int3 getCellIndex(double3 p, double h) {
  return make_int3(static_cast<int>(p.x / h), static_cast<int>(p.y / h),
                   static_cast<int>(p.z / h));
}

CUDA_FORCEINLINE CUDA_DEVICE double3 grad(CudaTextureAccessor<float> field,
                                          const float3 &pos, int3 resolution, float h) {

}

    CUDA_GLOBAL void kernelSaxpy(CudaSurfaceAccessor<float> x,
                                 CudaSurfaceAccessor<float> y,
                                 float alpha,
                                 CudaSurfaceAccessor<uint8_t> active,
                                 int3 resolution) {
      get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
      if (!active.read(i, j, k)) return;
      float val = x.read(i, j, k) + alpha * y.read(i, j, k);
      x.write(val, i, j, k);
    }

CUDA_GLOBAL void kernelScaleAndAdd(CudaSurfaceAccessor<float> x,
                                   CudaSurfaceAccessor<float> y,
                                   float alpha,
                                   CudaSurfaceAccessor<uint8_t> active,
                                   int3 resolution);

inline void scaleAndAdd(CudaSurface<float> &x,
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
                                         DeviceArrayAccessor<float> result,
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

CUDA_GLOBAL void kernelNorm(CudaSurfaceAccessor<float> surface,
                            CudaSurfaceAccessor<uint8_t> active,
                            DeviceArrayAccessor<float> result,
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

void saxpy(CudaSurface<float> &x,
           const CudaSurface<float> &y,
           float alpha,
           const CudaSurface<uint8_t> &active,
           int3 resolution) {
  cudaSafeCheck(kernelSaxpy<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      x.surfaceAccessor(), y.surfaceAccessor(), alpha, active.surfaceAccessor(), resolution));
}

}
#endif //SIM_CRAFT_CUDA_UTILS_H