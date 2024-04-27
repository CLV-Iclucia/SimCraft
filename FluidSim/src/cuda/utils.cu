//
// Created by creeper on 4/26/24.
//
#include <FluidSim/cuda/utils.h>

namespace fluid::cuda {
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
                                   int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (!active.read(i, j, k)) return;
  x.write(x.read(i, j, k) + alpha * y.read(i, j, k), i, j, k);
}
CUDA_GLOBAL void kernelDotProduct(CudaSurfaceAccessor<float> surfaceA,
                                  CudaSurfaceAccessor<float> surfaceB,
                                  CudaSurfaceAccessor<uint8_t> active,
                                  DeviceArrayAccessor<double> result,
                                  int3 dimensions) {
  get_and_restrict_tid_3d(x, y, z, dimensions.x, dimensions.y, dimensions.z);
  auto block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  auto valueA = static_cast<double>(surfaceA.read(x, y, z));
  auto valueB = static_cast<double>(surfaceB.read(x, y, z));
  auto local_result = valueA * valueB * active.read(x, y, z);
  using BlockReduce = cub::BlockReduce<double, kThreadBlockSize3D,
                                       cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                       kThreadBlockSize3D,
                                       kThreadBlockSize3D>;
  CUDA_SHARED
  BlockReduce::TempStorage temp_storage;
  double block_result = BlockReduce(temp_storage).Sum(local_result,
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
}