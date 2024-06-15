//
// Created by creeper on 4/26/24.
//
#include <FluidSim/cuda/utils.h>

namespace fluid::cuda {
CUDA_GLOBAL void kernelSaxpy(CudaSurfaceAccessor<float> x,
                             CudaSurfaceAccessor<float> y,
                             double alpha,
                             CudaSurfaceAccessor<uint8_t> active,
                             int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (!active.read(i, j, k)) return;
  double val = x.read(i, j, k) + alpha * y.read(i, j, k);
  x.write(val, i, j, k);
}

CUDA_GLOBAL void kernelScaleAndAdd(CudaSurfaceAccessor<float> x,
                                   CudaSurfaceAccessor<float> y,
                                   double alpha,
                                   CudaSurfaceAccessor<uint8_t> active,
                                   int3 resolution) {
  get_and_restrict_tid_3d(i, j, k, resolution.x, resolution.y, resolution.z);
  if (!active.read(i, j, k)) return;
  double val = x.read(i, j, k) + alpha * y.read(i, j, k);
  x.write(val, i, j, k);
}
CUDA_GLOBAL void kernelDotProduct(CudaSurfaceAccessor<float> surfaceA,
                                  CudaSurfaceAccessor<float> surfaceB,
                                  CudaSurfaceAccessor<uint8_t> active,
                                  DeviceArrayAccessor<double> result,
                                  int3 dimensions) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  bool valid = x < dimensions.x && y < dimensions.y && z < dimensions.z;
  auto block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  auto valueA = static_cast<double>(surfaceA.read<cudaBoundaryModeZero>(x, y, z));
  auto valueB = static_cast<double>(surfaceB.read<cudaBoundaryModeZero>(x, y, z));
  auto local_result = valueA * valueB * active.read<cudaBoundaryModeZero>(x, y, z) * valid;
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

CUDA_GLOBAL void kernelLinfNorm(CudaSurfaceAccessor<float> surface,
                                CudaSurfaceAccessor<uint8_t> active,
                                DeviceArrayAccessor<double> result,
                                int3 dimensions) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  bool valid = x < dimensions.x && y < dimensions.y && z < dimensions.z;
  uint32_t block_idx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  double local_result =
      fabs(surface.read<cudaBoundaryModeZero>(x, y, z)) * active.read<cudaBoundaryModeZero>(x, y, z) * valid;
  using BlockReduce = cub::BlockReduce<double, kThreadBlockSize3D,
                                       cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                       kThreadBlockSize3D,
                                       kThreadBlockSize3D>;
  CUDA_SHARED BlockReduce::TempStorage temp_storage;
  auto block_result = BlockReduce(temp_storage).Reduce(local_result, cub::Max());
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    result[block_idx] = block_result;
}
}