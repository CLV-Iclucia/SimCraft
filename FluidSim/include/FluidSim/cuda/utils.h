//
// Created by creeper on 24-3-22.
//

#ifndef SIM_CRAFT_CUDA_UTILS_H
#define SIM_CRAFT_CUDA_UTILS_H
#include <Core/cuda-utils.h>
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <cub/cub.cuh>
#include <numeric>
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
                             int3 resolution);

CUDA_GLOBAL void kernelScaleAndAdd(CudaSurfaceAccessor<float> x,
                                   CudaSurfaceAccessor<float> y,
                                   float alpha,
                                   CudaSurfaceAccessor<uint8_t> active,
                                   int3 resolution);
CUDA_GLOBAL void kernelDotProduct(CudaSurfaceAccessor<float> surfaceA,
                                  CudaSurfaceAccessor<float> surfaceB,
                                  CudaSurfaceAccessor<uint8_t> active,
                                  DeviceArrayAccessor<double> result,
                                  int3 dimensions);
CUDA_GLOBAL void kernelLinfNorm(CudaSurfaceAccessor<float> surface,
                                CudaSurfaceAccessor<uint8_t> active,
                                DeviceArrayAccessor<float> result,
                                int3 dimensions);

inline float dotProduct(CudaSurface<float> &surfaceA,
                        CudaSurface<float> &surfaceB,
                        const CudaSurface<uint8_t> &active,
                        DeviceArray<double> &device_reduce_buffer,
                        std::vector<double> &host_reduce_buffer,
                        int3 dimensions) {
  int num_block_x = (dimensions.x + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_block_y = (dimensions.y + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_block_z = (dimensions.z + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_blocks = num_block_x * num_block_y * num_block_z;
  cudaSafeCheck(kernelDotProduct<<<LAUNCH_THREADS_3D(dimensions.x, dimensions.y, dimensions.z)>>>(
      surfaceA.surfaceAccessor(),
      surfaceB.surfaceAccessor(),
      active.surfaceAccessor(),
      device_reduce_buffer.accessor(),
      dimensions));
  device_reduce_buffer.copyTo(host_reduce_buffer);
  return static_cast<float>(std::accumulate(host_reduce_buffer.begin(), host_reduce_buffer.begin() + num_blocks, 0.0));
}

inline float LinfNorm(CudaSurface<float> &surface,
                      const CudaSurface<uint8_t> &active,
                      DeviceArray<double> &device_reduce_buffer,
                      std::vector<double> &host_reduce_buffer,
                      int3 dimensions) {
  int num_block_x = (dimensions.x + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_block_y = (dimensions.y + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_block_z = (dimensions.z + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_blocks = num_block_x * num_block_y * num_block_z;
  cudaSafeCheck(kernelLinfNorm<<<LAUNCH_THREADS_3D(dimensions.x, dimensions.y, dimensions.z)>>>(
      surface.surfaceAccessor(),
      active.surfaceAccessor(),
      device_reduce_buffer.accessor(),
      dimensions));
  device_reduce_buffer.copyTo(host_reduce_buffer);
  float max_val = 0.0;
  for (int i = 0; i < num_blocks; i++)
    max_val = std::max(max_val, static_cast<float>(host_reduce_buffer[i]));
  return max_val;
}

inline void scaleAndAdd(CudaSurface<float> &x,
                        const CudaSurface<float> &y,
                        float alpha,
                        const CudaSurface<uint8_t> &active,
                        int3 resolution) {
  cudaSafeCheck(kernelScaleAndAdd<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      x.surfaceAccessor(), y.surfaceAccessor(), alpha, active.surfaceAccessor(), resolution));
}

inline void saxpy(CudaSurface<float> &x,
                  const CudaSurface<float> &y,
                  float alpha,
                  const CudaSurface<uint8_t> &active,
                  int3 resolution) {
  cudaSafeCheck(kernelSaxpy<<<LAUNCH_THREADS_3D(resolution.x, resolution.y, resolution.z)>>>(
      x.surfaceAccessor(), y.surfaceAccessor(), alpha, active.surfaceAccessor(), resolution));
}

}
#endif //SIM_CRAFT_CUDA_UTILS_H