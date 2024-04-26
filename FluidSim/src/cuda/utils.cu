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
}