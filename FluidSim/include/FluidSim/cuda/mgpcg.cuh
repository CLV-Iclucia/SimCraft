//
// Created by creeper on 4/25/24.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_CUDA_MGPCG_CUH_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_CUDA_MGPCG_CUH_
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <FluidSim/cuda/vec-op.cuh>

namespace fluid::cuda {
constexpr int kVcycleLevel = 5;
constexpr int kSmoothingIters = 20;

__global__ void PrecomputeDownSampleKernel(CudaSurfaceAccessor<uint8_t> u,
                                           CudaSurfaceAccessor<uint8_t> uc, uint n);
__global__ void RestrictKernel(CudaSurfaceAccessor<float> u,
                               CudaSurfaceAccessor<float> uc, uint n);
__global__ void ProlongateKernel(CudaSurfaceAccessor<float> uc,
                                 CudaSurfaceAccessor<float> u, uint n);
__global__ void DampedJacobiKernel(CudaSurfaceAccessor<float> u,
                                   CudaSurfaceAccessor<float> uc,
                                   CudaSurfaceAccessor<float> f, uint n, float alpha);
void prepareWeights();
}
#endif //SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_CUDA_MGPCG_CUH_
