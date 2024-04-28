//
// Created by creeper on 4/25/24.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_CUDA_MGPCG_CUH_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_CUDA_MGPCG_CUH_
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <FluidSim/cuda/vec-op.cuh>

namespace fluid::cuda {
constexpr int kVcycleLevel = 3;
constexpr int kSmoothingIters = 15;
constexpr int kBottomSolveIters = 20;
constexpr double kDampedJacobiOmega = 2.0 / 3.0;
constexpr double kTolerance = 1e-5;
constexpr double kMaxIters = 10;
__device__ __forceinline__ float localLaplacian(CudaSurfaceAccessor<float> u, CudaSurfaceAccessor<uint8_t> active,
                                                int x, int y, int z) {
  uint8_t axp = active.read<cudaBoundaryModeZero>(x - 1, y, z);
  uint8_t axn = active.read<cudaBoundaryModeZero>(x + 1, y, z);
  uint8_t ayp = active.read<cudaBoundaryModeZero>(x, y - 1, z);
  uint8_t ayn = active.read<cudaBoundaryModeZero>(x, y + 1, z);
  uint8_t azp = active.read<cudaBoundaryModeZero>(x, y, z - 1);
  uint8_t azn = active.read<cudaBoundaryModeZero>(x, y, z + 1);
  auto cnt = static_cast<double>(axp + axn + ayp + ayn + azp + azn);
  float pxp = static_cast<float>(axp) * u.read<cudaBoundaryModeClamp>(x - 1, y, z);
  float pxn = static_cast<float>(axn) * u.read<cudaBoundaryModeClamp>(x + 1, y, z);
  float pyp = static_cast<float>(ayp) * u.read<cudaBoundaryModeClamp>(x, y - 1, z);
  float pyn = static_cast<float>(ayn) * u.read<cudaBoundaryModeClamp>(x, y + 1, z);
  float pzp = static_cast<float>(azp) * u.read<cudaBoundaryModeClamp>(x, y, z - 1);
  float pzn = static_cast<float>(azn) * u.read<cudaBoundaryModeClamp>(x, y, z + 1);
  return u.read(x, y, z) * cnt - (pxp + pxn + pyp + pyn + pzp + pzn);
}
__global__ void PrecomputeDownSampleKernel(CudaSurfaceAccessor<uint8_t> u,
                                           CudaSurfaceAccessor<uint8_t> uc, uint n);
__global__ void RestrictKernel(CudaSurfaceAccessor<float> u,
                               CudaSurfaceAccessor<float> uc, uint n);
__global__ void ProlongateKernel(CudaSurfaceAccessor<float> uc,
                                 CudaSurfaceAccessor<float> u, uint n);
__global__ void DampedJacobiKernel(CudaSurfaceAccessor<float> u,
                                   CudaSurfaceAccessor<uint8_t> active,
                                   CudaSurfaceAccessor<float> f, uint n);
void vCycle(std::array<std::unique_ptr<CudaSurface<uint8_t >>, kVcycleLevel> &active,
            std::array<std::unique_ptr<CudaSurface<float >>, kVcycleLevel> &u,
            std::array<std::unique_ptr<CudaSurface<float >>, kVcycleLevel> &uBuf,
            std::array<std::unique_ptr<CudaSurface<float >>, kVcycleLevel> &b,
            int n);
void mgpcg(std::array<std::unique_ptr<CudaSurface<uint8_t>>, kVcycleLevel> &active,
           std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel> &r,
           std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel> &p,
           std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel> &pBuf,
           std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel> &z,
           std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel> &zBuf,
           CudaSurface<float> &x,
           DeviceArray<double> &device_buffer,
           std::vector<double> &host_buffer,
           int active_cnt, int n, int maxIters, float tolerance);
void prepareWeights();
}
#endif //SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_CUDA_MGPCG_CUH_
