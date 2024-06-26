//
// Created by creeper on 1/15/24.
//

#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <FluidSim/cuda/vec-op.cuh>

inline void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace fluid::cuda {
using core::uint;
__device__ __forceinline__ bool withinSource(float x, float y, float z,
                                             int n) {
  float centre = static_cast<float>(n) * 0.5f;
  bool insideBall = (x - centre) * (x - centre) + (z - centre) * (z - centre) +
      (y - centre) * (y - centre) < centre * centre;
  return (x - centre) * (x - centre) + (z - centre) * (z - centre) + (y) * (y) < 256.f && insideBall;
}

__device__ __forceinline__ float3 makeCellCenter(int x, int y, int z) {
  return make_float3(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
                     static_cast<float>(z) + 0.5f);
}
__global__ void AdvectKernel(CudaTextureAccessor<float4> texVel,
                             CudaSurfaceAccessor<float4> surf_loc,
                             CudaSurfaceAccessor<uint8_t> active,
                             uint n,
                             float dt);

template<typename T>
__global__ void ResampleKernel(CudaSurfaceAccessor<float4> surf_loc,
                               CudaTextureAccessor<T> tex,
                               CudaSurfaceAccessor<T> tex_nxt,
                               CudaSurfaceAccessor<uint8_t> active,
                               T src_val, uint n) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  if (x >= n || y >= n || z >= n || !active.read(x, y, z))
    return;
  float4 loc = surf_loc.read(x, y, z);
  auto val = withinSource(loc.x, loc.y, loc.z, n)
             ? src_val
             : tex.sample(loc.x, loc.y, loc.z);
  tex_nxt.write(val, x, y, z);
}

__global__ void CoolingKernel(CudaSurfaceAccessor<float> surf_T,
                              CudaSurfaceAccessor<float> surf_T_nxt,
                              CudaSurfaceAccessor<float> surf_rho,
                              CudaSurfaceAccessor<float> surf_rho_nxt,
                              CudaSurfaceAccessor<uint8_t> active,
                              uint n, float ambientTemperature, float dt);
__global__ void SmoothingKernel(CudaSurfaceAccessor<float> surf_rho,
                                CudaSurfaceAccessor<float> surf_rho_nxt,
                                CudaSurfaceAccessor<uint8_t> active,
                                uint n);
__global__ void DivergenceKernel(CudaSurfaceAccessor<float4> surf_vel,
                                 CudaSurfaceAccessor<float> surf_div,
                                 CudaSurfaceAccessor<uint8_t> active,
                                 uint n);
__global__ void CurlKernel(CudaSurfaceAccessor<float4> surf_vel,
                           CudaSurfaceAccessor<float4> surf_curl,
                           CudaSurfaceAccessor<uint8_t> active,
                           uint n);
__global__ void JacobiKernel(CudaSurfaceAccessor<float> surf_div,
                             CudaSurfaceAccessor<float> surf_p,
                             CudaSurfaceAccessor<float> surf_p_nxt,
                             uint n);
__global__ void SubgradientKernel(CudaSurfaceAccessor<float> surf_p,
                                  CudaSurfaceAccessor<float4> surf_vel,
                                  CudaSurfaceAccessor<uint8_t> active,
                                  float4 src_vel, uint n);
__global__ void AccumulateForceKernel(CudaSurfaceAccessor<float4> surf_vel,
                                      CudaSurfaceAccessor<float4> surf_vort,
                                      CudaSurfaceAccessor<float> surf_T,
                                      CudaSurfaceAccessor<float> surf_rho,
                                      CudaSurfaceAccessor<uint8_t> active,
                                      uint n, float alpha, float beta, float
                                      epsilon, float ambientTemperature,
                                      float dt);
__global__ void ApplyForceKernel(CudaSurfaceAccessor<float4> surf_vel,
                                 CudaSurfaceAccessor<float4> surf_vel_nxt,
                                 CudaSurfaceAccessor<float4> surf_force,
                                 CudaSurfaceAccessor<uint8_t> active,
                                 uint n, float dt);
__global__ void kernelSetupFluidRegion(CudaSurfaceAccessor<uint8_t> surf,
                                       DeviceArrayAccessor<uint8_t> region,
                                       uint n);

__global__ void ExtrapolateKernel(CudaSurfaceAccessor<float4> surf,
                                  CudaSurfaceAccessor<float4> surfBuf,
                                  CudaSurfaceAccessor<uint8_t> valid,
                                  CudaSurfaceAccessor<uint8_t> validBuf,
                                  uint n);
__global__ void ExtrapolateKernel(CudaSurfaceAccessor<float> surf,
                                  CudaSurfaceAccessor<float> surfBuf,
                                  CudaSurfaceAccessor<uint8_t> valid,
                                  CudaSurfaceAccessor<uint8_t> validBuf,
                                  uint n);

template<typename T>
__global__ void ClearInactiveKernel(CudaSurfaceAccessor<float4> vel,
                                    CudaSurfaceAccessor<uint8_t> active,
                                    T zero,
                                    uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z))
    vel.write(zero, x, y, z);
}
}

#endif //KERNELS_CUH