#ifndef SIMCRAFT_FLUIDSIM_GPU_SMOKE_SIMULATOR_H_
#define SIMCRAFT_FLUIDSIM_GPU_SMOKE_SIMULATOR_H_

#include <Core/core.h>
#include <Core/properties.h>
#include <Core/animation.h>
#include <FluidSim/cuda/kernels.cuh>
#include <FluidSim/cuda/mgpcg.cuh>
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <vector_types.h>
#include <memory>
#include <array>
#include <numeric>

namespace fluid::cuda {
struct GpuSmokeSimulator final : core::Animation, core::NonCopyable {
  uint32_t n;
  std::unique_ptr<CudaSurface<float4>> loc;
  std::unique_ptr<CudaTexture<float4>> vel;
  std::unique_ptr<CudaTexture<float4>> velBuf;
  std::unique_ptr<CudaTexture<float4>> force;
  std::unique_ptr<CudaTexture<float>> rho;
  std::unique_ptr<CudaTexture<float>> rhoBuf;
  std::unique_ptr<CudaTexture<float>> T;
  std::unique_ptr<CudaTexture<float>> TBuf;
  std::array<std::unique_ptr<CudaSurface<uint8_t>>, kVcycleLevel + 1> fluidRegion{};
  std::unique_ptr<CudaSurface<float4>> vort;
  std::unique_ptr<CudaSurface<float4>> normal;
  std::unique_ptr<CudaSurface<float>> p{};
  std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel + 1> r;
  std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel + 1> pcg_p;
  std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel + 1> pcg_pbuf;
  std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel + 1> pcg_z;
  std::array<std::unique_ptr<CudaSurface<float>>, kVcycleLevel + 1> pcg_zbuf;
  std::vector<double> host_reduce_buffer;
  std::unique_ptr<DeviceArray<double>> device_reduce_buffer{};
  std::vector<uint> sizes;
  int active_cnt = 0;
  float alpha = 1.f;
  float beta = 1.f;
  float epsilon = 1.5f;
  float ambientTemperature = 0.f;
  explicit GpuSmokeSimulator(unsigned int _n)
      : n(_n),
        loc(std::make_unique<CudaSurface<float4>>(uint3{n, n, n})),
        vel(std::make_unique<CudaTexture<float4>>(uint3{n, n, n})),
        velBuf(std::make_unique<CudaTexture<float4>>(uint3{n, n, n})),
        force(std::make_unique<CudaTexture<float4>>(uint3{n, n, n})),
        rho(std::make_unique<CudaTexture<float>>(uint3{n, n, n})),
        rhoBuf(std::make_unique<CudaTexture<float>>(uint3{n, n, n})),
        T(std::make_unique<CudaTexture<float>>(uint3{n, n, n})),
        TBuf(std::make_unique<CudaTexture<float>>(uint3{n, n, n})),
        vort(std::make_unique<CudaSurface<float4>>(uint3{n, n, n})),
        normal(std::make_unique<CudaSurface<float4>>(uint3{n, n, n})),
        p(std::make_unique<CudaSurface<float>>(uint3{n, n, n})),
        device_reduce_buffer(std::make_unique<DeviceArray<double>>(n * n * n)),
        host_reduce_buffer(n * n * n) {
    prepareWeights();
    for (int i = 0; i < kVcycleLevel + 1; i++) {
      r[i] = std::make_unique<CudaSurface<float>>(uint3{n >> i, n >> i, n >> i});
      pcg_p[i] = std::make_unique<CudaSurface<float>>(uint3{n >> i, n >> i, n >> i});
      pcg_pbuf[i] = std::make_unique<CudaSurface<float>>(uint3{n >> i, n >> i, n >> i});
      pcg_z[i] = std::make_unique<CudaSurface<float>>(uint3{n >> i, n >> i, n >> i});
      pcg_zbuf[i] = std::make_unique<CudaSurface<float>>(uint3{n >> i, n >> i, n >> i});
    }
    for (int i = 0; i < kVcycleLevel + 1; i++)
      fluidRegion[i] = std::make_unique<CudaSurface<uint8_t>>(uint3{n >> i, n >> i, n >> i});
    int nthreads_dim = 8;
    int nblocks = (n + nthreads_dim - 1) / 8;
    FillKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        vel->surfaceAccessor(), make_float4(0.f, 0.f, 0.f, 0.f), n);
    FillKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        rho->surfaceAccessor(), 0.f, n);
    FillKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        T->surfaceAccessor(), 0.f, n);
  }

  void setActiveRegion(const std::vector<uint8_t> &region_marker) {
    std::unique_ptr<DeviceArray<uint8_t>> region_marker_dev =
        std::make_unique<DeviceArray<uint8_t>>(region_marker);
    active_cnt = std::accumulate(region_marker.begin(), region_marker.end(), 0);
    kernelSetupFluidRegion<<<dim3(n / 8, n / 8, n / 8),
    dim3(8, 8, 8)>>>(fluidRegion[0]->surfaceAccessor(), region_marker_dev->accessor(), n);
    for (int i = 1; i < kVcycleLevel + 1; i++) {
      int nthreads_dim = 8;
      int nblocks = (n >> i) + nthreads_dim - 1 / nthreads_dim;
      PrecomputeDownSampleKernel<<<dim3(nblocks, nblocks, nblocks),
      dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
          fluidRegion[i - 1]->surfaceAccessor(), fluidRegion[i]->surfaceAccessor(), n >> i);
    }
  }
  void advection(float dt) {
    uint nthreads_dim = 8;
    uint nblocks = (n + nthreads_dim - 1) / 8;
    AdvectKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        vel->texAccessor(), loc->surfaceAccessor(), n, dt);
    checkCUDAErrorWithLine("AdvectKernel failed!");
    ResampleKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        loc->surfaceAccessor(), rho->texAccessor(), rhoBuf->surfaceAccessor(),
        1.f, n);
    checkCUDAErrorWithLine("Resample rho failed!");
    ResampleKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        loc->surfaceAccessor(), vel->texAccessor(), velBuf->surfaceAccessor(),
        make_float4(0.f, 0.0f, 0.f, 0.f), n);
    checkCUDAErrorWithLine("Resample vel failed!");
    ResampleKernel<float><<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        loc->surfaceAccessor(), T->texAccessor(), TBuf->surfaceAccessor(),
        200.f, n);
    checkCUDAErrorWithLine("Resample T failed!");
    std::swap(vel, velBuf);
    std::swap(rho, rhoBuf);
    std::swap(T, TBuf);
  }
  void cool(float dt) {
    uint nthreads_dim = 8;
    uint nblocks = (n + nthreads_dim - 1) / 8;
    CoolingKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        T->surfaceAccessor(), TBuf->surfaceAccessor(), rho->surfaceAccessor(),
        rhoBuf->surfaceAccessor(), n, ambientTemperature, dt);
    std::swap(T, TBuf);
    std::swap(rho, rhoBuf);
  }
  void applyForce(float dt) {
    uint nthreads_dim = 8;
    uint nblocks = (n + nthreads_dim - 1) / 8;
    CurlKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        vel->surfaceAccessor(), vort->surfaceAccessor(), n);
    checkCUDAErrorWithLine("Compute vorticity failed!");
    AccumulateForceKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        vel->surfaceAccessor(), vort->surfaceAccessor(), T->surfaceAccessor(),
        rho->surfaceAccessor(), n, alpha, beta, epsilon, ambientTemperature,
        dt);
    checkCUDAErrorWithLine("Accumulate force failed!");
    ApplyForceKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        vel->surfaceAccessor(), velBuf->surfaceAccessor(), force->surfaceAccessor(),
        n, dt);
    checkCUDAErrorWithLine("Apply force failed!");
    std::swap(vel, velBuf);
  }
  void projection(float dt) {
    uint nthreads_dim = 8;
    uint nblocks = (n + nthreads_dim - 1) / 8;
    DivergenceKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        vel->surfaceAccessor(), r[0]->surfaceAccessor(), n);
    checkCUDAErrorWithLine("Compute divergence failed!");
    mgpcg(fluidRegion,
          r,
          pcg_p,
          pcg_pbuf,
          pcg_z,
          pcg_zbuf,
          *p,
          *device_reduce_buffer,
          host_reduce_buffer,
          active_cnt,
          n,
          kMaxIters,
          kTolerance);
    SubgradientKernel<<<dim3(nblocks, nblocks, nblocks),
    dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
        p->surfaceAccessor(), vel->surfaceAccessor(),
        make_float4(0.f, 0.0f, 0.f, 0.f), n);
    checkCUDAErrorWithLine("Subgradient failed!");
  }
  void smooth(float dt) {
    uint nthreads_dim = 8;
    uint nblocks = (n + nthreads_dim - 1) / 8;
    for (int i = 0; i < 4; i++) {
      SmoothingKernel<<<dim3(nblocks, nblocks, nblocks),
      dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
          rho->surfaceAccessor(), rhoBuf->surfaceAccessor(), n);
      checkCUDAErrorWithLine("Smooth failed!");
      std::swap(rho, rhoBuf);
    }
  }
  void substep(float dt) {
    advection(dt);
    cool(dt);
//    smooth(dt);
    applyForce(dt);
    projection(dt);
  }
  void step(core::Frame &frame) override {
    // std::cout << "Substep " << i << std::endl;
    substep(frame.dt);
    frame.onAdvance();
  }
};
}

#endif