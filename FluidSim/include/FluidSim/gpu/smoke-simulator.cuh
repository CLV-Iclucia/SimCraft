#ifndef SIMCRAFT_FLUIDSIM_GPU_SMOKE_SIMULATOR_H_
#define SIMCRAFT_FLUIDSIM_GPU_SMOKE_SIMULATOR_H_

#include <device_launch_parameters.h>
#include <Core/core.h>
#include <Core/properties.h>
#include <FluidSim/gpu/gpu-arrays.h>
#include <vector_types.h>
#include <memory>

namespace fluid {
struct GpuSmokeSimulator : core::NonCopyable {
  uint n;
  std::unique_ptr<CudaSurface<float4>> loc;
  std::unique_ptr<CudaTexture<float4>> vel;
  std::unique_ptr<CudaTexture<float4>> vel_nxt;
  std::unique_ptr<CudaTexture<float4>> clr;
  std::unique_ptr<CudaTexture<float4>> clr_nxt;
  std::unique_ptr<CudaSurface<float>> div;
  std::unique_ptr<CudaSurface<float>> p;
  std::unique_ptr<CudaSurface<float>> p_nxt;

  std::vector<std::unique_ptr<CudaSurface<float>>> res;
  std::vector<std::unique_ptr<CudaSurface<float>>> res2;
  std::vector<std::unique_ptr<CudaSurface<float>>> err2;
  std::vector<uint> sizes;
  explicit GpuSmokeSimulator(unsigned int _n, unsigned int _n0 = 16)
    : n(_n)
      , loc(std::make_unique<CudaSurface<float4>>(uint3{n, n, n}))
      , vel(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
      , vel_nxt(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
      , clr(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
      , clr_nxt(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
      , div(std::make_unique<CudaSurface<float>>(uint3{n, n, n})) {
    for (uint tn = n; tn >= _n0; tn >>= 1) {
      res.push_back(std::make_unique<CudaSurface<float>>(uint3{tn, tn, tn}));
      res2.push_back(
          std::make_unique<CudaSurface<float>>(uint3{tn / 2, tn / 2, tn / 2}));
      err2.push_back(
          std::make_unique<CudaSurface<float>>(uint3{tn / 2, tn / 2, tn / 2}));
      sizes.push_back(tn);
    }
  }
  __global__ void AdvectKernel(CudaTextureAccessor<float4> texVel,
                               CudaSurfaceAccessor<float4> surf_loc, uint n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
      return;
    auto sample = [](CudaTextureAccessor<float4> tex, float3 loc) -> float3 {
      float4 vel = tex.sample(loc.x, loc.y, loc.z);
      return make_float3(vel.x, vel.y, vel.z);
    };

    float3 loc = make_float3(x + 0.5f, y + 0.5f, z + 0.5f);
    float3 vel1 = sample(texVel, loc);
    float3 vel2 = sample(texVel, loc - 0.5f * vel1);
    float3 vel3 = sample(texVel, loc - 0.75f * vel2);
    loc -= (2.f / 9.f) * vel1 + (1.f / 3.f) * vel2 + (4.f / 9.f) * vel3;
    surf_loc.write(make_float4(loc.x, loc.y, loc.z, 0.f), x, y, z);
  }

  __global__ void ResampleKernel(CudaSurfaceAccessor<float4> surf_loc,
                                 CudaTextureAccessor<float4> tex,
                                 CudaSurfaceAccessor<float4> tex_nxt, uint n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
      return;
    float4 loc = surf_loc.read(x, y, z);
    float4 val = tex.sample(x, y, z);
    tex_nxt.write(val, x, y, z);
  }

  __global__ void DivergenceKernel(CudaSurfaceAccessor<float4> surf_vel,
                                   CudaSurfaceAccessor<float> surf_div,
                                   uint n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
      return;
    float vxn = surf_vel.read<cudaBoundaryModeZero>(x + 1, y, z).x;
    float vxp = surf_vel.read<cudaBoundaryModeZero>(x - 1, y, z).x;
    float vyn = surf_vel.read<cudaBoundaryModeZero>(x, y + 1, z).y;
    float vyp = surf_vel.read<cudaBoundaryModeZero>(x, y - 1, z).y;
    float vzn = surf_vel.read<cudaBoundaryModeZero>(x, y, z + 1).y;
    float vzp = surf_vel.read<cudaBoundaryModeZero>(x, y, z - 1).y;
    float div = (vxn - vxp + vyn - vyp + vzn - vzp) * 0.5f;
    surf_div.write(div, x, y, z);
  }

  __global__ void JacobiKernel(CudaSurfaceAccessor<float> surf_div,
                               CudaSurfaceAccessor<float> surf_p,
                               CudaSurfaceAccessor<float> surf_p_nxt, uint n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
      return;
    float pxp = surf_p.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pxn = surf_p.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pyp = surf_p.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pyn = surf_p.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pzp = surf_p.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float pzn = surf_p.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float div = surf_div.read(x, y, z);
    surf_p_nxt.write((pxp + pxn + pyp + pyn + pzp + pzn - div) / 6.f, x, y, z);
  }

  __global__ void SubgradientKernel(CudaSurfaceAccessor<float> surf_p,
                                    CudaSurfaceAccessor<float4> surf_vel,
                                    //                                CudaSurfaceAccessor<char> surf_bound,
                                    uint n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n)
      return;
    //  if (surf_bound.read(x, y, z) < 0) return ;
    float pxp = surf_p.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pxn = surf_p.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pyp = surf_p.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pyn = surf_p.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pzp = surf_p.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float pzn = surf_p.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float4 vel = surf_vel.read(x, y, z);
    vel.x -= (pxn - pxp) * 0.5f;
    vel.y -= (pyn - pyp) * 0.5f;
    vel.z -= (pzn - pzp) * 0.5f;
    surf_vel.write(vel, x, y, z);
  }
  void advection() {
    uint nthreads_dim = 8;
    uint nblocks = (n + nthreads_dim - 1) / 8;
    AdvectKernel<<<dim3(nblocks, nblocks, nblocks),
        dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
            vel->TexAccessor(), loc->surfAccessor(), n);
    ResampleKernel<<<dim3(nblocks, nblocks, nblocks),
        dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
            loc->surfAccessor(), clr->TexAccessor(), vel_nxt->surfAccessor(),
            n);
    ResampleKernel<<<dim3(nblocks, nblocks, nblocks),
        dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
            loc->surfAccessor(), vel->TexAccessor(), vel_nxt->surfAccessor(),
            n);
    std::swap(vel, vel_nxt);
    std::swap(clr, clr_nxt);
  }

  void projection() {
    uint nthreads_dim = 8;
    uint nblocks = (n + nthreads_dim - 1) / 8;
    DivergenceKernel<<<dim3(nblocks, nblocks, nblocks),
        dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
            vel->surfAccessor(), div->surfAccessor(), n);
    for (int i = 0; i < 400; i++) {
      JacobiKernel<<<dim3(nblocks, nblocks, nblocks),
          dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
              div->surfAccessor(), p->surfAccessor(), p_nxt->surfAccessor(), n);
      std::swap(p, p_nxt);
    }
    SubgradientKernel<<<dim3(nblocks, nblocks, nblocks),
        dim3(nthreads_dim, nthreads_dim, nthreads_dim)>>>(
            p->surfAccessor(), vel->surfAccessor(), n);
  }
  void substep(float dt) {

  }
  void step(float dt) {
    float t = 0.f;
    while (t < dt) {

    }
  }
};
}

#endif