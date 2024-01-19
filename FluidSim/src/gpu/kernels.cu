#include <FluidSim/gpu/kernels.cuh>
#include <FluidSim/gpu/gpu-arrays.h>
#include <FluidSim/gpu/vec-op.cuh>

namespace fluid {
__global__ void AdvectKernel(CudaTextureAccessor<float4> texVel,
                             CudaSurfaceAccessor<float4> surf_loc, uint n, float dt) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  if (x >= n || y >= n || z >= n)
    return;
  auto sample = [](CudaTextureAccessor<float4> tex, float3 loc) -> float3 {
    auto [x, y, z, w] = tex.sample(loc.x, loc.y, loc.z);
    return make_float3(x, y, z);
  };

  float3 loc = makeCellCenter(x, y, z);
  float3 vel1 = sample(texVel, loc);
  assert(vel1.x == vel1.x && vel1.y == vel1.y && vel1.z == vel1.z);
  float3 vel2 = sample(texVel, loc - 0.5f * vel1 * dt);
  assert(vel2.x == vel2.x && vel2.y == vel2.y && vel2.z == vel2.z);
  float3 vel3 = sample(texVel, loc - 0.75f * vel2 * dt);
  assert(vel3.x == vel3.x && vel3.y == vel3.y && vel3.z == vel3.z);
  loc -= (2.f / 9.f) * vel1 * dt + (1.f / 3.f) * vel2 * dt + (4.f / 9.f) * vel3 * dt;
  assert(loc.x == loc.x && loc.y == loc.y && loc.z == loc.z);
  surf_loc.write(make_float4(loc.x, loc.y, loc.z, 0.f), x, y, z);
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

__global__ void CurlKernel(CudaSurfaceAccessor<float4> surf_vel,
                           CudaSurfaceAccessor<float4> surf_curl,
                           uint n) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  if (x >= n || y >= n || z >= n)
    return;
  float vxny = surf_vel.read<cudaBoundaryModeZero>(x, y + 1, z).x;
  float vxpy = surf_vel.read<cudaBoundaryModeZero>(x, y - 1, z).x;
  float vxnz = surf_vel.read<cudaBoundaryModeZero>(x, y, z + 1).x;
  float vxpz = surf_vel.read<cudaBoundaryModeZero>(x, y, z - 1).x;
  float vynx = surf_vel.read<cudaBoundaryModeZero>(x + 1, y, z).y;
  float vypx = surf_vel.read<cudaBoundaryModeZero>(x - 1, y, z).y;
  float vynz = surf_vel.read<cudaBoundaryModeZero>(x, y, z + 1).y;
  float vypz = surf_vel.read<cudaBoundaryModeZero>(x, y, z - 1).y;
  float vznx = surf_vel.read<cudaBoundaryModeZero>(x + 1, y, z).z;
  float vzpx = surf_vel.read<cudaBoundaryModeZero>(x - 1, y, z).z;
  float vzny = surf_vel.read<cudaBoundaryModeZero>(x, y + 1, z).z;
  float vzpy = surf_vel.read<cudaBoundaryModeZero>(x, y - 1, z).z;
  float pvzpy = (vzny - vzpy) * 0.5f;
  float pvypz = (vynz - vypz) * 0.5f;
  float pvzpx = (vznx - vzpx) * 0.5f;
  float pvxpz = (vxnz - vxpz) * 0.5f;
  float pvxpy = (vxny - vxpy) * 0.5f;
  float pvypx = (vynx - vypx) * 0.5f;
  float curlx = pvzpy - pvypz;
  float curly = pvzpx - pvxpz;
  float curlz = pvxpy - pvypx;
  return surf_curl.write(make_float4(curlx, curly, curlz, 0.f), x, y, z);
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
  assert(div == div);
  surf_p_nxt.write((pxp + pxn + pyp + pyn + pzp + pzn - div) / 6.f, x, y, z);
}

__global__ void SubgradientKernel(CudaSurfaceAccessor<float> surf_p,
                                  CudaSurfaceAccessor<float4> surf_vel,
                                  float4 src_vel, uint n) {
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
  assert(vel.x == vel.x && vel.y == vel.y && vel.z == vel.z);
  vel.x -= (pxn - pxp) * 0.5f;
  vel.y -= (pyn - pyp) * 0.5f;
  vel.z -= (pzn - pzp) * 0.5f;
  assert(vel.x == vel.x && vel.y == vel.y && vel.z == vel.z);
  float4 result = withinSource(x, y, z, n) ? src_vel : vel;
  surf_vel.write(result, x, y, z);
}
__global__ void AccumulateForceKernel(CudaSurfaceAccessor<float4> surf_force,
                                      CudaSurfaceAccessor<float4> surf_vort,
                                      CudaSurfaceAccessor<float> surf_T,
                                      CudaSurfaceAccessor<float> surf_rho,
                                      uint n, float alpha, float beta,
                                      float epsilon,
                                      float ambientTemperature, float dt) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  if (x >= n || y >= n || z >= n)
    return;
  // Buoyancy = -alpha * (T - T_ambient) * z + beta * (T - T_ambient) * z
  float T = surf_T.read(x, y, z);
  assert(T == T);
  float rho = surf_rho.read(x, y, z);
  assert(rho == rho);
  float3 loc = makeCellCenter(x, y, z);
  float buoyancy = -alpha * rho * loc.y / n + beta * (T - ambientTemperature) * loc.y / n;
  assert(buoyancy == buoyancy);
  float nwnx = norm(surf_vort.read<cudaBoundaryModeZero>(x + 1, y, z));
  float nwpx = norm(surf_vort.read<cudaBoundaryModeZero>(x - 1, y, z));
  float nwny = norm(surf_vort.read<cudaBoundaryModeZero>(x, y + 1, z));
  float nwpy = norm(surf_vort.read<cudaBoundaryModeZero>(x, y - 1, z));
  float nwnz = norm(surf_vort.read<cudaBoundaryModeZero>(x, y, z + 1));
  float nwpz = norm(surf_vort.read<cudaBoundaryModeZero>(x, y, z - 1));
  float4 vort = surf_vort.read(x, y, z);
  assert(vort.x == vort.x && vort.y == vort.y && vort.z == vort.z);
  float3 eta = make_float3((nwnx - nwpx) * 0.5f,
                           (nwny - nwpy) * 0.5f,
                           (nwnz - nwpz) * 0.5f);
  assert(eta.x == eta.x && eta.y == eta.y && eta.z == eta.z);
  float3 confine = epsilon * cross(eta, make_float3(vort.x, vort.y, vort.z));
  float neta = norm(eta);
  confine = neta == 0.f ? make_float3(0.f, 0.f, 0.f) : confine / neta;
  assert(
      confine.x == confine.x && confine.y == confine.y && confine.z == confine.
      z);
  float wind = 9.8f;
  float gravity = -9.8f;
  surf_force.write(make_float4(confine.x, confine.y + buoyancy + gravity,
                               confine.z + wind, 0.f), x, y, z);
}
__global__ void ApplyForceKernel(CudaSurfaceAccessor<float4> surf_vel,
                                 CudaSurfaceAccessor<float4> surf_vel_nxt,
                                 CudaSurfaceAccessor<float4> surf_force, uint n,
                                 float dt) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  if (x >= n || y >= n || z >= n)
    return;
  float4 vel = surf_vel.read(x, y, z);
  float4 force = surf_force.read(x, y, z);
  surf_vel_nxt.write(vel + dt * force, x, y, z);
}
__global__ void CoolingKernel(CudaSurfaceAccessor<float> surf_T,
                              CudaSurfaceAccessor<float> surf_T_nxt,
                              CudaSurfaceAccessor<float> surf_rho,
                              CudaSurfaceAccessor<float> surf_rho_nxt,
                              uint n, float ambientTemperature, float dt) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  if (x >= n || y >= n || z >= n)
    return;
  float T = surf_T.read(x, y, z);
  float rho = surf_rho.read(x, y, z);
  float coolingRate = (T - ambientTemperature);
  float decayingRate = rho;
  surf_T_nxt.write(T - coolingRate * dt, x, y, z);
  surf_rho_nxt.write(rho - 0.1f * decayingRate * dt, x, y, z);
}
__global__ void SmoothingKernel(CudaSurfaceAccessor<float> surf_rho,
                                CudaSurfaceAccessor<float> surf_rho_nxt,
                                uint n) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  if (withinSource(x, y, z, n))
    return;
  float rxp = surf_rho.read<cudaBoundaryModeClamp>(x + 1, y, z);
  float rxn = surf_rho.read<cudaBoundaryModeClamp>(x - 1, y, z);
  float ryp = surf_rho.read<cudaBoundaryModeClamp>(x, y + 1, z);
  float ryn = surf_rho.read<cudaBoundaryModeClamp>(x, y - 1, z);
  float rzp = surf_rho.read<cudaBoundaryModeClamp>(x, y, z + 1);
  float rzn = surf_rho.read<cudaBoundaryModeClamp>(x, y, z - 1);
  float rel = surf_rho.read(x, y, z);
  assert(vel.x == vel.x && vel.y == vel.y && vel.z == vel.z);
  float result = (rxp + rxn + ryp + ryn + rzp + rzn + 6.f * rel) / 12.f;
  surf_rho_nxt.write(result, x, y, z);
}
}