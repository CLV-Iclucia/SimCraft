#include <FluidSim/cuda/kernels.cuh>
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <FluidSim/cuda/vec-op.cuh>

namespace fluid::cuda {
__global__ void AdvectKernel(CudaTextureAccessor<float4> texVel,
                             CudaSurfaceAccessor<float4> surf_loc,
                             CudaSurfaceAccessor<uint8_t> active,
                             uint n,
                             float dt) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z))
    return;
  auto sample = [](CudaTextureAccessor<float4> tex, float3 loc) -> float3 {
    auto [x, y, z, w] = tex.sample(loc.x, loc.y, loc.z);
    return make_float3(x, y, z);
  };

  float3 loc = makeCellCenter(x, y, z);
  float3 vel1 = sample(texVel, loc);
  float3 vel2 = sample(texVel, loc - 0.5f * vel1 * dt);
  float3 vel3 = sample(texVel, loc - 0.75f * vel2 * dt);
  loc -= (2.f / 9.f) * vel1 * dt + (1.f / 3.f) * vel2 * dt + (4.f / 9.f) * vel3
      * dt;
  surf_loc.write(make_float4(loc.x, loc.y, loc.z, 0.f), x, y, z);
}

__global__ void DivergenceKernel(CudaSurfaceAccessor<float4> surf_vel,
                                 CudaSurfaceAccessor<float> surf_div,
                                 CudaSurfaceAccessor<uint8_t> active,
                                 uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z))
    return;
  float vxn = surf_vel.read<cudaBoundaryModeZero>(x + 1, y, z).x * active.read<cudaBoundaryModeZero>(x + 1, y, z);
  float vxp = surf_vel.read<cudaBoundaryModeZero>(x - 1, y, z).x * active.read<cudaBoundaryModeZero>(x - 1, y, z);
  float vyn = surf_vel.read<cudaBoundaryModeZero>(x, y + 1, z).y * active.read<cudaBoundaryModeZero>(x, y + 1, z);
  float vyp = surf_vel.read<cudaBoundaryModeZero>(x, y - 1, z).y * active.read<cudaBoundaryModeZero>(x, y - 1, z);
  float vzn = surf_vel.read<cudaBoundaryModeZero>(x, y, z + 1).z * active.read<cudaBoundaryModeZero>(x, y, z + 1);
  float vzp = surf_vel.read<cudaBoundaryModeZero>(x, y, z - 1).z * active.read<cudaBoundaryModeZero>(x, y, z - 1);
  float div = (vxn - vxp + vyn - vyp + vzn - vzp) * 0.5f;
  surf_div.write(div, x, y, z);
}

__global__ void CurlKernel(CudaSurfaceAccessor<float4> surf_vel,
                           CudaSurfaceAccessor<float4> surf_curl,
                           CudaSurfaceAccessor<uint8_t> active,
                           uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z))
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
  get_and_restrict_tid_3d(x, y, z, n, n, n);
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
                                  CudaSurfaceAccessor<uint8_t> active,
                                  float4 src_vel, uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z))
    return;
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
  float4 result = withinSource(x, y, z, n) ? normalize(make_float4(x - n / 2, y - n / 2, z - n / 2, 0.f)) : vel;
  surf_vel.write(result, x, y, z);
}
__global__ void AccumulateForceKernel(CudaSurfaceAccessor<float4> surf_force,
                                      CudaSurfaceAccessor<float4> surf_vort,
                                      CudaSurfaceAccessor<float> surf_T,
                                      CudaSurfaceAccessor<float> surf_rho,
                                      CudaSurfaceAccessor<uint8_t> active,
                                      uint n, float alpha, float beta,
                                      float epsilon,
                                      float ambientTemperature, float t) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z))
    return;
  float T = surf_T.read(x, y, z);
  float rho = surf_rho.read(x, y, z);
  float3 loc = makeCellCenter(x, y, z);
  float buoyancy = -alpha * rho * loc.y / n + beta * (T - ambientTemperature) *
      loc.y / n;
  float nwnx = norm(surf_vort.read<cudaBoundaryModeZero>(x + 1, y, z));
  float nwpx = norm(surf_vort.read<cudaBoundaryModeZero>(x - 1, y, z));
  float nwny = norm(surf_vort.read<cudaBoundaryModeZero>(x, y + 1, z));
  float nwpy = norm(surf_vort.read<cudaBoundaryModeZero>(x, y - 1, z));
  float nwnz = norm(surf_vort.read<cudaBoundaryModeZero>(x, y, z + 1));
  float nwpz = norm(surf_vort.read<cudaBoundaryModeZero>(x, y, z - 1));
  float4 vort = surf_vort.read(x, y, z);
  float3 eta = make_float3((nwnx - nwpx) * 0.5f,
                           (nwny - nwpy) * 0.5f,
                           (nwnz - nwpz) * 0.5f);
  float3 confine = 1.75 * epsilon * cross(eta, make_float3(vort.x, vort.y, vort.z));
  float neta = norm(eta);
  confine = neta == 0.f || withinSource(x, y, z, n)
            ? make_float3(0.f, 0.f, 0.f)
            : confine / neta;
  float wind = 48.f;
  float gravity = 9.8f;
  surf_force.write(make_float4(confine.x + wind * cos(t / 10), confine.y + buoyancy + gravity,
                               confine.z + wind * sin(t / 10), 0.f), x, y, z);
}
__global__ void ApplyForceKernel(CudaSurfaceAccessor<float4> surf_vel,
                                 CudaSurfaceAccessor<float4> surf_vel_nxt,
                                 CudaSurfaceAccessor<float4> surf_force,
                                 CudaSurfaceAccessor<uint8_t> active,
                                 uint n,
                                 float dt) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z))
    return;
  float4 vel = surf_vel.read(x, y, z);
  float4 force = surf_force.read(x, y, z);
  surf_vel_nxt.write(vel + dt * force, x, y, z);
}
__global__ void CoolingKernel(CudaSurfaceAccessor<float> surf_T,
                              CudaSurfaceAccessor<float> surf_T_nxt,
                              CudaSurfaceAccessor<float> surf_rho,
                              CudaSurfaceAccessor<float> surf_rho_nxt,
                              CudaSurfaceAccessor<uint8_t> active,
                              uint n, float ambientTemperature, float dt) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  float rho = surf_rho.read(x, y, z);
  float T = surf_T.read(x, y, z);
  float txp = surf_T.read<cudaBoundaryModeClamp>(x + 1, y, z) * active.read<cudaBoundaryModeZero>(x + 1, y, z);
  float txn = surf_T.read<cudaBoundaryModeClamp>(x - 1, y, z) * active.read<cudaBoundaryModeZero>(x - 1, y, z);
  float typ = surf_T.read<cudaBoundaryModeClamp>(x, y + 1, z) * active.read<cudaBoundaryModeZero>(x, y + 1, z);
  float tyn = surf_T.read<cudaBoundaryModeClamp>(x, y - 1, z) * active.read<cudaBoundaryModeZero>(x, y - 1, z);
  float tzp = surf_T.read<cudaBoundaryModeClamp>(x, y, z + 1) * active.read<cudaBoundaryModeZero>(x, y, z + 1);
  float tzn = surf_T.read<cudaBoundaryModeClamp>(x, y, z - 1) * active.read<cudaBoundaryModeZero>(x, y, z - 1);
  float avg = (txp + txn + typ + tyn + tzp + tzn) / 6.f;
  surf_T_nxt.write((T - (T - avg) * dt) * exp(-0.01f), x, y, z);
  surf_rho_nxt.write(rho - rho * 0.2 * dt, x, y, z);
}
__global__ void SmoothingKernel(CudaSurfaceAccessor<float> surf_rho,
                                CudaSurfaceAccessor<float> surf_rho_nxt,
                                CudaSurfaceAccessor<uint8_t> active,
                                uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z))
    return;
  if (withinSource(x, y, z, n))
    return;
  float rxp = surf_rho.read<cudaBoundaryModeClamp>(x + 1, y, z) * active.read(x + 1, y, z);
  float rxn = surf_rho.read<cudaBoundaryModeClamp>(x - 1, y, z) * active.read(x - 1, y, z);
  float ryp = surf_rho.read<cudaBoundaryModeClamp>(x, y + 1, z) * active.read(x, y + 1, z);
  float ryn = surf_rho.read<cudaBoundaryModeClamp>(x, y - 1, z) * active.read(x, y - 1, z);
  float rzp = surf_rho.read<cudaBoundaryModeClamp>(x, y, z + 1) * active.read(x, y, z + 1);
  float rzn = surf_rho.read<cudaBoundaryModeClamp>(x, y, z - 1) * active.read(x, y, z - 1);
  float rel = surf_rho.read(x, y, z);
  float result = (rxp + rxn + ryp + ryn + rzp + rzn + 6.f * rel) / 12.f;
  surf_rho_nxt.write(result, x, y, z);
}
__global__ void kernelSetupFluidRegion(CudaSurfaceAccessor<uint8_t> surf,
                                       DeviceArrayAccessor<uint8_t> region,
                                       uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  int idx = x * n * n + y * n + z; // by default the region should be in xyz order
  surf.write(region[idx], x, y, z);
}
}