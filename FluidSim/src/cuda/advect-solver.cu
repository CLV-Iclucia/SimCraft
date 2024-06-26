//
// Created by creeper on 24-3-21.
//
#include <memory>
#include <FluidSim/cuda/advect-solver.h>
#include <Core/utils.h>
#include <FluidSim/cuda/vec-op.cuh>

namespace fluid::cuda {
static CUDA_GLOBAL void kernelPicAdvect(int nParticles,
                                        VelAccessor vel,
                                        PosAccessor pos,
                                        CudaTextureAccessor<float> u,
                                        CudaTextureAccessor<float> v,
                                        CudaTextureAccessor<float> w,
                                        int3 resolution,
                                        double h,
                                        float dt) {
  get_and_restrict_tid(tid, nParticles);
  const double3 &velocity = vel.read(tid);
  double3 p = pos.read(tid);
  double3 cur_vel = make_double3(u.sample(p.x / h, p.y / h, p.z / h),
                                 v.sample(p.x / h, p.y / h, p.z / h),
                                 w.sample(p.x / h, p.y / h, p.z / h));
  double3 mid_p = p + cur_vel * 0.5 * dt;
  mid_p.x = core::clamp(mid_p.x, 0.0, resolution.x * h);
  mid_p.y = core::clamp(mid_p.y, 0.0, resolution.y * h);
  mid_p.z = core::clamp(mid_p.z, 0.0, resolution.z * h);
  double3 mid_vel = make_double3(u.sample(mid_p.x / h, mid_p.y / h, mid_p.z / h),
                                 v.sample(mid_p.x / h, mid_p.y / h, mid_p.z / h),
                                 w.sample(mid_p.x / h, mid_p.y / h, mid_p.z / h));
  double3 final_p = p + mid_vel * 0.5 * dt;
  pos.write(tid, final_p);
}

static CUDA_GLOBAL void kernelPicG2P(int nParticles,
                                     VelAccessor vel,
                                     PosAccessor pos,
                                     CudaTextureAccessor<float> u,
                                     CudaTextureAccessor<float> v,
                                     CudaTextureAccessor<float> w,
                                     int3 resolution,
                                     float h, float dt) {
  get_and_restrict_tid(tid, nParticles);
  double3 p = pos.read(tid);
  double3 cur_vel = make_double3(u.sample(p.x / h, p.y / h, p.z / h),
                                 v.sample(p.x / h, p.y / h, p.z / h),
                                 w.sample(p.x / h, p.y / h, p.z / h));
  pos.write(tid, pos.read(tid) + cur_vel * dt);
}

static CUDA_GLOBAL void kernelPicP2G(int nParticles,
                                     VelAccessor vel,
                                     PosAccessor pos,
                                     CudaTextureAccessor<float> u,
                                     CudaTextureAccessor<float> v,
                                     CudaTextureAccessor<float> w,
                                     float h, float dt) {
  get_and_restrict_tid(tid, nParticles);
  double3 p = pos.read(tid);

}

void SemiLagrangianSolver::advect(ParticleSystem &particles,
                                  const CudaTexture<float> &u,
                                  const CudaTexture<float> &v,
                                  const CudaTexture<float> &w,
                                  int3 resolution,
                                  float h,
                                  float dt) {
  kernelPicAdvect<<<LAUNCH_THREADS(particles.size())>>>(
      particles.size(),
      particles.velAccessor(),
      particles.posAccessor(),
      u.texAccessor(),
      v.texAccessor(),
      w.texAccessor(),
      resolution, h, dt);
}

void SemiLagrangianSolver::moveParticles(const ParticleSystem &particles,
                                         const CudaTexture<float> &u,
                                         const CudaTexture<float> &v,
                                         const CudaTexture<float> &w,
                                         int3 resolution,
                                         float h, float dt) {
  kernelPicG2P<<<LAUNCH_THREADS(particles.size())>>>(
      particles.size(),
      particles.velAccessor(),
      particles.posAccessor(),
      u.texAccessor(), v.texAccessor(), w.texAccessor(),
      resolution, h, dt);
}
}