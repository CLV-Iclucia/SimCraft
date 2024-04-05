//
// Created by creeper on 24-3-21.
//
#include <memory>
#include <FluidSim/cuda/advect-solver.h>
#include <Core/cuda-utils.h>
#include <device_launch_parameters.h>
#include <FluidSim/cuda/vec-op.cuh>

namespace fluid::cuda {
static CUDA_GLOBAL void kernelPicAdvect(int nParticles,
                                        VelAccessor vel,
                                        PosAccessor pos,
                                        CudaTextureAccessor<double> u,
                                        CudaTextureAccessor<double> v,
                                        CudaTextureAccessor<double> w,
                                        double h,
                                        Real dt) {
  get_and_restrict_tid(tid, nParticles);
  const double3& velocity = vel.read(tid);
  double3 p = pos.read(tid) / h;
  double3 cur_vel = make_double3(u.sample(p.x, p.y, p.z),
                                 v.sample(p.x, p.y, p.z),
                                 w.sample(p.x, p.y, p.z));
  double3 mid_p = p + cur_vel * 0.5 * dt;
  mid_p.x = ::clamp(mid_p, 0.0, 1.0);
  mid_p.y = ::clamp(mid_p, 0.0, 1.0);
  mid_p.z = ::clamp(mid_p, 0.0, 1.0);
  double3 mid_vel = make_double3(u.sample(mid_p.x, mid_p.y, mid_p.z),
                                 v.sample(mid_p.x, mid_p.y, mid_p.z),
                                 w.sample(mid_p.x, mid_p.y, mid_p.z));
  double3 final_p = p + mid_vel * 0.5 * dt;
  pos.write(tid, final_p * h);
}

static CUDA_GLOBAL void kernelPicG2P(int nParticles,
                                     VelAccessor vel,
                                     PosAccessor pos,
                                     CudaTextureAccessor<double> u,
                                     CudaTextureAccessor<double> v,
                                     CudaTextureAccessor<double> w,
                                     Real h, Real dt) {
  get_and_restrict_tid(tid, nParticles);
  double3 p = pos.read(tid) / h;
  double3 cur_vel = make_double3(u.sample(p.x, p.y, p.z),
                                 v.sample(p.x, p.y, p.z),
                                 w.sample(p.x, p.y, p.z));
  pos.write(tid, pos.read(tid) + cur_vel * dt);
}
void PicSolver::advect(const ParticleSystem& particles,
                       CudaTextureAccessor<double> u,
                       CudaTextureAccessor<double> v,
                       CudaTextureAccessor<double> w,
                       Real h,
                       Real dt) {
  kernelPicAdvect<<<LAUNCH_THREADS(particles.size())>>>(
      particles.size(),
      particles.velAccessor(),
      particles.posAccessor(),
      u, v, w, h, dt);
}

void PicSolver::solveG2P(const ParticleSystem& particles,
                         CudaTextureAccessor<double> u,
                         CudaTextureAccessor<double> v,
                         CudaTextureAccessor<double> w,
                         Real h, Real dt) {
  kernelPicG2P<<<LAUNCH_THREADS(particles.size())>>>(
      particles.size(),
      particles.velAccessor(),
      particles.posAccessor(),
      u, v, w, h, dt);
}

void PicSolver::solveP2G(const ParticleSystem& particles, Real h, Real dt) {
}
}