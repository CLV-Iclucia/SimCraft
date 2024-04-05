//
// Created by creeper on 24-3-21.
//

#ifndef SIM_CRAFT_ADVECT_SOLVER_H
#define SIM_CRAFT_ADVECT_SOLVER_H
#include <memory>
#include <FluidSim/fluid-sim.h>
#include <FluidSim/cuda/particle-system.h>

namespace fluid::cuda {
struct AdvectionSolver {
  std::unique_ptr<ParticleSystem> particles;
  virtual void advect(const ParticleSystem& particles,
                      CudaTextureAccessor<double> u,
                      CudaTextureAccessor<double> v,
                      CudaTextureAccessor<double> w,
                      double h,
                      Real dt) = 0;
  virtual void solveP2G(const ParticleSystem& particles,
                        CudaTextureAccessor<double> u,
                        CudaTextureAccessor<double> v,
                        CudaTextureAccessor<double> w,
                        CudaTextureAccessor<double> collider_sdf,
                        CudaSurfaceAccessor<double> uw,
                        CudaSurfaceAccessor<double> vw,
                        CudaSurfaceAccessor<double> ww,
                        CudaSurfaceAccessor<int> uValid,
                        CudaSurfaceAccessor<int> vValid,
                        CudaSurfaceAccessor<int> wValid,
                        Real h, Real dt) = 0;
  virtual void solveG2P(const ParticleSystem& particles,
                        CudaTextureAccessor<double> u,
                        CudaTextureAccessor<double> v,
                        CudaTextureAccessor<double> w
                        , Real h, Real dt) = 0;
  virtual ~AdvectionSolver() = default;
};

struct PicSolver final : AdvectionSolver {
  void advect(const ParticleSystem& particles,
              CudaTextureAccessor<double> u,
              CudaTextureAccessor<double> v,
              CudaTextureAccessor<double> w,
              Real h,
              Real dt) override;
  void solveP2G(const ParticleSystem& particles, CudaTextureAccessor<double> u,
                CudaTextureAccessor<double> v, CudaTextureAccessor<double> w,
                CudaTextureAccessor<double> collider_sdf,
                CudaSurfaceAccessor<double> uw, CudaSurfaceAccessor<double> vw,
                CudaSurfaceAccessor<double> ww, CudaSurfaceAccessor<int> uValid,
                CudaSurfaceAccessor<int> vValid,
                CudaSurfaceAccessor<int> wValid,
                Real h, Real dt) override;
  void solveG2P(const ParticleSystem& particles, CudaTextureAccessor<double> u,
                CudaTextureAccessor<double> v, CudaTextureAccessor<double> w,
                Real h, Real dt) override;
  PicSolver() = default;
};
}
#endif //SIM_CRAFT_ADVECT_SOLVER_H