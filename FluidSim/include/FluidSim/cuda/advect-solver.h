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
  virtual void advect(ParticleSystem &particles,
                      const CudaTexture<Real> &u,
                      const CudaTexture<Real> &v,
                      const CudaTexture<Real> &w,
                      int3 resolution,
                      double h,
                      Real dt) = 0;
  virtual void solveP2G(const ParticleSystem &particles,
                        const CudaTexture<Real> &u,
                        const CudaTexture<Real> &v,
                        const CudaTexture<Real> &w,
                        const CudaTexture<Real> &collider_sdf,
                        const CudaSurface<Real> &uw,
                        const CudaSurface<Real> &vw,
                        const CudaSurface<Real> &ww,
                        const CudaSurface<int> &uValid,
                        const CudaSurface<int> &vValid,
                        const CudaSurface<int> &wValid,
                        int3 resolution,
                        Real h, Real dt) = 0;
  virtual void solveG2P(const ParticleSystem &particles,
                        const CudaTexture<Real> &u,
                        const CudaTexture<Real> &v,
                        const CudaTexture<Real> &w,
                        int3 resolution,
                        Real h, Real dt) = 0;
  virtual ~AdvectionSolver() = default;
};

struct PicSolver final : AdvectionSolver {
  void advect(ParticleSystem &particles,
              const CudaTexture<Real> &u,
              const CudaTexture<Real> &v,
              const CudaTexture<Real> &w,
              int3 resolution,
              Real h,
              Real dt) override;
  void solveP2G(const ParticleSystem &particles, const CudaTexture<Real>& u,
                const CudaTexture<Real>& v, const CudaTexture<Real>& w,
                const CudaTexture<Real>& collider_sdf,
                const CudaSurface<Real>& uw, const CudaSurface<Real>& vw,
                const CudaSurface<Real>& ww, const CudaSurface<int>& uValid,
                const CudaSurface<int>& vValid,
                const CudaSurface<int>& wValid,
                int3 resolution,
                Real h, Real dt) override;
  void solveG2P(const ParticleSystem &particles, const CudaTexture<Real>& u,
                const CudaTexture<Real>& v, const CudaTexture<Real>& w,
                int3 resolution,
                Real h, Real dt) override;
  PicSolver() = default;
};
}
#endif //SIM_CRAFT_ADVECT_SOLVER_H