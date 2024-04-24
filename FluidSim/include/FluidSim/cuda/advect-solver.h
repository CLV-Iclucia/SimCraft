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
                      const CudaTexture<float> &u,
                      const CudaTexture<float> &v,
                      const CudaTexture<float> &w,
                      int3 resolution,
                      float h,
                      float dt) = 0;
  virtual void solveP2G(const ParticleSystem &particles,
                        const CudaTexture<float> &u,
                        const CudaTexture<float> &v,
                        const CudaTexture<float> &w,
                        const CudaTexture<float> &collider_sdf,
                        const CudaSurface<float> &uw,
                        const CudaSurface<float> &vw,
                        const CudaSurface<float> &ww,
                        const CudaSurface<uint8_t> &uValid,
                        const CudaSurface<uint8_t> &vValid,
                        const CudaSurface<uint8_t> &wValid,
                        int3 resolution,
                        float h, float dt) = 0;
  virtual void solveG2P(const ParticleSystem &particles,
                        const CudaTexture<float> &u,
                        const CudaTexture<float> &v,
                        const CudaTexture<float> &w,
                        int3 resolution,
                        float h, float dt) = 0;
  virtual ~AdvectionSolver() = default;
};

struct PicSolver final : AdvectionSolver {
  void advect(ParticleSystem &particles,
              const CudaTexture<float> &u,
              const CudaTexture<float> &v,
              const CudaTexture<float> &w,
              int3 resolution,
              float h,
              float dt) override;
  void solveP2G(const ParticleSystem &particles, const CudaTexture<float>& u,
                const CudaTexture<float>& v, const CudaTexture<float>& w,
                const CudaTexture<float>& collider_sdf,
                const CudaSurface<float>& uw, const CudaSurface<float>& vw,
                const CudaSurface<float>& ww, const CudaSurface<uint8_t>& uValid,
                const CudaSurface<uint8_t>& vValid,
                const CudaSurface<uint8_t>& wValid,
                int3 resolution,
                float h, float dt) override;
  void solveG2P(const ParticleSystem &particles, const CudaTexture<float>& u,
                const CudaTexture<float>& v, const CudaTexture<float>& w,
                int3 resolution,
                float h, float dt) override;
  PicSolver() = default;
};
}
#endif //SIM_CRAFT_ADVECT_SOLVER_H