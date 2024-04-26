//
// Created by creeper on 24-3-21.
//

#ifndef SIM_CRAFT_ADVECT_SOLVER_H
#define SIM_CRAFT_ADVECT_SOLVER_H
#include <memory>
#include <FluidSim/fluid-sim.h>
#include <FluidSim/cuda/particle-system.h>

namespace fluid::cuda {
struct ParticleLevelSetSolver {
  virtual void advect(ParticleSystem &particles,
                      const CudaTexture<float> &u,
                      const CudaTexture<float> &v,
                      const CudaTexture<float> &w,
                      int3 resolution,
                      float h,
                      float dt) = 0;
  virtual void moveParticles(const ParticleSystem &particles,
                             const CudaTexture<float> &u,
                             const CudaTexture<float> &v,
                             const CudaTexture<float> &w,
                             int3 resolution,
                             float h, float dt) = 0;
  virtual ~ParticleLevelSetSolver() = default;
};

struct SemiLagrangianSolver final : ParticleLevelSetSolver {
  void advect(ParticleSystem &particles,
              const CudaTexture<float> &u,
              const CudaTexture<float> &v,
              const CudaTexture<float> &w,
              int3 resolution,
              float h,
              float dt) override;
  void moveParticles(const ParticleSystem &particles, const CudaTexture<float>& u,
                     const CudaTexture<float>& v, const CudaTexture<float>& w,
                     int3 resolution,
                     float h, float dt) override;
  SemiLagrangianSolver() = default;
};
}
#endif //SIM_CRAFT_ADVECT_SOLVER_H