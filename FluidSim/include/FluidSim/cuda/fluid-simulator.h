//
// Created by creeper on 24-3-20.
//

#ifndef SIM_CRAFT_CUDA_FLUID_SIMULATOR_H
#define SIM_CRAFT_CUDA_FLUID_SIMULATOR_H

//#include <FluidSim/fluid-simulator.h>
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <FluidSim/cuda/project-solver.h>
#include <FluidSim/cuda/advect-solver.h>
#include <format>
namespace fluid::cuda {
class FluidSimulator {
 public:
  FluidSimulator(int nParticles_, const Vec3d &size, const Vec3i &resolution) {

  }
  void setCollider(const Mesh &collider_mesh) const {
  }

  void setInitialFluid(const Mesh &fluid_mesh);

  int3 resolution;
  Real h;
  std::unique_ptr<CudaTexture<float>> fluidSurface{}, fluidSurfaceBuf{};
  std::unique_ptr<CudaTexture<float>> colliderSdf{};
  std::unique_ptr<CudaTexture<float>> u{}, v{}, w{}, uBuf{}, vBuf{}, wBuf{};
  std::unique_ptr<CudaSurface<float>> uw{}, vw{}, ww{}, p{};
  std::unique_ptr<CudaSurface<uint8_t>> uValid{}, vValid{}, wValid{}, uValidBuf{}, vValidBuf{}, wValidBuf{}, sdfValid{},
      sdfValidBuf{};
  std::unique_ptr<ProjectionSolver> projectionSolver{};
  std::unique_ptr<AdvectionSolver> advectionSolver{};
  std::unique_ptr<ParticleSystem> particles{};
  void substep(Real dt);
  double CFL() const;
  void step(core::Frame &frame) {
    Real t = 0;
    std::cout << std::format("********* Frame {} *********", frame.idx) <<
              std::endl;
    int substep_cnt = 0;
    while (t < frame.dt) {
      Real cfl = CFL();
      Real dt = std::min(cfl, frame.dt - t);
      substep_cnt++;
      std::cout << std::format("<<<<< Substep {}, dt = {} >>>>>", substep_cnt, dt)
                << std::endl;
      substep(dt);
      t += dt;
    }
    frame.onAdvance();
  }

  ~FluidSimulator() = default;
  void clear() const;
  void smoothFluidSurface(int iters);
  void applyCollider() const;
  void applyForce(Real dt) const;
  void applyDirichletBoundary() const;
  void extrapolateFluidSdf(int iters);
};
}

#endif //SIM_CRAFT_CUDA_FLUID_SIMULATOR_H