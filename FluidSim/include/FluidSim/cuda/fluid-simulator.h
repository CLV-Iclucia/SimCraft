//
// Created by creeper on 24-3-20.
//

#ifndef SIM_CRAFT_CUDA_FLUID_SIMULATOR_H
#define SIM_CRAFT_CUDA_FLUID_SIMULATOR_H

#include <FluidSim/fluid-simulator.h>
#include <FluidSim/cuda/gpu-arrays.h>
#include <FluidSim/cuda/project-solver.h>
#include <FluidSim/cuda/advect-solver.h>

namespace fluid::cuda {
class FluidSimulator final : public FluidComputeBackend {
 public:
  void setCollider(const Mesh &collider_mesh) const override {
  }

  void setInitialFluid(const Mesh &fluid_mesh) override;

  void setAdvector(AdvectionMethod advection_method) override {
    if (advection_method == AdvectionMethod::PIC) {
    }
  }

  void setProjector(ProjectSolver project_solver) override {
    if (project_solver == ProjectSolver::FVM) {
      projectSolver = std::make_unique<FvmSolver>(resolution.x, resolution.y,
                                                  resolution.z);
    } else {
      ERROR("Invalid project solver");
    }
  }
  int3 resolution;
  Real h;
  std::unique_ptr<CudaTexture<double>> fluidSurface, fluidSurfaceBuf;
  std::unique_ptr<CudaTexture<double>> colliderSdf;
  std::unique_ptr<CudaTexture<double>> u, v, w, uBuf, vBuf, wBuf;
  std::unique_ptr<CudaSurface<double>> uw, vw, ww, p;
  std::unique_ptr<CudaSurface<uint8_t>>
      uValid, vValid, wValid, uValidBuf, vValidBuf, wValidBuf, sdfValid,
      sdfValidBuf;
  std::unique_ptr<ProjectionSolver> projectSolver{};
  std::unique_ptr<AdvectionSolver> advectionSolver{};
  void step(Real dt) {
  }

  ~FluidSimulator() override = default;
  void smoothFluidSurface(int iters);
  void applyCollider() const;
};
}

#endif //SIM_CRAFT_CUDA_FLUID_SIMULATOR_H