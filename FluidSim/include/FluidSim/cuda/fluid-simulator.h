//
// Created by creeper on 24-3-20.
//

#ifndef SIM_CRAFT_CUDA_FLUID_SIMULATOR_H
#define SIM_CRAFT_CUDA_FLUID_SIMULATOR_H

#include <FluidSim/fluid-simulator.h>

#include <FluidSim/cuda/gpu-arrays.h>

namespace fluid::cuda {
class FluidSimulator final : public FluidComputeBackend, NonCopyable {
public:
  void setCollider(const Mesh& collider_mesh) const override {
  }

  void setInitialFluid(const Mesh& fluid_mesh) override;

  void setAdvector(AdvectionMethod advection_method) override {
    if (advection_method == AdvectionMethod::PIC) {
    }
  }

  void setProjector(ProjectSolver project_solver) override;
  std::unique_ptr<CudaTexture<double>> fluidSurface, fluidSurfaceBuf;
  std::unique_ptr<CudaTexture<double>> colliderSdf;
  std::unique_ptr<CudaTexture<double>> u, v, w, uBuf, vBuf, wBuf;
  std::unique_ptr<CudaSurface<double>> uw, vw, ww, p;
  std::unique_ptr<CudaSurface<uint8_t>>
      uValid, vValid, wValid, uValidBuf, vValidBuf, wValidBuf, sdfValid,
      sdfValidBuf;

  void step(Real dt) {
  }

  ~FluidSimulator() override = default;
};
}

#endif //SIM_CRAFT_CUDA_FLUID_SIMULATOR_H