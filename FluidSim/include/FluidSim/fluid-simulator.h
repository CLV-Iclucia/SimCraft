//
// Created by creeper on 24-3-20.
//

#ifndef FLUID_SIMULATOR_H
#define FLUID_SIMULATOR_H

#include <FluidSim/fluid-sim.h>
#include <Core/debug.h>
#include <memory>
#include <utility>

namespace fluid {
enum class AdvectionMethod {
  PIC,
  FLIP,
  APIC
};

enum class ProjectSolver {
  FVM,
  FDM
};

enum class Backend {
  CPU,
  CUDA
};

enum class PreconditionerMethod {
  Multigrid,
  ModifiedIncompleteCholesky
};

enum class CompressedSolverMethod {
  CG
};

enum class ReconstructionMethod {
  Naive
};

struct FluidComputeBackend : NonCopyable {
  virtual void setCollider(const Mesh& collider_mesh) const = 0;
  virtual void setInitialFluid(const Mesh& fluid_mesh) = 0;
  virtual void setAdvector(AdvectionMethod advection_method) = 0;
  virtual void setProjector(ProjectSolver project_solver) = 0;
  virtual void setCompressedSolver(CompressedSolverMethod solver_method,
                                   PreconditionerMethod preconditioner_method) =
  0;
  virtual void setParticleReconstructor(ReconstructionMethod reconstruction_method) = 0;
  virtual ~FluidComputeBackend() = default;
};

struct Scene {
  Mesh collider_mesh;
  Mesh fluid_init_mesh;
};


struct SimConfig {
  int nParticles = 0;
  Vec3d size;
  Vec3d orig;
  Vec3i resolution;
};

struct FluidSimulator final : core::Animation {
  explicit FluidSimulator(SimConfig config) : config(std::move(config)) {
  }

  SimConfig config;
  std::unique_ptr<Scene> scene{};
  std::unique_ptr<FluidComputeBackend> backend{};

  void loadScene(const std::string& collider_path,
                 const std::string& fluid_path) {
    if (!scene)
      scene = std::make_unique<Scene>();
    if (!myLoadObj(collider_path, &scene->collider_mesh)) {
      throw std::runtime_error("Failed to load fluidRegion mesh");
    }
    if (!myLoadObj(fluid_path, &scene->fluid_init_mesh)) {
      throw std::runtime_error("Failed to load fluid mesh");
    }
  }

  void step(core::Frame& frame) override {
  }

  void setBackend(Backend backend);
};
}
#endif //FLUID_SIMULATOR_H