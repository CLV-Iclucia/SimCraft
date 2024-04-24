//
// Created by creeper on 24-3-26.
//

#ifndef SIM_CRAFT_FLUIDSIM_CUDA_PROJECT_SOLVER_H
#define SIM_CRAFT_FLUIDSIM_CUDA_PROJECT_SOLVER_H
#include <memory>
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <Core/cuda-utils.h>
#include <array>

namespace fluid::cuda {
enum Directions {
  Left,
  Right,
  Up,
  Down,
  Front,
  Back
};
class ProjectionSolver {
 public:
  ProjectionSolver() = default;

  ProjectionSolver(int w, int h, int d)
      : width(w), height(h), depth(d) {
  }

  virtual void buildSystem(
      const CudaSurface<float> &ug,
      const CudaSurface<float> &vg,
      const CudaSurface<float> &wg,
      const CudaSurface<float> &fluidSdf,
      const CudaTexture<float> &colliderSdf,
      int3 resolution,
      float h,
      float dt) = 0;
  virtual float solvePressure(const CudaSurface<float> &fluidSdf,
                             CudaSurface<float> &p,
                             int3 resolution,
                             float h,
                             float dt) = 0;
  virtual void project(const CudaSurface<float> &ug,
                       const CudaSurface<float> &vg,
                       const CudaSurface<float> &wg,
                       const CudaSurface<float> &fluidSdf,
                       const CudaTexture<float> &colliderSdf,
                       CudaSurface<float> &pg,
                       int3 resolution,
                       float h,
                       float dt) = 0;
//  virtual void setCompressedSolver(CompressedSolverMethod solver_method,
//                                   PreconditionerMethod precond_method) = 0;
  virtual ~ProjectionSolver() = default;

 protected:
  int width = 0, height = 0, depth = 0;
};

struct CompressedSystem;
}
namespace core {
template<>
struct Accessor<fluid::cuda::CompressedSystem> {
  fluid::cuda::CudaSurfaceAccessor<float> diag{};
  fluid::cuda::CudaSurfaceAccessor<float> neighbour[6] = {{}, {}, {}, {}, {}, {}};
  fluid::cuda::CudaSurfaceAccessor<float> rhs{};
};

template<>
struct ConstAccessor<fluid::cuda::CompressedSystem> {
  fluid::cuda::CudaSurfaceAccessor<float> diag{};
  fluid::cuda::CudaSurfaceAccessor<float> neighbour[6] = {{}, {}, {}, {}, {}, {}};
  fluid::cuda::CudaSurfaceAccessor<float> rhs{};
};
}
namespace fluid::cuda {
struct CompressedSystem : core::DeviceMemoryAccessible<CompressedSystem> {
  std::unique_ptr<CudaSurface<float>> diag{};
  std::array<std::unique_ptr<CudaSurface<float>>, 6> neighbour{};
  std::unique_ptr<CudaSurface<float>> rhs{};
  CompressedSystem(int w, int h, int d) {
    diag = std::make_unique<CudaSurface<float>>(make_uint3(w, h, d));
    rhs = std::make_unique<CudaSurface<float>>(make_uint3(w, h, d));
    for (int i = 0; i < 6; ++i)
      neighbour[i] = std::make_unique<CudaSurface<float>>(make_uint3(w, h, d));
  }

  Accessor<CompressedSystem> accessor() {
    return {diag->surfaceAccessor(),
            {neighbour[0]->surfaceAccessor(),
             neighbour[1]->surfaceAccessor(),
             neighbour[2]->surfaceAccessor(),
             neighbour[3]->surfaceAccessor(),
             neighbour[4]->surfaceAccessor(),
             neighbour[5]->surfaceAccessor()},
            rhs->surfaceAccessor()};
  }
  [[nodiscard]] ConstAccessor<CompressedSystem> constAccessor() const {
    return {diag->surfaceAccessor(),
            {neighbour[0]->surfaceAccessor(),
             neighbour[1]->surfaceAccessor(),
             neighbour[2]->surfaceAccessor(),
             neighbour[3]->surfaceAccessor(),
             neighbour[4]->surfaceAccessor(),
             neighbour[5]->surfaceAccessor()},
            rhs->surfaceAccessor()};
  }
};

class CompressedSolver {
 public:
  CompressedSolver(int m, int h, int d) {
    r = std::make_unique<CudaSurface<float>>(make_uint3(m, h, d));
  }

  virtual float solve(const CompressedSystem &sys,
                     const CudaSurface<uint8_t> &active,
                     CudaSurface<float> &pg,
                     int3 resolution)= 0;
//  virtual void setPreconditioner(PreconditionerMethod precond_method) = 0;
  virtual ~CompressedSolver() = default;

 protected:
  std::unique_ptr<CudaSurface<float>> r;
};

class Preconditioner {
 public:
  virtual void precond(
      const CompressedSystem &sys,
      const CudaSurface<float> &r,
      const CudaSurface<uint8_t> &active,
      const CudaSurface<float> &z) = 0;
  virtual ~Preconditioner() = default;
};

class CgSolver final : public CompressedSolver {
 public:
  CgSolver(int w, int h, int d) : CompressedSolver(w, h, d) {
    z = std::make_unique<CudaSurface<float>>(make_uint3(w, h, d));
    s = std::make_unique<CudaSurface<float>>(make_uint3(w, h, d));
    device_reduce_buffer = std::make_unique<DeviceArray<float>>(
        (w + kThreadBlockSize3D - 1) / kThreadBlockSize3D *
            (h + kThreadBlockSize3D - 1) / kThreadBlockSize3D *
            (d + kThreadBlockSize3D - 1) / kThreadBlockSize3D);
    host_reduce_buffer.resize(device_reduce_buffer->size());
  }

  float solve(const CompressedSystem &sys,
             const CudaSurface<uint8_t> &active,
             CudaSurface<float> &pg,
             int3 resolution) override;

//  void setPreconditioner(PreconditionerMethod precond_method) override {
//    if (precond_method == PreconditionerMethod::ModifiedIncompleteCholesky) {
//      ERROR("Not supported by GPU!");
//    } else if (precond_method == PreconditionerMethod::Multigrid) {
//      ERROR("Not supported yet!");
//    } else
//      ERROR("Unknown preconditioner type");
//  }

  ~CgSolver() override = default;

 protected:
  float L1Norm(const CudaSurface<float> &surface,
              const CudaSurface<uint8_t> &active,
              int3 resolution) const;
  float dotProduct(const CudaSurface<float> &surfaceA,
                  const CudaSurface<float> &surfaceB,
                  const CudaSurface<uint8_t> &active,
                  int3 resolution) const;
  std::unique_ptr<CudaSurface<float>> z{}, s{};
  std::unique_ptr<Preconditioner> preconditioner{};
  std::unique_ptr<DeviceArray<float>> device_reduce_buffer{};
  mutable std::vector<float> host_reduce_buffer{};
  float tolerance = 1e-6;
  int max_iterations = 300;
};

class FvmSolver final : public ProjectionSolver {
 public:
  FvmSolver(int w, int h, int d) {
    sys = std::make_unique<CompressedSystem>(w, h, d);
    uWeights = std::make_unique<CudaSurface<float>>(make_uint3(w, h, d));
    vWeights = std::make_unique<CudaSurface<float>>(make_uint3(w, h, d));
    wWeights = std::make_unique<CudaSurface<float>>(make_uint3(w, h, d));
    active = std::make_unique<CudaSurface<uint8_t>>(make_uint3(w, h, d));
  }

  void buildSystem(
      const CudaSurface<float> &u,
      const CudaSurface<float> &v,
      const CudaSurface<float> &w,
      const CudaSurface<float> &fluidSdf,
      const CudaTexture<float> &colliderSdf,
      int3 resolution,
      float h,
      float dt) override;

  float solvePressure(const CudaSurface<float> &fluidSdf,
                     CudaSurface<float> &pg,
                     int3 resolution,
                     float h,
                     float dt) override {
    return solver->solve(*sys, *active, pg, resolution);
  }

  void project(const CudaSurface<float> &ug,
               const CudaSurface<float> &vg,
               const CudaSurface<float> &wg,
               const CudaSurface<float> &fluidSdf,
               const CudaTexture<float> &colliderSdf,
               CudaSurface<float> &pg,
               int3 resolution,
               float h,
               float dt) override {
    buildSystem(ug, vg, wg, fluidSdf, colliderSdf, resolution, h, dt);
    float residual = solvePressure(fluidSdf, pg, resolution, h, dt);
//    if (residual > 1e-6)
//      ERROR("Pressure solve did not converge");
  }

//  void setCompressedSolver(CompressedSolverMethod solver_method,
//                           PreconditionerMethod precond_method) override {
//    if (solver_method == CompressedSolverMethod::CG) {
//      auto cg_solver = std::make_unique<CgSolver>(width, height, depth);
//      cg_solver->setPreconditioner(precond_method);
//      solver = std::move(cg_solver);
//    } else
//      ERROR("Unknown solver type");
//  }

  ~FvmSolver() override = default;

 protected:
  std::unique_ptr<CudaSurface<float>> uWeights{}, vWeights{}, wWeights{};
  std::unique_ptr<CompressedSystem> sys{};
  std::unique_ptr<CompressedSolver> solver{};
  std::unique_ptr<CudaSurface<uint8_t>> active{};
};

}

#endif //SIM_CRAFT_FLUIDSIM_CUDA_PROJECT_SOLVER_H