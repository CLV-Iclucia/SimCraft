//
// Created by creeper on 24-3-26.
//

#ifndef SIM_CRAFT_FLUIDSIM_CUDA_PROJECT_SOLVER_H
#define SIM_CRAFT_FLUIDSIM_CUDA_PROJECT_SOLVER_H
#include <memory>
#include <FluidSim/cuda/fluid-simulator.h>
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
      const CudaSurface<Real> &ug,
      const CudaSurface<Real> &vg,
      const CudaSurface<Real> &wg,
      const CudaSurface<Real> &fluidSdf,
      const CudaSurface<Real> &colliderSdf,
      int3 resolution,
      Real h,
      Real dt) = 0;
  virtual Real solvePressure(const CudaSurface<Real> &fluidSdf,
                             CudaSurface<Real> &p,
                             Real dt) = 0;
  virtual void project(const CudaSurface<Real> &ug,
                       const CudaSurface<Real> &vg,
                       const CudaSurface<Real> &wg,
                       const CudaSurface<Real> &fluidSdf,
                       const CudaSurface<Real> &colliderSdf,
                       CudaSurface<Real> &pg,
                       int3 resolution,
                       Real h,
                       Real dt) = 0;
  virtual void setCompressedSolver(CompressedSolverMethod solver_method,
                                   PreconditionerMethod precond_method) = 0;
  virtual ~ProjectionSolver() = default;

 protected:
  int width = 0, height = 0, depth = 0;
};

struct CompressedSystem;
}
namespace core {
template<>
struct Accessor<fluid::cuda::CompressedSystem> {
  fluid::cuda::CudaSurfaceAccessor<Real> diag{};
  fluid::cuda::CudaSurfaceAccessor<Real> neighbour[6] = {{}, {}, {}, {}, {}, {}};
  fluid::cuda::CudaSurfaceAccessor<Real> rhs{};
};

template<>
struct ConstAccessor<fluid::cuda::CompressedSystem> {
  fluid::cuda::CudaSurfaceAccessor<Real> diag{};
  fluid::cuda::CudaSurfaceAccessor<Real> neighbour[6] = {{}, {}, {}, {}, {}, {}};
  fluid::cuda::CudaSurfaceAccessor<Real> rhs{};
};
}
namespace fluid::cuda {
struct CompressedSystem : core::DeviceMemoryAccessible<CompressedSystem> {
  std::unique_ptr<CudaSurface<Real>> diag{};
  std::array<std::unique_ptr<CudaSurface<Real>>, 6> neighbour{};
  std::unique_ptr<CudaSurface<Real>> rhs{};
  CompressedSystem(int w, int h, int d) {
    diag = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    rhs = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    for (int i = 0; i < 6; ++i)
      neighbour[i] = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
  }

  Accessor<CompressedSystem> accessor() {
    return {diag->surfAccessor(),
            {neighbour[0]->surfAccessor(),
             neighbour[1]->surfAccessor(),
             neighbour[2]->surfAccessor(),
             neighbour[3]->surfAccessor(),
             neighbour[4]->surfAccessor(),
             neighbour[5]->surfAccessor()},
            rhs->surfAccessor()};
  }
  [[nodiscard]] ConstAccessor<CompressedSystem> constAccessor() const {
    return {diag->surfAccessor(),
            {neighbour[0]->surfAccessor(),
             neighbour[1]->surfAccessor(),
             neighbour[2]->surfAccessor(),
             neighbour[3]->surfAccessor(),
             neighbour[4]->surfAccessor(),
             neighbour[5]->surfAccessor()},
            rhs->surfAccessor()};
  }
};

class CompressedSolver {
 public:
  CompressedSolver(int m, int h, int d) {
    r = std::make_unique<CudaSurface<Real>>(make_uint3(m, h, d));
  }

  virtual Real solve(const CompressedSystem &sys,
                     const CudaSurface<uint8_t> &active,
                     CudaSurface<Real> &pg) = 0;
  virtual void setPreconditioner(PreconditionerMethod precond_method) = 0;
  virtual ~CompressedSolver() = default;

 protected:
  std::unique_ptr<CudaSurface<Real>> r;
};

class Preconditioner {
 public:
  virtual void precond(
      const CompressedSystem &sys,
      const CudaSurface<Real> &r,
      const CudaSurface<uint8_t> &active,
      const CudaSurface<Real> &z) = 0;
  virtual ~Preconditioner() = default;
};

class CgSolver final : public CompressedSolver {
 public:
  CgSolver(int w, int h, int d)
      : CompressedSolver(w, h, d) {
    z = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    s = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    device_reduce_buffer = std::make_unique<DeviceArray<Real>>(
        (w + kThreadBlockSize3D - 1) / kThreadBlockSize3D *
            (h + kThreadBlockSize3D - 1) / kThreadBlockSize3D *
            (d + kThreadBlockSize3D - 1) / kThreadBlockSize3D);
    host_reduce_buffer.resize(device_reduce_buffer->size());
  }

  Real solve(const CompressedSystem &sys,
             const CudaSurface<uint8_t> &active,
             CudaSurface<Real> &pg) override;

  void setPreconditioner(PreconditionerMethod precond_method) override {
    if (precond_method == PreconditionerMethod::ModifiedIncompleteCholesky) {
      ERROR("Not supported by GPU!");
    } else if (precond_method == PreconditionerMethod::Multigrid) {
      ERROR("Not supported yet!");
    } else
      ERROR("Unknown preconditioner type");
  }

  ~CgSolver() override = default;

 protected:
  Real L1Norm(const CudaSurface<Real> &surface,
              const CudaSurface<uint8_t> &active,
              int3 resolution) const;
  Real dotProduct(const CudaSurface<Real> &surfaceA,
                  const CudaSurface<Real> &surfaceB,
                  const CudaSurface<uint8_t> &active,
                  int3 resolution) const;
  std::unique_ptr<CudaSurface<Real>> z{}, s{};
  std::unique_ptr<Preconditioner> preconditioner{};
  std::unique_ptr<DeviceArray<Real>> device_reduce_buffer{};
  mutable std::vector<Real> host_reduce_buffer{};
  Real tolerance = 1e-6;
  int max_iterations = 300;
  Real solve(const CompressedSystem &sys, const CudaSurface <uint8_t> &active, CudaSurface <Real> &pg, int3 resolution);
};

class FvmSolver final : public ProjectionSolver {
 public:
  FvmSolver(int w, int h, int d) {
    sys = std::make_unique<CompressedSystem>(w, h, d);
    uWeights = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    vWeights = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    wWeights = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    active = std::make_unique<CudaSurface<uint8_t>>(make_uint3(w, h, d));
  }

  void buildSystem(
      const CudaSurface<Real> &u,
      const CudaSurface<Real> &v,
      const CudaSurface<Real> &w,
      const CudaSurface<Real> &fluidSdf,
      const CudaSurface<Real> &colliderSdf,
      int3 resolution,
      Real h,
      Real dt) override;

  Real solvePressure(const CudaSurface<Real> &fluidSdf,
                     CudaSurface<Real> &pg,
                     Real dt) override {
    return solver->solve(*sys, *active, pg);
  }

  void project(const CudaSurface<Real> &ug,
               const CudaSurface<Real> &vg,
               const CudaSurface<Real> &wg,
               const CudaSurface<Real> &fluidSdf,
               const CudaSurface<Real> &colliderSdf,
               CudaSurface<Real> &pg,
               int3 resolution,
               Real h,
               Real dt) override {
    buildSystem(ug, vg, wg, fluidSdf, colliderSdf, resolution, h, dt);
    Real residual = solvePressure(fluidSdf, pg, dt);
    if (residual > 1e-6)
      ERROR("Pressure solve did not converge");
  }

  void setCompressedSolver(CompressedSolverMethod solver_method,
                           PreconditionerMethod precond_method) override {
    if (solver_method == CompressedSolverMethod::CG) {
      auto cg_solver = std::make_unique<CgSolver>(width, height, depth);
      cg_solver->setPreconditioner(precond_method);
      solver = std::move(cg_solver);
    } else
      ERROR("Unknown solver type");
  }

  ~FvmSolver() override = default;

 protected:
  std::unique_ptr<CudaSurface<Real>> uWeights{}, vWeights{}, wWeights{};
  std::unique_ptr<CompressedSystem> sys{};
  std::unique_ptr<CompressedSolver> solver{};
  std::unique_ptr<CudaSurface<uint8_t>> active{};
};

}

#endif //SIM_CRAFT_FLUIDSIM_CUDA_PROJECT_SOLVER_H