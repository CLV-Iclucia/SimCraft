//
// Created by creeper on 24-3-26.
//

#ifndef SIM_CRAFT_FLUIDSIM_CUDA_PROJECT_SOLVER_H
#define SIM_CRAFT_FLUIDSIM_CUDA_PROJECT_SOLVER_H
#include <memory>
#include <FluidSim/cuda/fluid-simulator.h>
#include <array>

namespace fluid::cuda {
class ProjectionSolver {
public:
  ProjectionSolver() = default;

  ProjectionSolver(int w, int h, int d)
    : width(w), height(h), depth(d) {
  }

  virtual void buildSystem(
      CudaSurfaceAccessor<Real> ug,
      CudaSurfaceAccessor<Real> vg,
      CudaSurfaceAccessor<Real> wg,
      CudaSurfaceAccessor<Real> fluidSdf,
      CudaSurfaceAccessor<Real> colliderSdf,
      Real h,
      Real dt) = 0;
  virtual Real solvePressure(CudaSurfaceAccessor<Real> fluidSdf,
                             CudaSurfaceAccessor<Real> p,
                             Real dt) = 0;
  virtual void project(CudaSurfaceAccessor<Real> ug,
                       CudaSurfaceAccessor<Real> vg,
                       CudaSurfaceAccessor<Real> wg,
                       CudaSurfaceAccessor<Real> pg,
                       CudaSurfaceAccessor<Real> fluidSdf,
                       CudaSurfaceAccessor<Real> colliderSdf,
                       Real h,
                       Real dt) = 0;
  virtual void setCompressedSolver(CompressedSolverMethod solver_method,
                                   PreconditionerMethod precond_method) = 0;
  virtual ~ProjectionSolver() = default;

protected:
  int width = 0, height = 0, depth = 0;
};

class CompressedSolver {
public:
  CompressedSolver(int m, int h, int d) {
    r = std::make_unique<CudaSurface<Real>>(make_uint3(m, h, d));
  }

  virtual Real solve(CudaSurfaceAccessor<Real> Adiag,
                     const std::array<CudaSurfaceAccessor<Real>, 6>& Aneighbour,
                     CudaSurfaceAccessor<Real> rhs,
                     CudaSurfaceAccessor<uint8_t> active,
                     CudaSurfaceAccessor<Real> pg) = 0;
  virtual void setPreconditioner(PreconditionerMethod precond_method) = 0;
  virtual ~CompressedSolver() = default;

protected:
  std::unique_ptr<CudaSurface<Real>> r;
};

class Preconditioner {
public:
  virtual void precond(
      CudaSurfaceAccessor<Real> Adiag,
      const std::array<CudaSurfaceAccessor<Real>, 6>& Aneighbour,
      CudaSurfaceAccessor<Real> r,
      CudaSurfaceAccessor<uint8_t> active,
      CudaSurfaceAccessor<Real> z) = 0;
  virtual ~Preconditioner() = default;
};

class CgSolver final : public CompressedSolver {
public:
  CgSolver(int w, int h, int d)
    : CompressedSolver(w, h, d) {
    z = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    s = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
  }

  Real solve(CudaSurfaceAccessor<Real> Adiag,
             const std::array<CudaSurfaceAccessor<Real>, 6>& Aneighbour,
             CudaSurfaceAccessor<Real> rhs,
             CudaSurfaceAccessor<uint8_t> active,
             CudaSurfaceAccessor<Real> pg) override;

  void setPreconditioner(PreconditionerMethod precond_method) override {
    if (precond_method == PreconditionerMethod::ModifiedIncompleteCholesky) {
      ERROR("Not supported by GPU!");
    } else if (precond_method == PreconditionerMethod::Multigrid) {
      ERROR("Not supported yet!");
    } else
      ERROR("Unknown preconditioner type: {}", precond_name);
  }

  ~CgSolver() override = default;

protected:
  std::unique_ptr<CudaSurface<Real>> z{}, s{};
  std::unique_ptr<Preconditioner> preconditioner{};

  Real tolerance = 1e-6;
  int max_iterations = 300;
};

class FvmSolver final : public ProjectionSolver {
public:
  FvmSolver(int w, int h, int d) {
    Adiag = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    rhs = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    uWeights = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    vWeights = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    wWeights = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    for (int i = 0; i < 6; ++i)
      Aneighbour[i] = std::make_unique<CudaSurface<Real>>(make_uint3(w, h, d));
    active = std::make_unique<CudaSurface<uint8_t>>(make_uint3(w, h, d));
  }

  void buildSystem(
      CudaSurfaceAccessor<double> u,
      CudaSurfaceAccessor<double> v,
      CudaSurfaceAccessor<double> w,
      CudaSurfaceAccessor<double> fluidSdf,
      CudaSurfaceAccessor<double> colliderSdf,
      Real h,
      Real dt) override;

  Real solvePressure(CudaSurfaceAccessor<Real> fluidSdf,
                     CudaSurfaceAccessor<Real> pg,
                     Real dt) override {
    static auto accessors = std::array{
        Aneighbour[0]->surfAccessor(),
        Aneighbour[1]->surfAccessor(),
        Aneighbour[2]->surfAccessor(),
        Aneighbour[3]->surfAccessor(),
        Aneighbour[4]->surfAccessor(),
        Aneighbour[5]->surfAccessor()};
    return solver->solve(Adiag->surfAccessor(),
                         accessors, rhs->surfAccessor(), active->surfAccessor(),
                         pg);
  }

  void project(CudaSurfaceAccessor<Real> ug,
               CudaSurfaceAccessor<Real> vg,
               CudaSurfaceAccessor<Real> wg,
               CudaSurfaceAccessor<Real> pg,
               CudaSurfaceAccessor<Real> fluidSdf,
               CudaSurfaceAccessor<Real> colliderSdf,
               Real h,
               Real dt) override {
    buildSystem(ug, vg, wg, fluidSdf, colliderSdf, h, dt);
    Real residual = solvePressure(fluidSdf, pg, dt);
    if (residual > 1e-6)
      ERROR("Pressure solve did not converge: {}", residual);
  }

  void setCompressedSolver(CompressedSolverMethod solver_method,
                           PreconditionerMethod precond_method) override {
    if (solver_method == CompressedSolverMethod::CG) {
      auto cg_solver = std::make_unique<CgSolver>(width, height, depth);
      cg_solver->setPreconditioner(precond_method);
      solver = std::move(cg_solver);
    } else
      ERROR("Unknown solver type: {}", solver_name);
  }

  ~FvmSolver() override = default;

protected:
  void checkSystemSymmetry() const;
  std::unique_ptr<CudaSurface<Real>>
      Adiag{}, rhs{}, uWeights{}, vWeights{}, wWeights{};
  std::array<std::unique_ptr<CudaSurface<Real>>, 6> Aneighbour{};
  std::unique_ptr<CompressedSolver> solver{};
  std::unique_ptr<CudaSurface<uint8_t>> active{};
};
}

#endif //SIM_CRAFT_FLUIDSIM_CUDA_PROJECT_SOLVER_H