//
// Created by creeper on 10/37/33.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_PROJECT_SOLVER_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_PROJECT_SOLVER_H_
#include <Spatify/arrays.h>
#include <Spatify/grids.h>
#include <FluidSim/common/fluid-sim.h>
#include <FluidSim/common/sdf.h>
#include <array>
#include <memory>

namespace fluid {
using spatify::Array3D;
class CompressedSolver3D;

class ProjectionSolver2D {
  public:
    ProjectionSolver2D() = default;
    ProjectionSolver2D(int n, int w, int h)
      : n_particles(n), width(w), height(h) {
    }
    virtual void buildSystem(const spatify::GhostArray2D<Marker, 1>& markers,
                             const FaceCentredGrid<Real, Real, 2, 0>* u_grid,
                             const FaceCentredGrid<Real, Real, 2, 1>* v_grid,
                             const SDF<2>& collider_sdf,
                             Real density, Real dt) = 0;
    virtual Real solve(CellCentredGrid<Real, Real, 2>* p_grid) = 0;
    virtual ~ProjectionSolver2D() = default;

  protected:
    int n_particles = 0, width = 0, height = 0;
};

class CompressedSolver2D;
class FvmSolver2D : public ProjectionSolver2D {
  public:
    void buildSystem(const spatify::GhostArray2D<Marker, 1>& markers,
                     const FaceCentredGrid<Real, Real, 2, 0>* ug,
                     const FaceCentredGrid<Real, Real, 2, 1>* vg,
                     const SDF<2>& collider_sdf,
                     Real density, Real dt) override;
    void setCompressedSolver(CompressedSolver2D* solver) {
      this->solver = solver;
    }
    Real solve(CellCentredGrid<Real, Real, 2>* pg) override;

  protected:
    spatify::Array2D<Real> Adiag, rhs;
    std::array<spatify::Array2D<Real>, 4> Aneighbour;
    CompressedSolver2D* solver{};
    std::vector<Vec2i> active_cells;
};

class CompressedSolver2D {
  public:
    virtual Real solve(const spatify::Array2D<Real>& Adiag,
                       const std::array<spatify::Array2D<Real>, 4>& Aneighbour,
                       const spatify::Array2D<Real>& rhs,
                       CellCentredGrid<Real, Real, 2>* pg) = 0;
    virtual ~CompressedSolver2D() = default;
};

class Preconditioner2D {
  public:
    virtual void precond() = 0;
    virtual ~Preconditioner2D() = default;
};

class CgSolver2D : public CompressedSolver2D {
  public:
    Real solve(const spatify::Array2D<Real>& Adiag,
               const std::array<spatify::Array2D<Real>, 4>& Aneighbour,
               const spatify::Array2D<Real>& rhs,
               CellCentredGrid<Real, Real, 2>* pg) override;

  protected:
    std::vector<Real> z;
    std::unique_ptr<Preconditioner2D> precond = {};
};

class ProjectionSolver3D {
  public:
    ProjectionSolver3D() = default;
    ProjectionSolver3D(int w, int h, int d)
      : width(w), height(h), depth(d) {
    }
    virtual void buildSystem(const spatify::GhostArray3D<Marker, 1>& markers,
                             const FaceCentredGrid<Real, Real, 3, 0>* ug,
                             const FaceCentredGrid<Real, Real, 3, 1>* vg,
                             const FaceCentredGrid<Real, Real, 3, 2>* wg,
                             const SDF<3>* fluid_sdf,
                             const SDF<3>& collider_sdf,
                             Real density, Real dt) = 0;
    virtual Real solvePressure(Array3D<Real>& pg) = 0;
    virtual void project(const spatify::GhostArray3D<Marker, 1>& markers,
                         FaceCentredGrid<Real, Real, 3, 0>* ug,
                         FaceCentredGrid<Real, Real, 3, 1>* vg,
                         FaceCentredGrid<Real, Real, 3, 2>* wg,
                         const SDF<3>* fluid_sdf,
                         const SDF<3>& collider_sdf,
                         Real density, Real dt) = 0;

    virtual ~ProjectionSolver3D() = default;

  protected:
    int width = 0, height = 0, depth = 0;
};

class CompressedSolver3D {
  public:
    CompressedSolver3D(int m, int h, int d)
      : r(m, h, d) {
    }
    virtual Real solve(const Array3D<Real>& Adiag,
                       const std::array<Array3D<Real>, 6>& Aneighbour,
                       const Array3D<Real>& rhs,
                       Array3D<Real>& pg) = 0;
    virtual ~CompressedSolver3D() = default;

  protected:
    Array3D<Real> r;
};

class Preconditioner3D {
  public:
    virtual void precond(const Array3D<Real>& Adiag,
                         const std::array<Array3D<Real>, 6>& Aneighbour,
                         const Array3D<Real>& r,
                         Array3D<Real>& z) = 0;
    virtual ~Preconditioner3D() = default;
};

class ModifiedIncompleteCholesky3D : public Preconditioner3D {
  public:
    ModifiedIncompleteCholesky3D(int w, int h, int d, Real tuning = 0.97,
                                 Real safety = 0.25)
      : E(w, h, d), q(w, h, d), tau(tuning), sigma(safety) {
    }
    void precond(const Array3D<Real>& Adiag,
                 const std::array<Array3D<Real>, 6>& Aneighbour,
                 const Array3D<Real>& r,
                 Array3D<Real>& z) override;

  protected:
    Array3D<Real> E;
    Array3D<Real> q;
    Real tau, sigma;
};

class CgSolver3D : public CompressedSolver3D {
  public:
    CgSolver3D(int w, int h, int d)
      : CompressedSolver3D(w, h, d), z(w, h, d),
        s(w, h, d) {
    }
    Real solve(const Array3D<Real>& Adiag,
               const std::array<Array3D<Real>, 6>& Aneighbour,
               const Array3D<Real>& rhs,
               Array3D<Real>& pg) override;
    void setPreconditioner(Preconditioner3D* precond) {
      this->preconditioner = precond;
    }
    ~CgSolver3D() override = default;

  protected:
    Array3D<Real> z;
    Array3D<Real> s;
    Preconditioner3D* preconditioner = {};
    Real tolerance = 1e-6;
    int max_iterations = 100;
};

class FvmSolver3D : public ProjectionSolver3D {
  public:
    FvmSolver3D(int w, int h, int d)
      : Adiag(w, h, d), uWeights(w + 1, h, d), vWeights(w, h + 1, d),
        wWeights(w, h, d + 1), rhs(w, h, d),
        Aneighbour{Array3D<Real>(w, h, d), Array3D<Real>(w, h, d),
                   Array3D<Real>(w, h, d), Array3D<Real>(w, h, d),
                   Array3D<Real>(w, h, d), Array3D<Real>(w, h, d),},
        active_cells(w * h * d) {
    }
    void buildSystem(const spatify::GhostArray3D<Marker, 1>& markers,
                     const FaceCentredGrid<Real, Real, 3, 0>* ug,
                     const FaceCentredGrid<Real, Real, 3, 1>* vg,
                     const FaceCentredGrid<Real, Real, 3, 2>* wg,
                     const SDF<3>* fluid_sdf,
                     const SDF<3>& collider_sdf,
                     Real density, Real dt) override;
    Real solvePressure(Array3D<Real>& pg) override {
      return solver->solve(Adiag, Aneighbour, rhs, pg);
    }
    void project(const spatify::GhostArray3D<Marker, 1>& markers,
                 FaceCentredGrid<Real, Real, 3, 0>* ug,
                 FaceCentredGrid<Real, Real, 3, 1>* vg,
                 FaceCentredGrid<Real, Real, 3, 2>* wg,
                 const SDF<3>* fluid_sdf,
                 const SDF<3>& collider_sdf,
                 Real density, Real dt) override;
    void setCompressedSolver(CompressedSolver3D* solver) {
      this->solver = solver;
    }
    ~FvmSolver3D() override = default;

  protected:
    Array3D<Real> Adiag, rhs, uWeights, vWeights, wWeights;
    std::array<Array3D<Real>, 6> Aneighbour;
    CompressedSolver3D* solver{};
    std::vector<Vec3i> active_cells;
};
}
#endif //SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_PROJECT_SOLVER_H_