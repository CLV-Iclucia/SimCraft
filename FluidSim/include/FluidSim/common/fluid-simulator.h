//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
#include <Core/animation.h>
#include <Core/timer.h>
#include <Core/rand-gen.h>
#include <Spatify/grids.h>
#include <Spatify/arrays.h>
#include <FluidSim/common/sdf.h>
#include <FluidSim/common/fluid-sim.h>
#include <FluidSim/common/advect-solver.h>
#include <FluidSim/common/project-solver.h>
#include <memory>

enum class AdvectionOption {
};

namespace fluid {
class ProjectionSolver2D;
class HybridAdvectionSolver2D;

class HybridFluidSimulator2D : public core::Animation {
  public:
    core::Timer timer;
    ~HybridFluidSimulator2D() override = default;
    void init(int n, const Vec2d& size, const Vec2i& resolution);
    void initSolver(const Vec2i& resolution);
    void step(core::Frame& frame) override;
    [[nodiscard]] const std::vector<Vec2d>& positions() const {
      return m_particles.positions;
    }
    std::vector<Vec2d>& positions() { return m_particles.positions; }

  private:
    int nParticles;
    struct ParticleList {
      std::vector<Vec2d> positions;
      std::vector<Vec2d> velocities;
    } m_particles;
    [[nodiscard]] const Vec2d& pos(int i) const {
      return m_particles.positions[i];
    }
    [[nodiscard]] const Vec2d& vel(int i) const {
      return m_particles.velocities[i];
    }
    Vec2d& pos(int i) { return m_particles.positions[i]; }
    Vec2d& vel(int i) { return m_particles.velocities[i]; }
    Real density = 1e3;
    template <typename Func>
    void forActiveCells(Func&& func) {
      for (const auto& idx : active_cells) {
        int x = idx.x, y = idx.y;
        func(x, y);
      }
    }
    template <typename Func>
    void forActiveCells(Func&& func) const {
      for (const auto& idx : active_cells) {
        int x = idx.x, y = idx.y;
        func(x, y);
      }
    }
    Real pressure(int x, int y) {
      return markers(x, y) == Marker::Fluid ? p_grid(x, y) : 0.0;
    }
    void clear();
    void applyForce(Real dt);
    Real CFL();
    void substep(Real dt);
    void markCell();
    // when do we use unique_ptr: for those grids that needs to be swapped with buffers
    // and for polymorphic solvers
    std::unique_ptr<FaceCentredGrid<Real, Real, 2, 0>> u_grid, u_buf_grid;
    std::unique_ptr<FaceCentredGrid<Real, Real, 2, 1>> v_grid, v_buf_grid;
    CellCentredGrid<Real, Real, 2> p_grid;
    CellCentredGrid<Real, Real, 2> collider_sdf;
    spatify::Array2D<Real> Adiag, Aneighbour[4], rhs;
    std::unique_ptr<ProjectionSolver2D> projector;
    std::unique_ptr<HybridAdvectionSolver2D> advector;
    std::vector<Vec2i> active_cells;
    spatify::GhostArray2D<Marker, 1> markers;
};

class HybridFluidSimulator3D : public core::Animation {
  public:
    core::Timer timer;
    ~HybridFluidSimulator3D() override = default;
    HybridFluidSimulator3D(int n, const Vec3d& size, const Vec3i& resolution,
                           Real density)
      : nParticles(n), density(density), pg(resolution),
        colliderSdf(resolution, size), markers(resolution) {
      m_particles.positions.resize(n);
      // compute grid spacing
      Vec3d gridSpacing = size / Vec3d(resolution);
      // initialize grids
      ug = std::make_unique<FaceCentredGrid<Real, Real, 3, 0>>(
          resolution + Vec3i(1, 0, 0), Vec3d(gridSpacing));
      vg = std::make_unique<FaceCentredGrid<Real, Real, 3, 1>>(
          resolution + Vec3i(0, 1, 0), Vec3d(gridSpacing));
      wg = std::make_unique<FaceCentredGrid<Real, Real, 3, 2>>(
          resolution + Vec3i(0, 0, 1), Vec3d(gridSpacing));
      ubuf = std::make_unique<FaceCentredGrid<Real, Real, 3, 0>>(
          resolution + Vec3i(1, 0, 0), Vec3d(gridSpacing));
      vbuf = std::make_unique<FaceCentredGrid<Real, Real, 3, 1>>(
          resolution + Vec3i(0, 1, 0), Vec3d(gridSpacing));
      wbuf = std::make_unique<FaceCentredGrid<Real, Real, 3, 2>>(
          resolution + Vec3i(0, 0, 1), Vec3d(gridSpacing));
      // init u/v/wValid and u/v/wValidBuf
      uValid = std::make_unique<Array3D<char>>(resolution + Vec3i(1, 0, 0));
      vValid = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 1, 0));
      wValid = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 0, 1));
      uValidBuf = std::make_unique<Array3D<char>>(resolution + Vec3i(1, 0, 0));
      vValidBuf = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 1, 0));
      wValidBuf = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 0, 1));
      fluidSurface = std::make_unique<SDF<3>>(resolution, size);
      fluidSurfaceBuf = std::make_unique<SDF<3>>(resolution, size);
      for (auto& p : m_particles.positions)
        p = core::randomVec<Real, 3>() * 0.25 + Vec3d(0.25, 0.25, 0.25);
    }
    void buildCollider(const core::Mesh& colliderMesh) {
      std::cout << "Building collider SDF..." << std::endl;
      Array3D<int> closest(colliderSdf.width(), colliderSdf.height(),
                           colliderSdf.depth());
      Array3D<int> intersection_cnt(colliderSdf.width(), colliderSdf.height(),
                                    colliderSdf.depth());
      manifold2SDF(3, closest, intersection_cnt, colliderMesh,
                   &colliderSdf);
      std::cout << "Done." << std::endl;
    }
    void setAdvector(HybridAdvectionSolver3D* advectionSolver) {
      advector = advectionSolver;
    }
    void setProjector(ProjectionSolver3D* projectionSolver) {
      projector = projectionSolver;
    }
    void setReconstructor(ParticleSystemReconstructor<Real, 3>* reconstructor) {
      fluidSurfaceReconstructor = reconstructor;
    }
    void step(core::Frame& frame) override;
    [[nodiscard]] const std::vector<Vec3d>& positions() const {
      return m_particles.positions;
    }
    std::vector<Vec3d>& positions() { return m_particles.positions; }

  private:
    int nParticles;
    struct ParticleList {
      std::vector<Vec3d> positions;
    } m_particles;
    template <typename GridType>
    void extrapolate(std::unique_ptr<GridType>& g,
                     std::unique_ptr<GridType>& gbuf,
                     std::unique_ptr<Array3D<char>>& valid,
                     std::unique_ptr<Array3D<char>>& validBuf, int iters) {
      for (int i = 0; i < iters; i++) {
        g->forEach([&](int i, int j, int k) {
          if (valid->at(i, j, k)) return;
          Real sum{0.0};
          int count{0};
          if (i > 0 && valid->at(i - 1, j, k)) {
            sum += g->at(i - 1, j, k);
            count++;
          }
          if (i < g->width() - 1 && valid->at(i + 1, j, k)) {
            sum += g->at(i + 1, j, k);
            count++;
          }
          if (j > 0 && valid->at(i, j - 1, k)) {
            sum += g->at(i, j - 1, k);
            count++;
          }
          if (j < g->height() - 1 && valid->at(i, j + 1, k)) {
            sum += g->at(i, j + 1, k);
            count++;
          }
          if (k > 0 && valid->at(i, j, k - 1)) {
            sum += g->at(i, j, k - 1);
            count++;
          }
          if (k < g->depth() - 1 && valid->at(i, j, k + 1)) {
            sum += g->at(i, j, k + 1);
            count++;
          }
          gbuf->at(i, j, k) = sum / count;
          if (valid->at(i, j, k) && count > 0) validBuf->at(i, j, k) = 1;
        });
        std::swap(g, gbuf);
        std::swap(valid, validBuf);
      }
    }
    Real density = 1e3;
    Real pressure(int x, int y, int z) {
      return markers(x, y, z) == Marker::Fluid ? pg(x, y, z) : 0.0;
    }
    void clear();
    void applyForce(Real dt) const;
    void applyCollider() const;
    [[nodiscard]] Real CFL() const;
    void substep(Real dt);
    void markCell();
    void smoothFluidSurface(int iters);
    std::unique_ptr<FaceCentredGrid<Real, Real, 3, 0>> ug, ubuf;
    std::unique_ptr<FaceCentredGrid<Real, Real, 3, 1>> vg, vbuf;
    std::unique_ptr<FaceCentredGrid<Real, Real, 3, 2>> wg, wbuf;
    std::unique_ptr<Array3D<char>> uValid, vValid, wValid, uValidBuf, vValidBuf,
        wValidBuf;
    Array3D<Real> pg;
    SDF<3> colliderSdf;
    ParticleSystemReconstructor<Real, 3>* fluidSurfaceReconstructor{};
    std::unique_ptr<SDF<3>> fluidSurface{}, fluidSurfaceBuf{};
    ProjectionSolver3D* projector{};
    HybridAdvectionSolver3D* advector{};
    spatify::GhostArray3D<Marker, 1> markers;
};
} // namespace fluid
#endif  // SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_