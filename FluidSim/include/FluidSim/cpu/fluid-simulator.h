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
#include <FluidSim/fluid-sim.h>
#include <FluidSim/cpu/sdf.h>
#include <FluidSim/cpu/advect-solver.h>
#include <FluidSim/cpu/project-solver.h>
#include <Spatify/grids.h>
#include <memory>

namespace fluid {
using spatify::Grid;
using spatify::CellCentredGrid;
using spatify::FaceCentredGrid;
using spatify::PaddedCellCentredGrid;

class HybridFluidSimulator3D final : public core::Animation {
  public:
    core::Timer timer;
    ~HybridFluidSimulator3D() override = default;
    HybridFluidSimulator3D(int n, const Vec3d& size, const Vec3i& resolution)
      : nParticles(n), uw(resolution + Vec3i(1, 0, 0)),
        vw(resolution + Vec3i(0, 1, 0)), ww(resolution + Vec3i(0, 0, 1)),
        pg(resolution) {
      m_particles.positions.resize(n);
      // initialize grids
      ug = std::make_unique<FaceCentredGrid<
        Real, Real, 3, 0>>(resolution, size);
      vg = std::make_unique<FaceCentredGrid<
        Real, Real, 3, 1>>(resolution, size);
      wg = std::make_unique<FaceCentredGrid<
        Real, Real, 3, 2>>(resolution, size);
      ubuf = std::make_unique<FaceCentredGrid<Real, Real, 3, 0>>(
          resolution, size);
      vbuf = std::make_unique<FaceCentredGrid<Real, Real, 3, 1>>(
          resolution, size);
      wbuf = std::make_unique<FaceCentredGrid<Real, Real, 3, 2>>(
          resolution, size);
      uValid = std::make_unique<Array3D<char>>(resolution + Vec3i(1, 0, 0));
      vValid = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 1, 0));
      wValid = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 0, 1));
      uValidBuf = std::make_unique<Array3D<char>>(resolution + Vec3i(1, 0, 0));
      vValidBuf = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 1, 0));
      wValidBuf = std::make_unique<Array3D<char>>(resolution + Vec3i(0, 0, 1));
      sdfValid = std::make_unique<Array3D<char>>(resolution);
      sdfValidBuf = std::make_unique<Array3D<char>>(resolution);
      fluidSurface = std::make_unique<SDF<3>>(resolution, size);
      fluidSurfaceBuf = std::make_unique<SDF<3>>(resolution, size);
      colliderSdf = std::make_unique<SDF<3>>(resolution, size);
      for (auto& p : m_particles.positions) {
        p = core::randomVec<Real, 3>() * Vec3d(1.0, 0.45, 1.0) + Vec3d(0.0, 0.5, 0.0);
        p *= size;
      }
      assert(
          ug->gridSpacing() == vg->gridSpacing() && vg->gridSpacing() == wg->
          gridSpacing());
    }
    void buildCollider(const core::Mesh& colliderMesh) const {
      std::cout << "Building collider SDF..." << std::endl;
      Array3D<int> closest(colliderSdf->width(), colliderSdf->height(),
                           colliderSdf->depth());
      Array3D<int> intersection_cnt(colliderSdf->width(), colliderSdf->height(),
                                    colliderSdf->depth());
      manifold2SDF(3, closest, intersection_cnt, colliderMesh,
                   colliderSdf.get());
      std::cout << "Done." << std::endl;
    }
    void setAdvector(HybridAdvectionSolver3D* advectionSolver) {
      advector = advectionSolver;
    }
    void setProjector(ProjectionSolver3D* projectionSolver) {
      projector = projectionSolver;
    }
    void reconstruct() {
      fluidSurfaceReconstructor->reconstruct(
          m_particles.positions, 1.2 * ug->gridSpacing().x / std::sqrt(2.0),
          *fluidSurface, *sdfValid);
    }
    void smoothFluidSurface(int iters);
    void extrapolateFluidSdf(int iters) {
      for (int i = 0; i < iters; i++) {
        sdfValidBuf->fill(false);
        fluidSurface->grid.forEach([&](int i, int j, int k) {
          if (sdfValid->at(i, j, k)) {
            sdfValidBuf->at(i, j, k) = true;
            return;
          }
          Real sum{0.0};
          int count{0};
          if (i > 0 && sdfValid->at(i - 1, j, k)) {
            sum += fluidSurface->grid(i - 1, j, k);
            count++;
          }
          if (i < fluidSurface->width() - 1 && sdfValid->at(i + 1, j, k)) {
            sum += fluidSurface->grid(i + 1, j, k);
            count++;
          }
          if (j > 0 && sdfValid->at(i, j - 1, k)) {
            sum += fluidSurface->grid(i, j - 1, k);
            count++;
          }
          if (j < fluidSurface->height() - 1 && sdfValid->at(i, j + 1, k)) {
            sum += fluidSurface->grid(i, j + 1, k);
            count++;
          }
          if (k > 0 && sdfValid->at(i, j, k - 1)) {
            sum += fluidSurface->grid(i, j, k - 1);
            count++;
          }
          if (k < fluidSurface->depth() - 1 && sdfValid->at(i, j, k + 1)) {
            sum += fluidSurface->grid(i, j, k + 1);
            count++;
          }
          if (count > 0) {
            fluidSurfaceBuf->grid(i, j, k) = sum / count;
            sdfValidBuf->at(i, j, k) = true;
          } else
            sdfValidBuf->at(i, j, k) = false;
        });
        std::swap(fluidSurface, fluidSurfaceBuf);
        std::swap(sdfValid, sdfValidBuf);
      }
    }
    void setReconstructor(ParticleSystemReconstructor<Real, 3>* reconstructor) {
      fluidSurfaceReconstructor = reconstructor;
    }
    void step(core::Frame& frame) override;
    [[nodiscard]] const std::vector<Vec3d>& positions() const {
      return m_particles.positions;
    }
    [[nodiscard]] const SDF<3>& exportFluidSurface() const {
      return *fluidSurface;
    }
    [[nodiscard]] const SDF<3>& exportColliderSurface() const {
      return *colliderSdf;
    }
    std::vector<Vec3d>& positions() { return m_particles.positions; }

  private:
    int nParticles{};
    struct ParticleList {
      std::vector<Vec3d> positions{};
    } m_particles;
    template <typename GridType>
    void extrapolate(std::unique_ptr<GridType>& g,
                     std::unique_ptr<GridType>& gbuf,
                     std::unique_ptr<Array3D<char>>& valid,
                     std::unique_ptr<Array3D<char>>& validBuf, int iters) {
      for (int i = 0; i < iters; i++) {
        validBuf->fill(false);
        g->parallelForEach([&](int i, int j, int k) {
          if (valid->at(i, j, k)) {
            validBuf->at(i, j, k) = true;
            return;
          }
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
          if (count > 0) {
            gbuf->at(i, j, k) = sum / count;
            validBuf->at(i, j, k) = true;
          } else {
            gbuf->at(i, j, k) = 0.0;
            validBuf->at(i, j, k) = false;
          }
        });
        std::swap(g, gbuf);
        std::swap(valid, validBuf);
      }
    }
    void clear();
    void applyForce(Real dt) const;
    void applyCollider() const;
    void applyDirichletBoundary() const;
    [[nodiscard]] Real CFL() const;
    void substep(Real dt);
    std::unique_ptr<FaceCentredGrid<Real, Real, 3, 0>> ug, ubuf;
    std::unique_ptr<FaceCentredGrid<Real, Real, 3, 1>> vg, vbuf;
    std::unique_ptr<FaceCentredGrid<Real, Real, 3, 2>> wg, wbuf;
    std::unique_ptr<Array3D<char>> uValid, vValid, wValid, uValidBuf, vValidBuf,
        wValidBuf, sdfValid, sdfValidBuf;
    Array3D<Real> uw, vw, ww;
    Array3D<Real> pg;
    ParticleSystemReconstructor<Real, 3>* fluidSurfaceReconstructor{};
    std::unique_ptr<SDF<3>> fluidSurface{}, fluidSurfaceBuf{}, colliderSdf{};
    ProjectionSolver3D* projector{};
    HybridAdvectionSolver3D* advector{};
};
} // namespace fluid
#endif  // SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_