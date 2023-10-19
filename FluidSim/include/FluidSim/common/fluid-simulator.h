//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
#include <Core/animation.h>
#include <Core/data-structures/grids.h>
#include <Core/timer.h>
#include <FluidSim/common/fluid-sim.h>
#include <memory>
#include <PoissonSolver/poisson-solver.h>
#include <PoissonSolver/multigrid-solver.h>
#include <PoissonSolver/cg-solver.h>
namespace fluid {

class ApicSimulator2D : public core::Animation {
 public:
  core::Timer timer;
  ~ApicSimulator2D() override = default;
  void init(int n, const Vec2d &size, const Vec2i &resolution);
  void step(core::Frame &frame) override;
  const std::vector<Vec2d> &positions() const { return m_particles.positions; }
 private:
  int nParticles;
  Real density = 1e3;
  struct ParticleList {
    std::vector<Vec2d> positions;
    std::vector<Vec2d> velocities;
  } m_particles;
  const Vec2d &pos(int i) const { return m_particles.positions[i]; }
  const Vec2d &vel(int i) const { return m_particles.velocities[i]; }
  Vec2d &pos(int i) { return m_particles.positions[i]; }
  Vec2d &vel(int i) { return m_particles.velocities[i]; }
  void clear(const core::Frame &frame);
  void advect(const core::Frame &frame);
  void applyForce(const core::Frame &frame);
  void applyBoundary(const core::Frame &frame);
  void project(const core::Frame &frame);
  void p2g();
  void g2p();
  std::unique_ptr<FaceCentredGrid<Real, 2, 0>> u_grid, mu_grid;
  std::unique_ptr<FaceCentredGrid<Real, 2, 1>> v_grid, mv_grid;
  std::unique_ptr<core::CellCentredGrid<Real, 2>>
      rhs_grid; // usually this should be a multiple of the divergence field
  std::unique_ptr<core::CellCentredGrid<Real, 2>> p_grid, p_grid_buf;
  std::vector<Real> m_residual, cg_step, cg_Ap;
  poisson::MultigridOption mg_option;
  poisson::CgSolverOption cg_option;
  poisson::PoissonSolverOption poisson_option;
};

class ApicSimulator3D : public core::Animation {
 public:
  core::Timer timer;
  ~ApicSimulator3D() override = default;
  void init(int n, const Vec3d &size, const Vec3i &resolution);
  void step(core::Frame &frame) override;

 private:
  int nParticles = 0;
  Real density = 1e3;
  // size and resolution of the grid
  struct ParticleList {
    std::vector<Vec3d> positions;
    std::vector<Vec3d> velocities;
  } m_particles;
  const Vec3d &pos(int i) const { return m_particles.positions[i]; }
  const Vec3d &vel(int i) const { return m_particles.velocities[i]; }
  Vec3d &pos(int i) { return m_particles.positions[i]; }
  Vec3d &vel(int i) { return m_particles.velocities[i]; }
  void advect();
  void applyBoundary();
  void applyForce();
  void project();
  void p2g();
  void g2p();
  std::unique_ptr<FaceCentredGrid<Real, 3, 0>> u_grid;
  std::unique_ptr<FaceCentredGrid<Real, 3, 1>> v_grid;
  std::unique_ptr<FaceCentredGrid<Real, 3, 2>> w_grid;
  std::unique_ptr<core::CellCentredGrid<Real, 3>> div_v_grid, div_v_grid_buf;
  std::unique_ptr<core::CellCentredGrid<Real, 3>> p_grid, p_grid_buf;
  std::vector<Real> m_residual;
};
}  // namespace fluid
#endif  // SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
