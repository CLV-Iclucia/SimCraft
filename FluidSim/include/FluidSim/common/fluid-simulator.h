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
namespace fluid {

class ApicSimulator2D : public core::Animation {
 public:
  core::Timer timer;
  ~ApicSimulator2D() override = default;
  void init(int n, const Vec2f &size, const Vec2i &resolution);
  void step(core::Frame &frame) override;
  const std::vector<Vec2f> &exportParticlePositions() const {
    return m_particles.positions;
  }
  const std::vector<Vec2f> &positions() const { return m_particles.positions; }
 private:
  int nParticles;
  struct ParticleList {
    std::vector<Vec2f> positions;
    std::vector<Vec2f> velocities;
  } m_particles;
  const Vec2f &pos(int i) const { return m_particles.positions[i]; }
  const Vec2f &vel(int i) const { return m_particles.velocities[i]; }
  Vec2f &pos(int i) { return m_particles.positions[i]; }
  Vec2f &vel(int i) { return m_particles.velocities[i]; }
  void clear(const core::Frame &frame);
  void advect(const core::Frame &frame);
  void applyForce(const core::Frame &frame);
  void applyBoundary(const core::Frame &frame);
  void project(const core::Frame &frame);
  void p2g();
  void g2p();
  std::unique_ptr<FaceCentredGrid<float, 2, 0>> u_grid, u_grid_buf, mu_grid;
  std::unique_ptr<FaceCentredGrid<float, 2, 1>> v_grid, v_grid_buf, mv_grid;
  std::unique_ptr<core::CellCentredGrid<float, 2>> div_v_grid;
  std::unique_ptr<core::CellCentredGrid<float, 2>> p_grid, p_grid_buf;
};

class ApicSimulator3D : public core::Animation {
 public:
  core::Timer timer;
  ~ApicSimulator3D() override = default;
  void init(int n, const Vec3f &size, const Vec3i &resolution);
  void step(core::Frame &frame) override;

 private:
  int nParticles = 0;
  // size and resolution of the grid
  struct ParticleList {
    std::vector<Vec3f> positions;
    std::vector<Vec3f> velocities;
  } m_particles;
  const Vec3f &pos(int i) const { return m_particles.positions[i]; }
  const Vec3f &vel(int i) const { return m_particles.velocities[i]; }
  Vec3f &pos(int i) { return m_particles.positions[i]; }
  Vec3f &vel(int i) { return m_particles.velocities[i]; }
  void advect();
  void applyBoundary();
  void applyForce();
  void project();
  void p2g();
  void g2p();
  std::unique_ptr<FaceCentredGrid<float, 3, 0>> u_grid, u_grid_buf;
  std::unique_ptr<FaceCentredGrid<float, 3, 1>> v_grid, v_grid_buf;
  std::unique_ptr<FaceCentredGrid<float, 3, 2>> w_grid, w_grid_buf;
  std::unique_ptr<core::CellCentredGrid<float, 3>> div_v_grid;
  std::unique_ptr<core::CellCentredGrid<float, 3>> p_grid, p_grid_buf;
};
}  // namespace fluid
#endif  // SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
