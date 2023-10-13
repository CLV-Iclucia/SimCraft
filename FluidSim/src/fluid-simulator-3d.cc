#include <FluidSim/common/fluid-simulator.h>
namespace fluid {
void ApicSimulator3D::init(int n, const Vec3f &size, const Vec3i &resolution) {
  nParticles = n;
  m_particles.positions.resize(n);
  m_particles.velocities.resize(n);
  // compute grid spacing
  Vec3f gridSpacing = size / Vec3f(resolution);
  // initialize grids
  u_grid = std::make_unique<FaceCentredGrid<float, 3, 0>>(
      resolution + Vec3i(1, 0, 0), Vec3f(gridSpacing));
  u_grid_buf = std::make_unique<FaceCentredGrid<float, 3, 0>>(
      resolution + Vec3i(1, 0, 0), Vec3f(gridSpacing));
  v_grid = std::make_unique<FaceCentredGrid<float, 3, 1>>(
      resolution + Vec3i(0, 1, 0), Vec3f(gridSpacing));
  v_grid_buf = std::make_unique<FaceCentredGrid<float, 3, 1>>(
      resolution + Vec3i(0, 1, 0), Vec3f(gridSpacing));
  w_grid = std::make_unique<FaceCentredGrid<float, 3, 2>>(
      resolution + Vec3i(0, 0, 1), Vec3f(gridSpacing));
  p_grid = std::make_unique<CellCentredGrid<float, 3>>(resolution, gridSpacing);
}
void ApicSimulator3D::p2g() {}
void ApicSimulator3D::g2p() {}
void ApicSimulator3D::applyForce() {}
void ApicSimulator3D::project() {}

void ApicSimulator3D::advect() {}

void ApicSimulator3D::step(core::Frame &frame) {}
} // namespace fluid
