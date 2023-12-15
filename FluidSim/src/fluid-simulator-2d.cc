#include <Core/animation.h>
#include <Core/rand-gen.h>
#include <Core/transfer-stencil.h>
#include <FluidSim/common/fluid-simulator.h>
#include <FluidSim/common/advect-solver.h>
#include <FluidSim/common/project-solver.h>
#include <cassert>
namespace fluid {

void HybridFluidSimulator2D::initSolver(const Vec2i &resolution) {

}

void HybridFluidSimulator2D::init(int n, const Vec2d &size, const Vec2i &resolution) {
  nParticles = n;
  m_particles.positions.resize(n);
  m_particles.velocities.resize(n);
  // compute grid spacing
  Vec2d gridSpacing = size / Vec2d(resolution);
  // initialize grids
  u_grid = std::make_unique<FaceCentredGrid<Real, Real, 2, 0>>(
      resolution + Vec2i(1, 0), Vec2d(gridSpacing));
  mu_grid = std::make_unique<FaceCentredGrid<Real, Real, 2, 0>>(
      resolution + Vec2i(1, 0), Vec2d(gridSpacing));
  v_grid = std::make_unique<FaceCentredGrid<Real, Real, 2, 1>>(
      resolution + Vec2i(0, 1), Vec2d(gridSpacing));
  mv_grid = std::make_unique<FaceCentredGrid<Real, Real, 2, 1>>(
      resolution + Vec2i(0, 1), Vec2d(gridSpacing));
  p_grid = std::make_unique<CellCentredGrid<Real, Real, 2>>(resolution, gridSpacing);
  rhs_grid.init(resolution, gridSpacing);
  for (int i = 0; i < n; i++) {
    pos(i) = core::randomVec<Real, 2>() * 0.25 + Vec2d(0.25, 0.25);
    vel(i) = Vec2d(0.0, 0.0);
  }
  initSolver(resolution);
}

void HybridFluidSimulator2D::applyForce(Real dt) {
  v_grid->forInside([this, dt](const Vec2i &idx) {
    v_grid->at(idx) -= 9.8 * static_cast<Real>(dt);
  });
}

void HybridFluidSimulator2D::markCell() {
  markers.fill(Marker::Air, Marker::Solid);
  for (int i = 0; i < nParticles; i++)
    markers(u_grid->coordToCellIndex(pos(i))) = Marker::Fluid;
}

void HybridFluidSimulator2D::clear() {
  active_cells.clear();
  u_grid->fill(0);
  v_grid->fill(0);
  p_grid->fill(0);
}

void HybridFluidSimulator2D::substep(core::Real dt) {
  clear();
  advector->advect(positions().data(), u_grid.get(), v_grid.get(), collider_sdf.get(), dt);
  advector->solveP2G(positions().data(), u_grid.get(), v_grid.get(), dt);
  markCell();
  applyForce(dt);
  projector->buildSystem(markers, u_grid.get(), v_grid.get(), density, dt);
  projector->solve(p_grid.get());
  advector->solveG2P(positions().data(), u_grid.get(), v_grid.get(), dt);
}

Real HybridFluidSimulator2D::CFL() {
  Real cfl = 0.0;
  Real h = u_grid->gridSpacing().x;
  u_grid->forEach([&cfl, h, this](int x, int y) {
    cfl = std::max(cfl, h / u_grid->at(x, y));
  });
  v_grid->forEach([&cfl, h, this](int x, int y) {
    cfl = std::max(cfl, h / v_grid->at(x, y));
  });
  return cfl;
}
void HybridFluidSimulator2D::step(core::Frame &frame) {
  Real t = 0;
  while (t < frame.dt) {
    Real cfl = CFL();
    Real dt = std::min(cfl, frame.dt - t);
    substep(dt);
    t += dt;
  }
  frame.onAdvance();
}

} // namespace fluid