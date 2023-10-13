#include <Core/animation.h>
#include <Core/rand-gen.h>
#include <Core/transfer-stencil.h>
#include <FluidSim/common/fluid-simulator.h>
#include <Poisson/poisson-solver.h>
#include <cassert>
namespace fluid {

void ApicSimulator2D::init(int n, const Vec2f &size, const Vec2i &resolution) {
  nParticles = n;
  m_particles.positions.resize(n);
  m_particles.velocities.resize(n);
  // compute grid spacing
  Vec2f gridSpacing = size / Vec2f(resolution);
  // initialize grids
  u_grid = std::make_unique<FaceCentredGrid<float, 2, 0>>(
      resolution + Vec2i(1, 0), Vec2f(gridSpacing));
  v_grid = std::make_unique<FaceCentredGrid<float, 2, 1>>(
      resolution + Vec2i(0, 1), Vec2f(gridSpacing));
  u_grid_buf = std::make_unique<FaceCentredGrid<float, 2, 0>>(
      resolution + Vec2i(1, 0), Vec2f(gridSpacing));
  v_grid_buf = std::make_unique<FaceCentredGrid<float, 2, 1>>(
      resolution + Vec2i(0, 1), Vec2f(gridSpacing));
  p_grid = std::make_unique<CellCentredGrid<float, 2>>(resolution, gridSpacing);
  p_grid_buf =
      std::make_unique<CellCentredGrid<float, 2>>(resolution, gridSpacing);
  for (int i = 0; i < n; i++) {
    pos(i) = core::randomVec<float, 2>() * 0.5f + Vec2f(0.5f, 0.5f);
    vel(i) = core::randomVec<float, 2>() * 0.5f;
  }
}
void ApicSimulator2D::p2g() {
  // assertion: all the grids should have the same spacing
  assert(u_grid->gridSpacing() == v_grid->gridSpacing() &&
         v_grid->gridSpacing() == p_grid->gridSpacing());
  for (int i = 0; i < nParticles; i++) {
    auto p = pos(i);
    auto v = vel(i);
    auto h = u_grid->gridSpacing();
    Vec2i u_idx = u_grid->nearest(p);
    Vec2i v_idx = v_grid->nearest(p);
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        u_grid->at(u_idx + Vec2i(j, k)) +=
            v.x * core::CubicKernel::weight<float, 2>(
                      h, p - u_grid->indexToCoord(u_idx + Vec2i(j, k)));
        mu_grid->at(u_idx + Vec2i(j, k)) += core::CubicKernel::weight<float, 2>(
            h, p - u_grid->indexToCoord(u_idx + Vec2i(j, k)));
        v_grid->at(v_idx + Vec2i(j, k)) +=
            v.y * core::CubicKernel::weight<float, 2>(
                      h, p - v_grid->indexToCoord(v_idx + Vec2i(j, k)));
        mv_grid->at(v_idx + Vec2i(j, k)) += core::CubicKernel::weight<float, 2>(
            h, p - v_grid->indexToCoord(v_idx + Vec2i(j, k)));
      }
    }
  }
  u_grid->forEach([this](const Vec2i &idx) {
    if (mu_grid->at(idx) > 0.0f)
      u_grid->at(idx) /= mu_grid->at(idx);
  });
  v_grid->forEach([this](const Vec2i &idx) {
    if (mv_grid->at(idx) > 0.0f)
      v_grid->at(idx) /= mv_grid->at(idx);
  });
}
void ApicSimulator2D::g2p() {
  // assertion: all the grids should have the same spacing
  assert(u_grid->gridSpacing() == v_grid->gridSpacing() &&
         v_grid->gridSpacing() == p_grid->gridSpacing());
  for (int i = 0; i < nParticles; i++) {
    auto p = pos(i);
    auto h = u_grid->gridSpacing();
    Vec2i u_idx = u_grid->nearest(p);
    Vec2i v_idx = v_grid->nearest(p);
    Vec2f u, v;
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        u += u_grid->at(u_idx + Vec2i(j, k)) *
             core::CubicKernel::weight<float, 2>(
                 h, p - u_grid->indexToCoord(u_idx + Vec2i(j, k)));
        v += v_grid->at(v_idx + Vec2i(j, k)) *
             core::CubicKernel::weight<float, 2>(
                 h, p - v_grid->indexToCoord(v_idx + Vec2i(j, k)));
      }
    }
    vel(i) = Vec2f(u.x, v.y);
  }
}
void ApicSimulator2D::applyBoundary(const core::Frame &frame) {
  // clamp and bounce back
  for (int i = 0; i < nParticles; i++) {
    auto p = pos(i);
    if (p.x < 0.0f) {
      p.x = 0.0f;
      vel(i).x = -vel(i).x;
    }
    if (p.x > 1.0f) {
      p.x = 1.0f;
      vel(i).x = -vel(i).x;
    }
    if (p.y < 0.0f) {
      p.y = 0.0f;
      vel(i).y = -vel(i).y;
    }
    if (p.y > 1.0f) {
      p.y = 1.0f;
      vel(i).y = -vel(i).y;
    }
  }
  // for the speed components on the boundary, set them to zero
  for (int i = 0; i < u_grid->width(); i++) {
    u_grid->at(Vec2i(i, 0)) = 0.0f;
    u_grid->at(Vec2i(i, u_grid->height() - 1)) = 0.0f;
  }
  for (int i = 0; i < v_grid->height(); i++) {
    v_grid->at(Vec2i(0, i)) = 0.0f;
    v_grid->at(Vec2i(v_grid->width() - 1, i)) = 0.0f;
  }
}
void ApicSimulator2D::applyForce(const core::Frame &frame) {
  for (int i = 0; i < nParticles; i++)
    vel(i) += Vec2f(0.0f, -9.8f) * static_cast<float>(frame.dt);
}

// solve equation dt / rho * div(grad(p)) = div(u)
// and project out the divergence of the velocity field
void ApicSimulator2D::project(const core::Frame &frame) {
  // calculate the velocity divergence and store in div_v_grid
  const Vec2f &h = u_grid->gridSpacing();
  div_v_grid->forEach([this, &h, &frame](const Vec2i &idx) {
    div_v_grid->at(idx) =
        (u_grid->at(idx + Vec2i(1, 0)) - u_grid->at(idx)) / h.x +
        (v_grid->at(idx + Vec2i(0, 1)) - v_grid->at(idx)) / h.y;
    div_v_grid ->at(idx) /= static_cast<float>(frame.dt);
  });
  poisson::mgpcgSolve();
  // subtract the gradient of the pressure field
  p_grid->forInside([this, &h](const Vec2i &idx) {
    u_grid->at(idx) -= (p_grid->at(idx + Vec2i(-1, 0)) - p_grid->at(idx)) / h.x;
    v_grid->at(idx) -= (p_grid->at(idx + Vec2i(0, -1)) - p_grid->at(idx)) / h.y;
  });
}

void ApicSimulator2D::advect(const core::Frame &frame) {
  for (int i = 0; i < nParticles; i++)
    pos(i) += vel(i) * static_cast<float>(frame.dt);
}
void ApicSimulator2D::clear(const core::Frame &frame) {
  u_grid->clear();
  v_grid->clear();
  p_grid->clear();
  mu_grid->clear();
  mv_grid->clear();
}
void ApicSimulator2D::step(core::Frame &frame) {
  clear(frame);
  applyForce(frame);
  advect(frame);
  p2g();
  applyBoundary(frame);
  project(frame);
  g2p();
}

} // namespace fluid