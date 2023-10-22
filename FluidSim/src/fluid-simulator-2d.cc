#include <Core/animation.h>
#include <Core/rand-gen.h>
#include <Core/transfer-stencil.h>
#include <FluidSim/common/fluid-simulator.h>
#include <PoissonSolver/poisson-solver.h>
#include <cassert>
namespace fluid {

void ApicSimulator2D::init(int n, const Vec2d &size, const Vec2i &resolution) {
  nParticles = n;
  m_particles.positions.resize(n);
  m_particles.velocities.resize(n);
  // compute grid spacing
  Vec2d gridSpacing = size / Vec2d(resolution);
  // initialize grids
  u_grid = std::make_unique<FaceCentredGrid<Real, 2, 0>>(
      resolution + Vec2i(1, 0), Vec2d(gridSpacing));
  mu_grid = std::make_unique<FaceCentredGrid<Real, 2, 0>>(
      resolution + Vec2i(1, 0), Vec2d(gridSpacing));
  v_grid = std::make_unique<FaceCentredGrid<Real, 2, 1>>(
      resolution + Vec2i(0, 1), Vec2d(gridSpacing));
  mv_grid = std::make_unique<FaceCentredGrid<Real, 2, 1>>(
      resolution + Vec2i(0, 1), Vec2d(gridSpacing));
  p_grid = std::make_unique<CellCentredGrid<Real, 2>>(resolution, gridSpacing);
  p_grid_buf = std::make_unique<CellCentredGrid<Real, 2>>(resolution, gridSpacing);
  rhs_grid = std::make_unique<CellCentredGrid<Real, 2>>(
      resolution, gridSpacing);
  for (int i = 0; i < n; i++) {
    pos(i) = core::randomVec<Real, 2>() * 0.25 + Vec2d(0.25, 0.25);
    vel(i) = Vec2d(0.0, 0.0);
  }
  m_residual.reserve(resolution.x * resolution.y);
  cg_Ap.reserve(resolution.x * resolution.y);
  cg_step.reserve(resolution.x * resolution.y);
  // initialize poisson solver options
  poisson_option.dim = 2;
  poisson_option.width = resolution.x;
  poisson_option.height = resolution.y;
  poisson_option.tolerance = 1e-10;
  poisson_option.input_device = poisson::Device::CPU;
  poisson_option.solving_device = poisson::Device::CPU;
  poisson_option.output_device = poisson::Device::CPU;
  poisson_option.input_vars = p_grid->data();
  poisson_option.input_rhs = rhs_grid->data();
  poisson_option.residual = m_residual.data();
  mg_option.max_level = 6;
  mg_option.nthread = 1;
  mg_option.smooth_iter = 8;
  mg_option.bottom_iter = 40;
  mg_option.tolerance = 1e-10;
  mg_option.method = poisson::RelaxationMethod::RbGaussSeidel;
  mg_option.aux_vars = p_grid_buf->data();
  mg_option.aux_rhs = rhs_grid->data();
  mg_option.residual = m_residual.data();
  cg_option.max_iter = 1024;
  cg_option.aux_var_Ap = cg_Ap.data();
  cg_option.aux_var_step = cg_step.data();
  cg_option.preconditioner = poisson::Preconditioner::Multigrid;
}
void ApicSimulator2D::p2g() {
  // assertion: all the grids should have the same spacing
  assert(u_grid->gridSpacing() == v_grid->gridSpacing() &&
      v_grid->gridSpacing() == p_grid->gridSpacing());
  for (int i = 0; i < nParticles; i++) {
    auto &p = pos(i);
    auto &v = vel(i);
    auto &h = u_grid->gridSpacing();
    Vec2i u_idx = u_grid->nearest(p);
    Vec2i v_idx = v_grid->nearest(p);
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        // if inside the range
        if (u_idx.x + j >= 0 && u_idx.x + j < u_grid->width() &&
            u_idx.y + k >= 0 && u_idx.y + k < u_grid->height()) {
          u_grid->at(u_idx + Vec2i(j, k)) +=
              v.x * core::CubicKernel::weight<Real, 2>(
                  h, p - u_grid->indexToCoord(u_idx + Vec2i(j, k)));
          mu_grid->at(u_idx + Vec2i(j, k)) += core::CubicKernel::weight<Real, 2>(
              h, p - u_grid->indexToCoord(u_idx + Vec2i(j, k)));
        }
        if (v_idx.x + j >= 0 && v_idx.x + j < v_grid->width() &&
            v_idx.y + k >= 0 && v_idx.y + k < v_grid->height()) {
          v_grid->at(v_idx + Vec2i(j, k)) +=
              v.y * core::CubicKernel::weight<Real, 2>(
                  h, p - v_grid->indexToCoord(v_idx + Vec2i(j, k)));
          mv_grid->at(v_idx + Vec2i(j, k)) += core::CubicKernel::weight<Real, 2>(
              h, p - v_grid->indexToCoord(v_idx + Vec2i(j, k)));
        }
      }
    }
  }
  u_grid->forEach([this](const Vec2i &idx) {
    if (mu_grid->at(idx) > 0.0)
      u_grid->at(idx) /= mu_grid->at(idx);
    else u_grid->at(idx) = 0.0;
  });
  v_grid->forEach([this](const Vec2i &idx) {
    if (mv_grid->at(idx) > 0.0)
      v_grid->at(idx) /= mv_grid->at(idx);
    else v_grid->at(idx) = 0.0;
  });
}
void ApicSimulator2D::g2p() {
  // assertion: all the grids should have the same spacing
  assert(u_grid->gridSpacing() == v_grid->gridSpacing() &&
      v_grid->gridSpacing() == p_grid->gridSpacing());
  for (int i = 0; i < nParticles; i++) {
    auto &p = pos(i);
    auto &h = u_grid->gridSpacing();
    Vec2i u_idx = u_grid->nearest(p);
    Vec2i v_idx = v_grid->nearest(p);
    Real u = 0.0, v = 0.0;
    Real w_u = 0.0, w_v = 0.0;
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        if (u_idx.x + j >= 0 && u_idx.x + j < u_grid->width() &&
            u_idx.y + k >= 0 && u_idx.y + k < u_grid->height()) {
          u += u_grid->at(u_idx + Vec2i(j, k)) *
              core::CubicKernel::weight<Real, 2>(
                  h, p - u_grid->indexToCoord(u_idx + Vec2i(j, k)));
          w_u += core::CubicKernel::weight<Real, 2>(
              h, p - u_grid->indexToCoord(u_idx + Vec2i(j, k)));
        }
        if (v_idx.x + j >= 0 && v_idx.x + j < v_grid->width() &&
            v_idx.y + k >= 0 && v_idx.y + k < v_grid->height()) {
          v += v_grid->at(v_idx + Vec2i(j, k)) *
              core::CubicKernel::weight<Real, 2>(
                  h, p - v_grid->indexToCoord(v_idx + Vec2i(j, k)));
          w_v += core::CubicKernel::weight<Real, 2>(
              h, p - v_grid->indexToCoord(v_idx + Vec2i(j, k)));
        }
      }
    }
    vel(i).x = w_u > 0.0 ? u / w_u : 0.0;
    vel(i).y = w_v > 0.0 ? v / w_v : 0.0;
  }
}
void ApicSimulator2D::applyBoundary(const core::Frame &frame) {
  // for the speed components on the boundary, set them to zero
  for (int i = 0; i < u_grid->height(); i++) {
    u_grid->at(Vec2i(0, i)) = 0.0;
    u_grid->at(Vec2i(u_grid->height() - 1, i)) = 0.0;
  }
  for (int i = 0; i < v_grid->width(); i++) {
    v_grid->at(Vec2i(i, 0)) = 0.0;
    v_grid->at(Vec2i(i, v_grid->width() - 1)) = 0.0;
  }
}

void ApicSimulator2D::applyForce(const core::Frame &frame) {
  for (int i = 0; i < nParticles; i++)
    vel(i) += Vec2d(0.0, -9.8) * static_cast<Real>(frame.dt);
}

// solve equation dt / rho * div(grad(p)) = div(u)
// and project out the divergence of the velocity field
void ApicSimulator2D::project(const core::Frame &frame) {
  const Vec2d &h = u_grid->gridSpacing();
  // TODO: refine boundary condition handling
  rhs_grid->forEach([this, &h, &frame](const Vec2i &idx) {
    rhs_grid->at(idx) =
        -(u_grid->at(idx + Vec2i(1, 0)) - u_grid->at(idx)) * density * h.x  -
            (v_grid->at(idx + Vec2i(0, 1)) - v_grid->at(idx)) * density * h.x;
    if (idx.x == 0) {
      rhs_grid->at(idx) -= u_grid->at(idx) * density * h.x;
    } else if (idx.x == rhs_grid->width() - 1) {
      rhs_grid->at(idx) += u_grid->at(idx + Vec2i(1, 0)) * density * h.x;
    }
    if (idx.y == 0) {
      rhs_grid->at(idx) -= v_grid->at(idx) * density * h.x;
    } else if (idx.y == rhs_grid->height() - 1) {
      rhs_grid->at(idx) += v_grid->at(idx + Vec2i(0, 1)) * density * h.x;
    }
    rhs_grid->at(idx) /= static_cast<Real>(frame.dt);
  });
  poisson::cgSolve(poisson_option, cg_option);
  //poisson::mgpcgSolve(poisson_option, mg_option, cg_option);
  applyBoundary(frame);
  // subtract the gradient of the pressure field
  for (int i = 1; i < u_grid->width() - 1; i++)
    for (int j = 0; j < u_grid->height(); j++)
      u_grid->at(i, j) -= static_cast<Real>(frame.dt) * (p_grid->at(i, j) - p_grid->at(i - 1, j)) / h.x / density;
  for (int i = 0; i < v_grid->width(); i++)
    for (int j = 1; j < v_grid->height() - 1; j++)
      v_grid->at(i, j) -= static_cast<Real>(frame.dt) * (p_grid->at(i, j) - p_grid->at(i, j - 1)) / h.y / density;
  // compute the divergence of the velocity field to check whether it is divergence free
  rhs_grid->forEach([this, &h](const Vec2i &idx) {
    rhs_grid->at(idx) =
        (u_grid->at(idx + Vec2i(1, 0)) - u_grid->at(idx)) / h.x +
            (v_grid->at(idx + Vec2i(0, 1)) - v_grid->at(idx)) / h.y;
  });
}

void ApicSimulator2D::advect(const core::Frame &frame) {
  for (int i = 0; i < nParticles; i++) {
    auto &p = pos(i);
    p += vel(i) * static_cast<Real>(frame.dt);
    if (p.x < 0.0) {
      p.x = 0.0;
      vel(i).x = 0.0;
    }
    if (p.x > 1.0) {
      p.x = 1.0;
      vel(i).x = 0.0;
    }
    if (p.y < 0.0) {
      p.y = 0.0;
      vel(i).y = 0.0;
    }
    if (p.y > 1.0) {
      p.y = 1.0;
      vel(i).y = 0.0;
    }
  }
}
void ApicSimulator2D::clear(const core::Frame &frame) {
  u_grid->fill(0);
  v_grid->fill(0);
  p_grid->fill(0);
  mu_grid->fill(0);
  mv_grid->fill(0);
}
void ApicSimulator2D::step(core::Frame &frame) {
  clear(frame);
  applyForce(frame);
  advect(frame);
  p2g();
  // Note: we split the boundary condition handling into 2 steps and skip the step of extrapolating ghost pressures by:
  // 1. directly set the rhs in project since the condition is trivial ---- the wall cannot move at all
  // 2. after project, set the solid velocities
  project(frame);
//  applyBoundary(frame);
  g2p();
  frame.onAdvance();
}

} // namespace fluid