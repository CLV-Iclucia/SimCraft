//
// Created by creeper on 23-8-13.
//

#ifndef SIMCRAFT_MPM_INCLUDE_MPM_CPU_H_
#define SIMCRAFT_MPM_INCLUDE_MPM_CPU_H_
#include <Core/animation.h>
#include <Core/timer.h>
#include <Spatify/ns-util.h>
#include <MPM/mpm.h>
#include <algorithm>
#include <memory>
namespace mpm {
using std::max;
using std::min;
/**
 * Original MPM algorithm, velocity, mass, momentum and deformation gradient are
 * stored on particles.
 * @tparam 3 dimension of animation
 */
class ImplicitMpmCpu {
public:
  // constructor
  ImplicitMpmCpu() {
    v_grid = std::make_unique<CellCentredGrid<Vector<Real, 3>, Real, 3>>();
    mv_grid = std::make_unique<CellCentredGrid<Vector<Real, 3>, Real, 3>>();
    F_grid = std::make_unique<CellCentredGrid<Matrix<Real, 3>, Real, 3>>();
    m_grid = std::make_unique<CellCentredGrid<Real, Real, 3>>();
    gravity = Vector<Real, 3>(0, -9.8, 0);
  }
  void initStep() {
    core::TimerGuard<core::CpuTimer> guard("Initialize");
    m_grid->clear();
    mv_grid->clear();
    F_grid->clear();
  }
  // in this step, both the transfer and grid velocities are computed
  void P2G() {
    core::TimerGuard<core::CpuTimer> timer_guard("Grid Update");
    for (int i = 0; i < numParticles; i++) {
      const Vector<Real, 3> &p = pos(i);
      const Vector<int, 3> &size = v_grid->size();
      const Vector<Real, 3> &h = v_grid->gridSpacing();
      for (int j = max(
               static_cast<int>(std::ceil(p.x / h.x - Stencil::kSupportRadius)),
               0);
           j <= min(static_cast<int>(p.x / h.x + Stencil::kSupportRadius),
                    size.x - 1);
           j++) {
        for (int k = max(static_cast<int>(
                             std::ceil(p.y / h.y - Stencil::kSupportRadius)),
                         0);
             k <= min(static_cast<int>(p.y / h.y + Stencil::kSupportRadius),
                      size.y - 1);
             k++) {
            for (int l = max(static_cast<int>(
                                 ceil(p.z / h.z - Stencil::kSupportRadius)),
                             0);
                 l <= min(static_cast<int>(p.z / h.z + Stencil::kSupportRadius),
                          size.z - 1);
                 l++) {
              auto &mass = m_grid->at(j, k, l);
              auto &momentum = mv_grid->at(j, k, l);
              auto &deformation_gradient = F_grid->at(j, k, l);
              Real Nx = Stencil::N(p.x / h.x - j);
              Real Ny = Stencil::N(p.y / h.y - k);
              Real Nz = Stencil::N(p.z / h.z - l);
              mass += Nx * Ny * Nz * m(i);
              momentum += Nx * Ny * Nz * mv(i);
              deformation_gradient += Nx * Ny * Nz * F(i);
            }
        }
      }
    }
    for (int i = 0; i < v_grid->size().x; i++) {
      for (int j = 0; j < v_grid->size().y; j++) {
          for (int k = 0; k < v_grid->size().z; k++) {
            if (m_grid->at(i, j, k) > 0) {
              v_grid->at(i, j, k) = mv_grid->at(i, j, k) / m_grid->at(i, j, k);
              active_grids.emplace_back(i, j, k);
              active_idx->at(i, j, k) = active_grids.size() - 1;
            }
          }
      }
    }
    timer.stopCpuTimer();
    std::printf("Transferring done, taking %lf ms\n", timer.CpuElapsedTime());
  }
  void gridUpdate(Real dt) {
    core::TimerGuard<core::CpuTimer> timer_guard("Grid Update");
    vector<Triplet<Real>> t_list;
    auto assembleSparseMatrix = [&t_list](int i, int j, Real val) {
      t_list.emplace_back(i, j, val);
    };
    VectorXd rhs(active_grids.size() * 3);
    // first accumulate grid impacts
    for (int i = 0; i < numParticles; i++) {
      const Vector<Real, 3> &p = pos(i);
      const Vector<int, 3> &size = v_grid->size();
      const Vector<Real, 3> &h = v_grid->gridSpacing();
      Real Volume = V(i);
      for (int j = max(
               static_cast<int>(std::ceil(p.x / h.x - Stencil::kSupportRadius)),
               0);
           j <= min(static_cast<int>(p.x / h.x + Stencil::kSupportRadius),
                    size.x - 1);
           j++) {
        for (int k = max(static_cast<int>(
                             std::ceil(p.y / h.y - Stencil::kSupportRadius)),
                         0);
             k <= min(static_cast<int>(p.y / h.y + Stencil::kSupportRadius),
                      size.y - 1);
             k++) {
            for (int l = max(static_cast<int>(
                                 ceil(p.z / h.z - Stencil::kSupportRadius)),
                             0);
                 l <= min(static_cast<int>(p.z / h.z + Stencil::kSupportRadius),
                          size.z - 1);
                 l++) {
              rhs.segment<3>(active_idx->at(i, j) * 3) +=
                  Volume * PK1Stress<3>(material(i), F_grid->at(j, k, l)) *
                  F(i).transposed() *
                  Stencil::weightGradient<3>(
                      h, Vector<Real, 3>(p.x / h.x - j, p.y / h.y - k,
                                           p.z / h.z - l)) *
                  dt / m_grid->at(i, j, k);
            }
        }
      }
    }
    // add up original velocities and gravity as right hand side
    for (auto &grid : active_grids) {
      rhs.segment<3>(active_idx->at(grid) * 3) +=
          v_grid->at(grid) + gravity * dt;
    }
    // assemble left hand side
    // for this part, reference the equation (198) in Chenfanfu Jiang's
    // SIGGRAPH course notes on MPM in 2016
    // TODO: finish this
    // update deformation gradient on particles after we solve the velocities
    for (int i = 0; i < numParticles; i++) {
      const Vector<Real, 3> &p = pos(i);
      Matrix<Real, 3> diff;
      v_grid->forNeighbours(
          p, Stencil::kSupportRadius,
          [&diff, &p, dt, this](const Vector<int, 3> &idx) {
            const Vector<int, 3> &size = v_grid->size();
            const Vector<Real, 3> &h = v_grid->gridSpacing();
            diff += dt * core::tensorProduct(
                             v_grid->at(idx),
                             Stencil::weightGradient<3>(h, p / h - idx));
          });
      F(i) += diff * F(i);
    }
  }
  void G2P() {

  }
  void setGravity(const Vector<Real, 3> &g) { gravity = g; }
  const Vector<Real, 3> &getGravity() const { return gravity; }

private:
  int numParticles;
  // use SOA to store the particle quantities
  // the quantities stored here are SHUFFLED
  struct ParticleList {
    vector<Material> m_material;
    vector<Real> m_mass;
    vector<Real> m_V; // initial volume, this should not be changed
    vector<Vector<Real, 3>> m_pos, m_v, m_mv;
    vector<Matrix<Real, 3>> m_F;
  } particles;
  [[nodiscard]] Index particle(Index i) const { return particle_idx[i]; }
  const Material &material(Index i) const { return particles.m_material[i]; }
  [[nodiscard]] Real V(Index i) const { return particles.m_V[i]; }
  Real &m(Index i) { return particles.m_mass[i]; }
  [[nodiscard]] Real m(Index i) const { return particles.m_mass[i]; }
  Vector<Real, 3> &pos(Index i) { return particles.m_pos[i]; }
  [[nodiscard]] const Vector<Real, 3> &pos(Index i) const { return particles.m_pos[i]; }
  Vector<Real, 3> &v(Index i) { return particles.m_v[i]; }
  [[nodiscard]] const Vector<Real, 3> &v(Index i) const { return particles.m_v[i]; }
  Vector<Real, 3> &mv(Index i) { return particles.m_mv[i]; }
  [[nodiscard]] const Vector<Real, 3> &mv(Index i) const { return particles.m_mv[i]; }
  Matrix<Real, 3> &F(Index i) { return particles.m_F[i]; }
  [[nodiscard]] const Matrix<Real, 3> &F(Index i) const { return particles.m_F[i]; }
  // we need to maintain the particles in each grid
  // first find out what grids they are in
  // then sort them according to the grid index
  // so the particles in the same grid will be together
  void particleShuffle() {
    // TODO: fixing
    for (int i = 0; i < numParticles; i++) {
      Index grid_idx = v_grid->coordToIndex(pos(i));
      grid_particle_idx[grid_idx] = i;
      particle_grid_idx[i] = grid_idx;
      particle_idx[i] = i;
    }
    auto particle_shuffle = [this](Index i, Index j) {
      return particle_grid_idx[i] < particle_grid_idx[j];
    };
    std::sort(particle_idx.begin(), particle_idx.end(), particle_shuffle);
    // using particle_grid_idx to apply the permutation to other vectors
    // next we need to find out the beginning and end index of each grid
    for (int i = 0; i < numParticles; i++) {
      Index grid_idx = particle_grid_idx[particle(i)];
      if (i == 0 || grid_idx != particle_grid_idx[particle(i - 1)])
        grid_begin_idx[grid_idx] = i;
      if (i == numParticles - 1 || grid_idx != particle_grid_idx[particle(i + 1)])
        grid_end_idx[grid_idx] = i;
    }
  }
  // these grids should have the same shape
  std::unique_ptr<CellCentredGrid<Vector<Real, 3>, Real, 3>> v_grid;
  std::unique_ptr<CellCentredGrid<Vector<Real, 3>, Real, 3>> mv_grid;
  std::unique_ptr<CellCentredGrid<Matrix<Real, 3>, Real, 3>> F_grid;
  Array3D<int> active_idx;
  std::unique_ptr<CellCentredGrid<Real, Real, 3>> m_grid;
  spatify::ParticleNeighbourSearcher<Real, 3> ns;
  std::vector<Vector<int, 3>> active_grids;
  Vector<Real, 3> gravity;
  vector<Index> particle_idx;
  vector<Index> particle_grid_idx;
  vector<Index> grid_particle_idx;
  vector<Index> grid_begin_idx;
  vector<Index> grid_end_idx;
};
} // namespace mpm
#endif // SIMCRAFT_MPM_INCLUDE_MPM_CPU_H_
