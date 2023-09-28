//
// Created by creeper on 23-8-13.
//

#ifndef SIMCRAFT_MPM_INCLUDE_MPM_CPU_H_
#define SIMCRAFT_MPM_INCLUDE_MPM_CPU_H_
#include "Core/range-for.h"
#include "constitutive-model.h"
#include <Core/animation.h>
#include <Core/timer.h>
#include <MPM/mpm.h>
#include <algorithm>
#include <memory>
namespace mpm {
using std::max;
using std::min;
template <typename Derived, int Dim> class MpmCpu : public core::Animation {
public:
  Derived &derived() { return static_cast<Derived &>(*this); }
  const Derived &derived() const { return static_cast<const Derived &>(*this); }
  void step(core::Frame &frame) override {
    timer.startCpuTimer();
    derived().initStep();
    derived().P2G();
    derived().gridUpdate(frame.dt);
    derived().G2P();
    timer.stopCpuTimer();
  }

protected:
  mutable core::Timer timer;
  int numParticles = 0;
};

/**
 * Original MPM algorithm, velocity, mass, momentum and deformation gradient are
 * stored on particles.
 * @tparam Dim dimension of animation
 */
template <typename Stencil, int Dim>
class ImplicitMpmCpu : public MpmCpu<ImplicitMpmCpu<Stencil, Dim>, Dim> {
public:
  // constructor
  ImplicitMpmCpu() {
    v_grid = std::make_unique<Grid<Vector<Real, Dim>, Dim>>();
    mv_grid = std::make_unique<Grid<Vector<Real, Dim>, Dim>>();
    F_grid = std::make_unique<Grid<Matrix<Real, Dim>, Dim>>();
    active_idx = std::make_unique<Grid<int, Dim>>();
    m_grid = std::make_unique<Grid<Real, Dim>>();
    if constexpr (Dim == 2)
      gravity = Vector<Real, Dim>(0, -9.8);
    if constexpr (Dim == 3)
      gravity = Vector<Real, Dim>(0, -9.8, 0);
  }
  void initStep() {
    std::printf("Initializing step...\n");
    timer.startCpuTimer();
    m_grid->clear();
    mv_grid->clear();
    F_grid->clear();
    particleShuffle();
    timer.stopCpuTimer();
    std::printf("Initializing done, taking %lf ms\n", timer.CpuElapsedTime());
  }
  // in this step, both the transfer and grid velocities are computed
  void P2G() {
    std::printf("Transferring quantities from particles to grids...\n");
    timer.startCpuTimer();
    for (int i = 0; i < numParticles; i++) {
      const Vector<Real, Dim> &p = pos(i);
      const Vector<int, Dim> &size = v_grid->size();
      const Vector<Real, Dim> &h = v_grid->gridSpacing();
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
          if constexpr (Dim == 2) {
            Real &mass = m_grid->at(j, k);
            Vector<Real, Dim> &momentum = mv_grid->at(j, k);
            Matrix<Real, Dim> &deformation_gradient = F_grid->at(j, k);
            Real Nx = Stencil::N(p.x / h.x - j);
            Real Ny = Stencil::N(p.y / h.y - k);
            mass += Nx * Ny * m(i);
            momentum += Nx * Ny * mv(i);
            deformation_gradient += Nx * Ny * F(i);
          }
          if constexpr (Dim == 3) {
            for (int l = max(static_cast<int>(
                                 ceil(p.z / h.z - Stencil::kSupportRadius)),
                             0);
                 l <= min(static_cast<int>(p.z / h.z + Stencil::kSupportRadius),
                          size.z - 1);
                 l++) {
              Vector<Real, Dim> &mass = m_grid->at(j, k, l);
              Vector<Real, Dim> &momentum = mv_grid->at(j, k, l);
              Matrix<Real, Dim> &deformation_gradient = F_grid->at(j, k, l);
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
    }
    for (int i = 0; i < v_grid->size().x; i++) {
      for (int j = 0; j < v_grid->size().y; j++) {
        if constexpr (Dim == 2) {
          if (m_grid->at(i, j) > 0) {
            v_grid->at(i, j) = mv_grid->at(i, j) / m_grid->at(i, j);
            active_grids.emplace_back(i, j);
            active_idx->at(i, j) = active_grids.size() - 1;
          }
        }
        if constexpr (Dim == 3) {
          for (int k = 0; k < v_grid->size().z; k++) {
            if (m_grid->at(i, j, k) > 0) {
              v_grid->at(i, j, k) = mv_grid->at(i, j, k) / m_grid->at(i, j, k);
              active_grids.emplace_back(i, j, k);
              active_idx->at(i, j, k) = active_grids.size() - 1;
            }
          }
        }
      }
    }
    timer.stopCpuTimer();
    std::printf("Transferring done, taking %lf ms\n", timer.CpuElapsedTime());
  }
  void gridUpdate(Real dt) {
    std::printf("Updating quantities on grids...\n");
    timer.startCpuTimer();
    vector<Triplet<Real>> t_list;
    auto assembleSparseMatrix = [&t_list](int i, int j, Real val) {
      t_list.emplace_back(i, j, val);
    };
    VectorXd rhs(active_grids.size() * Dim);
    // first accumulate grid impacts
    for (int i = 0; i < numParticles; i++) {
      const Vector<Real, Dim> &p = pos(i);
      const Vector<int, Dim> &size = v_grid->size();
      const Vector<Real, Dim> &h = v_grid->gridSpacing();
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
          if constexpr (Dim == 2) {
            rhs.segment<Dim>(active_idx->at(i, j) * Dim) +=
                Volume * PK1Stress<Dim>(material(i), F_grid->at(j, k)) *
                glm::transpose(F(i)) *
                Stencil::weightGradient<Dim>(
                    h, Vector<Real, Dim>(p.x / h.x - j, p.y / h.y - k)) *
                dt / m_grid->at(i, j);
          }
          if constexpr (Dim == 3) {
            for (int l = max(static_cast<int>(
                                 ceil(p.z / h.z - Stencil::kSupportRadius)),
                             0);
                 l <= min(static_cast<int>(p.z / h.z + Stencil::kSupportRadius),
                          size.z - 1);
                 l++) {
              rhs.segment<Dim>(active_idx->at(i, j) * Dim) +=
                  Volume * PK1Stress<Dim>(material(i), F_grid->at(j, k, l)) *
                  F(i).transposed() *
                  Stencil::weightGradient<Dim>(
                      h, Vector<Real, Dim>(p.x / h.x - j, p.y / h.y - k,
                                           p.z / h.z - l)) *
                  dt / m_grid->at(i, j, k);
            }
          }
        }
      }
    }
    // add up original velocities and gravity as right hand side
    for (auto &grid : active_grids) {
      rhs.segment<Dim>(active_idx->at(grid) * Dim) +=
          v_grid->at(grid) + gravity * dt;
    }
    // assemble left hand side
    // for this part, reference the equation (198) in Chenfanfu Jiang's
    // SIGGRAPH course notes on MPM in 2016
    // TODO: finish this
    FourthOrderTensor<Real, Dim> pP_pF;
    F_grid->forEach([this](const Vector<int, Dim> &index_i) {
      F_grid->forGridNeighbours(
          index_i, 2 * Stencil::kSupportRadius,
          [&index_i, this](const Vector<int, Dim> &index_j) {
            Matrix<Real, Dim> H;
            core::Range<Dim> range = F_grid->computeIntersectionNeighbourhoods(
                index_i, index_j, 2 * Stencil::kSupportRadius);
            core::forRange(
                range, [&H, &index_i, &index_j, this](const auto &idx) {
                  // all the particles in a grid
                  Index begin = grid_begin_idx[idx];
                  Index end = grid_end_idx[idx];
                  for (int p_idx = begin; p_idx <= end; p_idx++) {
                    Index p = particle(p_idx);
                    const Vector<Real, Dim> &p_pos = pos(p);
                    const Vector<Real, Dim> &p_v = v(p);
                    const Vector<Real, Dim> &p_mv = mv(p);
                    const Matrix<Real, Dim> &p_F = F(p);

                  }
                });
          });
    });
    // update deformation gradient on particles after we solve the velocities
    for (int i = 0; i < numParticles; i++) {
      const Vector<Real, Dim> &p = pos(i);
      Matrix<Real, Dim> diff;
      v_grid->forNeighbours(
          p, Stencil::kSupportRadius,
          [&diff, &p, dt, this](const Vector<int, Dim> &idx) {
            const Vector<int, Dim> &size = v_grid->size();
            const Vector<Real, Dim> &h = v_grid->gridSpacing();
            diff += dt * core::tensorProduct(
                             v_grid->at(idx),
                             Stencil::weightGradient<Dim>(h, p / h - idx));
          });
      F(i) += diff * F(i);
    }
    timer.stopCpuTimer();
    std::printf("Updating done, taking %lf ms\n", timer.CpuElapsedTime());
  }
  void G2P() {
    std::printf("Updating quantities on grids...\n");
    timer.startCpuTimer();

    timer.stopCpuTimer();
    std::printf("Updating done, taking %lf ms\n", timer.CpuElapsedTime());
  }
  void setGravity(const Vector<Real, Dim> &g) { gravity = g; }
  const Vector<Real, Dim> &getGravity() const { return gravity; }

private:
  using Base = MpmCpu<ImplicitMpmCpu<Stencil, Dim>, Dim>;
  using Base::numParticles;
  using Base::timer;
  // use SOA to store the particle quantities
  // the quantities stored here are SHUFFLED
  struct ParticleList {
    vector<Material> m_material;
    vector<Real> m_mass;
    vector<Real> m_V; // initial volume, this should not be changed
    vector<Vector<Real, Dim>> m_pos, m_v, m_mv;
    vector<Matrix<Real, Dim>> m_F;
  } particles;
  Index particle(Index i) const { return particle_idx[i]; }
  const Material &material(Index i) const { return particles.m_material[i]; }
  Real V(Index i) const { return particles.m_V[i]; }
  Real &m(Index i) { return particles.m_mass[i]; }
  Real m(Index i) const { return particles.m_mass[i]; }
  Vector<Real, Dim> &pos(Index i) { return particles.m_pos[i]; }
  const Vector<Real, Dim> &pos(Index i) const { return particles.m_pos[i]; }
  Vector<Real, Dim> &v(Index i) { return particles.m_v[i]; }
  const Vector<Real, Dim> &v(Index i) const { return particles.m_v[i]; }
  Vector<Real, Dim> &mv(Index i) { return particles.m_mv[i]; }
  const Vector<Real, Dim> &mv(Index i) const { return particles.m_mv[i]; }
  Matrix<Real, Dim> &F(Index i) { return particles.m_F[i]; }
  const Matrix<Real, Dim> &F(Index i) const { return particles.m_F[i]; }
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
  // TODO: Now it only supports grids with offset (0, 0, 0), add more support
  std::unique_ptr<Grid<Vector<Real, Dim>, Dim>> v_grid;
  std::unique_ptr<Grid<Vector<Real, Dim>, Dim>> mv_grid;
  std::unique_ptr<Grid<Matrix<Real, Dim>, Dim>> F_grid;
  std::unique_ptr<Grid<int, Dim>> active_idx;
  std::unique_ptr<Grid<Real, Dim>> m_grid;
  std::vector<Vector<int, Dim>> active_grids;
  Vector<Real, Dim> gravity;
  vector<Index> particle_idx;
  vector<Index> particle_grid_idx;
  vector<Index> grid_particle_idx;
  vector<Index> grid_begin_idx;
  vector<Index> grid_end_idx;
};
} // namespace mpm
#endif // SIMCRAFT_MPM_INCLUDE_MPM_CPU_H_
