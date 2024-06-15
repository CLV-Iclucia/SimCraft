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
private:
  int numParticles;
  struct Particles {
    vector<Real> m_mass;
    vector<Real> m_V; // initial volume, this should not be changed
    vector<Vector<Real, 3>> m_pos, m_v, m_mv;
    vector<Matrix<Real, 3>> m_F;
  } particles;
  [[nodiscard]] Index particle(Index i) const { return particle_idx[i]; }
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
  std::unique_ptr<CellCentredGrid<Vector<Real, 3>, Real, 3>> v_grid;
  std::unique_ptr<CellCentredGrid<Vector<Real, 3>, Real, 3>> mv_grid;
  std::unique_ptr<CellCentredGrid<Matrix<Real, 3>, Real, 3>> F_grid;
  std::unique_ptr<Array3D<int>> active_idx;
  std::unique_ptr<CellCentredGrid<Real, Real, 3>> m_grid;
  std::vector<Vector<int, 3>> active_grids;
  Vector<Real, 3> external_force;
  vector<Index> particle_idx;
  vector<Index> particle_grid_idx;
  vector<Index> grid_particle_idx;
  vector<Index> grid_begin_idx;
  vector<Index> grid_end_idx;
};
} // namespace mpm
#endif // SIMCRAFT_MPM_INCLUDE_MPM_CPU_H_
