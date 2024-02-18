//
// Created by creeper on 23-8-9.
//

#ifndef SIMCRAFT_INCLUDE_SIMCRAFT_HAIRSIM_HAIRDOF_H_
#define SIMCRAFT_INCLUDE_SIMCRAFT_HAIRSIM_HAIRDOF_H_
#include <HairSim/hair-sim.h>

#ifndef NDEBUG
#include <cassert>
#endif
namespace hairsim {
class HairDof {
  public:
    HairDof() = default;
    explicit HairDof(int nVertices) {
      m_dof.resize(4 * nVertices - 1);
    }

    Real& theta(int i) {
      assert(m_dof.size() % 4 == 3);
      assert(4 * i + 3 < m_dof.size());
      return m_dof(4 * i + 3);
    }
    Real theta(int i) const {
      assert(m_dof.size() % 4 == 3);
      assert(4 * i + 3 < m_dof.size());
      return m_dof(4 * i + 3);
    }
    [[nodiscard]] Vec3d pos(int i) const {
      assert(m_dof.size() % 4 == 3);
      assert(4 * i + 2 < m_dof.size());
      return m_dof.segment<3>(4 * i);
    }
    Eigen::Ref<Vec3d> pos(int i) {
      assert(m_dof.size() % 4 == 3);
      assert(4 * i + 2 < m_dof.size());
      return {m_dof.segment<3>(4 * i)};
    }
    Real operator()(int i) const {
      assert(m_dof.size() % 4 == 3);
      assert(i < m_dof.size());
      return m_dof[i];
    }
    Real& operator()(int i) {
      assert(m_dof.size() % 4 == 3);
      assert(i < m_dof.size());
      return m_dof[i];
    }
    void addPos(int i, const Vec3d& v) {
      assert(m_dof.size() % 4 == 3);
      assert(4 * i + 2 < m_dof.size());
      m_dof.segment<3>(4 * i) += v;
    }
    void addTheta(int i, Real v) {
      assert(m_dof.size() % 4 == 3);
      assert(4 * i + 3 < m_dof.size());
      m_dof(4 * i + 3) += v;
    }
  private:
    VectorXd m_dof{};
};
}
#endif // SIMCRAFT_INCLUDE_SIMCRAFT_HAIRSIM_HAIRDOF_H_