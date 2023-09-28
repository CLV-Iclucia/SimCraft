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
private:
  int m_nVertices = 0;
  VectorXd m_dof{};
public:
  HairDof() = default;
  explicit HairDof(int nVertices) : m_nVertices(nVertices) {
    m_dof.resize(4 * nVertices - 1);
  }

  void init(int nVertices) {
    m_nVertices = nVertices;
    m_dof.resize(4 * nVertices - 1);
  }

  void setNumVertices(int N) { m_nVertices = N; }
  int getNumVertices() const { return m_nVertices; }
  Eigen::Map<VectorXd> thetas() {
    return Eigen::Map<VectorXd>(m_dof.data() + 3, m_nVertices - 1,
                               Eigen::Stride<1, 4>());
  }
  Real& theta(int i) {
    assert(i < m_nVertices - 1 && i >= 0);
    return m_dof(4 * i + 3);
  }
  Real theta(int i) const {
    assert(i < m_nVertices - 1 && i >= 0);
    return m_dof(4 * i + 3);
  }
  Vec3d pos(int i) const {
    assert(i < m_nVertices);
    return m_dof.segment<3>(4 * i);
  }
  Eigen::Ref<Vec3d> pos(int i) {
    assert(i < m_nVertices);
    return Eigen::Ref<Vec3d>(m_dof.data() + 4 * i);
  }
  Real operator()(int i) const {
    assert(i < m_dof.size());
    return m_dof[i];
  }
  Real& operator()(int i) {
    assert(i < m_dof.size());
    return m_dof[i];
  }
  void add2Pos(int i, const Vec3d &v) {
    assert(i < m_nVertices);
    m_dof.segment<3>(4 * i) += v;
  }
  void add2Theta(int i, Real v) {
    assert(i < m_nVertices - 1);
    m_dof(4 * i + 3) += v;
  }
};

}
#endif // SIMCRAFT_INCLUDE_SIMCRAFT_HAIRSIM_HAIRDOF_H_
