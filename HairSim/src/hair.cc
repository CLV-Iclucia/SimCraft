//
// Created by creeper on 23-8-9.
//
#include <HairSim/hair.h>
namespace hairsim {
void Hair::init(const vector<Vec3d> &init_pos, const vector<Vec3d> &init_vel,
                const vector<Index> &indices) {
  m_vertices = indices;
  for (int i = 0; i < m_nVertices; i++) {
    m_q.pos(i) = init_pos[indices[i]];
    m_qdot.pos(i) = init_vel[indices[i]];
  }
  // next compute the reference configuration
  ref.length.resize(m_nVertices - 1);
  ref.pos.resize(m_nVertices);
  ref.theta.resize(m_nVertices - 1);
  ref.m1.resize(m_nVertices - 1);
  ref.m2.resize(m_nVertices - 1);
  ref.e.resize(m_nVertices - 1);
  ref.t.resize(m_nVertices - 1);
  ref.mass.resize(m_nVertices - 1);
  for (int i = 0; i < m_nVertices - 1; i++) {
    ref.length[i] = (m_q.pos(i + 1) - m_q.pos(i)).norm();
    ref.pos[i] = (m_q.pos(i + 1) + m_q.pos(i)) / 2;
    ref.theta[i] = atan2(m_q.pos(i + 1)(1) - m_q.pos(i)(1),
                         m_q.pos(i + 1)(0) - m_q.pos(i)(0));
    ref.m1[i] = Vec3d(cos(ref.theta[i]), sin(ref.theta[i]), 0);
    ref.m2[i] = Vec3d(-sin(ref.theta[i]), cos(ref.theta[i]), 0);
    ref.e[i] = (m_q.pos(i + 1) - m_q.pos(i)) / ref.length[i];
    ref.t[i] = ref.e[i].normalized();
  }
  // initialize degrees of freedom
  m_q.theta(0) = 0;
  m_qdot.theta(0) = 0;
  for (int i = 0; i < m_nVertices - 1; i++) {
    m_q.theta(i + 1) = ref.theta[i];
    m_qdot.theta(i + 1) = 0;
  }
}
} // namespace hairsim
