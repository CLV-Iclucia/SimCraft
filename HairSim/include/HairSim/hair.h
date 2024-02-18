//
// Created by creeper on 23-8-9.
//

#ifndef SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_H_
#define SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_H_
#include <HairSim/hair-dof.h>
#include <HairSim/hair-sim.h>
#include <HairSim/utils.h>
namespace hairsim {
struct RefConfig {
  vector<Real> length;
  vector<Vec3d> pos;
  vector<Real> theta;
  vector<Vec3d> m1;
  vector<Vec3d> m2;
  vector<Vec3d> e;
  vector<Vec3d> t;
  vector<Vec4d> curv_vec;
};

struct HairParams {
  Real E = 1e9;
  Real G = 3.65e8;
  Real radius = 5e-5;
  Vec3d color = {0.8, 0.8, 0.8};
  Real rho = 1e3;
};
class Hair {
public:
  explicit Hair(int numVertices) {}
  const RefConfig &referenceConfig() const { return ref; }
  // this is for initing the vertices on the hair by indices
  void init(const vector<Vec3d> &init_pos, const vector<Vec3d> &init_vel,
            const vector<Index> &indices);
  int NumVertices() const { return m_nVertices; }
  Index v(Index i) const { return m_vertices[i]; }
  Vec3d pos(Index i) const { return m_q.pos(i); }
  Real theta(Index i) const { return m_q.theta(i); }
  Real radius() const { return m_radius; }
  [[nodiscard]] const HairDof &q() const { return m_q; }
  HairDof &q() { return m_q; }
  [[nodiscard]] const HairDof &qdot() const { return m_qdot; }
  HairDof &qdot() { return m_qdot; }
  [[nodiscard]] Vec3d edge(Index i) const { return m_q.pos(i + 1) - m_q.pos(i); }
  [[nodiscard]] Vec3d tangent(Index i) const {
    return (m_q.pos(i + 1) - m_q.pos(i)).normalized();
  }
  Vec3d m1(Index i) const { return {}; }
  Vec3d m2(Index i) const { return {}; }
  Real twsitingAngle(Index i) const {
    return i > 0 ? theta(i) - theta(i - 1) + ref.mass[i] : theta(i);
  }
  Real area() const { return PI * m_radius * m_radius; }
  void updateFrame() {}

  const Vec3d &curvatureBinormal(Index i) const { return m_kb[i]; }
  Real edgeLength(Index i) const {
    return (m_q.pos(i + 1) - m_q.pos(i)).norm();
  }
  Real kappa1(Index i, Index j) const {
    return -m2(i).dot(curvatureBinormal(j));
  }
  Real kappa2(Index i, Index j) const {
    return m1(i).dot(curvatureBinormal(j));
  }
  // (kappa1(i - 1, i), kappa1(i, i), -kappa2(i - 1, i), -kappa2(i, i))
  const Vec4d &materialCurvature(Index i) const { return m_curv_vec[i]; }
  Vec3d tangentTilde(Index i) const {
    return (tangent(i - 1) + tangent(i)) /
           (1.0 + tangent(i - 1).dot(tangent(i)));
  }
  Vec3d vertexTangent(Index i) const {
    return (tangent(i - 1) + tangent(i)) * 0.5;
  }
  Real m(int i) const {
    assert(i >= 0);
    if (i > 0) return theta(i) - theta(i - 1) + ref.theta[i];
    return theta(i);
  }
  [[nodiscard]] const HairDof &force() const { return m_force; }
  HairDof &force() { return m_force; }
  void addForce(Index i, const Vec3d &f) const { m_force.addPos(i, f); }
  void addTorsion(Index i, Real f) const { m_force.addTheta(i, f); }
  Real vertexReferenceLength(Index i) const {
    assert(i >= 0);
    if (i > 0 && i < m_nVertices - 1) return 0.5 * (ref.length[i] + ref.length[i - 1]);
    if (i == 0) return 0.5 * ref.length[0];
    return 0.5 * ref.length[m_nVertices - 2];
  }

private:
  std::vector<Vec3d> m_a1;
  std::vector<Vec3d> m_a2;
  std::vector<Vec3d> m_kb;
  std::vector<Vec4d> m_curv_vec;
  RefConfig ref;
};
} // namespace hairsim

#endif // SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_H_
