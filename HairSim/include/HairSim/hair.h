//
// Created by creeper on 23-8-9.
//

#ifndef SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_H_
#define SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_H_
#include <HairSim/hair-dof.h>
#include <HairSim/hair-sim.h>
#include <HairSim/math-utils.h>
namespace hairsim {
struct RefConfig {
  vector<Real> length;
  vector<Vec3d> pos;
  vector<Real> theta;
  vector<Vec3d> m1;
  vector<Vec3d> m2;
  vector<Vec3d> e;
  vector<Vec3d> t;
  vector<Vec4d> m_curv_vec;
  vector<Real> m;
};

class Hair {
public:
  explicit Hair(int numVertices)
      : m_nVertices(numVertices), m_q(numVertices), m_qdot(numVertices) {
    m_vertices.reserve(numVertices);
  }
  void setModulus(Real E_, Real G_) {
    this->m_E = E_;
    this->m_G = G_;
  }
  Real E() const { return m_E; }
  Real G() const { return m_G; }
  void setRadius(Real radius) { this->m_radius = radius; }
  void setColor(const Vec3d &color) { this->m_color = color; }
  void addVertex(Index v) { m_vertices.push_back(v); }
  void addVertices(const vector<Index> &vertices) {
    m_vertices.insert(m_vertices.end(), vertices.begin(), vertices.end());
  }
  const RefConfig &referenceConfig() const { return ref; }
  // this is for initing the vertices on the hair by indices
  void init(const vector<Vec3d> &init_pos, const vector<Vec3d> &init_vel,
            const vector<Index> &indices);
  int NumVertices() const { return m_nVertices; }
  Index v(Index i) const { return m_vertices[i]; }
  Vec3d pos(Index i) const { return m_q.pos(i); }
  Real theta(Index i) const { return m_q.theta(i); }
  Real radius() const { return m_radius; }
  const HairDof &q() const { return m_q; }
  HairDof &q() { return m_q; }
  const HairDof &qdot() const { return m_qdot; }
  HairDof &qdot() { return m_qdot; }
  Vec3d edge(Index i) const { return m_q.pos(i + 1) - m_q.pos(i); }
  Vec3d tangent(Index i) const {
    return (m_q.pos(i + 1) - m_q.pos(i)).normalized();
  }
  Vec3d m1(Index i) const { return; }
  Vec3d m2(Index i) const { return; }
  Real twsitingAngle(Index i) const {
    return i > 0 ? theta(i) - theta(i - 1) + ref.m[i] : theta(i);
  }
  Real area() const { return PI * m_radius * m_radius; }
  void updateFrame();

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
    else return theta(i);
  }
  const HairDof &force() const { return m_force; }
  HairDof &force() { return m_force; }
  void addForce(Index i, const Vec3d &f) const { m_force.add2Pos(i, f); }
  void addTorsion(Index i, Real f) const { m_force.add2Theta(i, f); }
  Real vertexReferenceLength(Index i) const {
    assert(i >= 0);
    if (i > 0 && i < m_nVertices - 1) return 0.5 * (ref.length[i] + ref.length[i - 1]);
    else if (i == 0) return 0.5 * ref.length[0];
    else return 0.5 * ref.length[m_nVertices - 2];
  }

private:
  int m_nVertices = 0;
  Real rho = 1e3;                // Default density
  Real m_E = 1e9, m_G = 3.65e8;  // Default modulus
  Real m_radius = 5e-5;          // Default radius
  Vec3d m_color{0.8, 0.8, 0.8};  // Default color
  std::vector<Index> m_vertices; // Indices of the vertices on this hair
  std::vector<Vec3d> m_a1;
  std::vector<Vec3d> m_a2;
  std::vector<Vec3d> m_kb;
  std::vector<Vec4d> m_curv_vec;
  HairDof m_q;             // Generalized positions
  mutable HairDof m_force; // Generalized forces
  HairDof m_qdot;          // Generalized velocities
  RefConfig ref;
};
} // namespace hairsim

#endif // SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_H_
