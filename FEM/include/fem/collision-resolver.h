//
// Created by creeper on 5/24/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_COLLISION_RESOLVER_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_COLLISION_RESOLVER_H_
#include <Spatify/lbvh.h>
#include <memory>
#include <fem/types.h>
#include <fem/tet-mesh.h>
#include <Core/utils.h>
namespace fem {
struct VertexTriangleCCDQuery {
  Vector<Real, 3> vertex_pos;
  Vector<Real, 3> vertex_vel;
  Matrix<Real, 3, 3> triangle_pos;
  Matrix<Real, 3, 3> triangle_vel;
};

struct EdgeEdgeCCDQuery {
  Matrix<Real, 3, 2> ea_pos;
  Matrix<Real, 3, 2> ea_vel;
  Matrix<Real, 3, 2> eb_pos;
  Matrix<Real, 3, 2> eb_vel;
};
struct CollisionInfo {
  Vector<Real, 3> pos;
  Matrix<Real, 3, 3> frame;
  Real t;
};
std::optional<CollisionInfo> eeCCD(const EdgeEdgeCCDQuery& query, Real dt);
std::optional<CollisionInfo> vtCCD(const VertexTriangleCCDQuery& query, Real dt);
class System;
class CollisionDetector {
 public:
  void detect(const System& system, Real dt);
 private:
  std::unique_ptr<spatify::LBVH<Real>> lbvh{};
  tbb::concurrent_vector<CollisionInfo> collisions{};
};
struct CollisionResolver {
  std::unique_ptr<CollisionDetector> collision_detector;
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_COLLISION_RESOLVER_H_
