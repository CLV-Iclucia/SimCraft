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
struct CCDQuery {
  Vector<Real, 3> x1;
  Vector<Real, 3> x2_x1;
  Vector<Real, 3> x3_x1;
  Vector<Real, 3> x4_x1;
  Vector<Real, 3> u1;
  Vector<Real, 3> u2_u1;
  Vector<Real, 3> u3_u1;
  Vector<Real, 3> u4_u1;
};
struct Contact {
  Vector<Real, 3> pos;
  Real t;
};
std::optional<Contact> eeCCD(const CCDQuery &query, Real toi);
std::optional<Contact> vtCCD(const CCDQuery &query, Real toi);
class System;
class CollisionDetector {
 public:
  void detect(const System &system, const VecXd& p, Real dt);
 private:
  std::unique_ptr<spatify::LBVH<Real>> lbvh{};
  tbb::concurrent_vector<Contact> active_contacts{};
};
struct CollisionResolver {
  std::unique_ptr<CollisionDetector> collision_detector;
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_COLLISION_RESOLVER_H_
