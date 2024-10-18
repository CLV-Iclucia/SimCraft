//
// Created by creeper on 9/9/24.
//
#include <fem/ipc/collision-detector.h>
#include <gtest/gtest.h>
// the coplanar solver takes 4 points and their velocities
// it solves for the time when the 4 points are coplanar
using namespace fem;
using namespace fem::ipc;
CCDQuery constructQuery(const Vector<Real, 3> &p1,
                        const Vector<Real, 3> &p2,
                        const Vector<Real, 3> &p3,
                        const Vector<Real, 3> &p4,
                        const Vector<Real, 3> &v1,
                        const Vector<Real, 3> &v2,
                        const Vector<Real, 3> &v3,
                        const Vector<Real, 3> &v4) {
  return CCDQuery{p1, p2 - p1, p3 - p1, p4 - p1, v1, v2 - v1, v3 - v1, v4 - v1};
}

TEST(CoplanarSolver, P1MovesTowardsThreeStaticPoint) {
  // fix p2, p3, p4 on the same plane, p1 moves towards the plane
  // p1 is the only point moving
  Vector<Real, 3> p1{0, 0, 1};
  Vector<Real, 3> p2{0, 1, 0};
  Vector<Real, 3> p3{1, 0, 0};
  Vector<Real, 3> p4{0, 0, 0};
  Vector<Real, 3> v1{0, 0, -1};
  Vector<Real, 3> v2{0, 0, 0};
  Vector<Real, 3> v3{0, 0, 0};
  Vector<Real, 3> v4{0, 0, 0};
  auto query = constructQuery(p1, p2, p3, p4, v1, v2, v3, v4);
  auto roots = solveCoplanarTime(query, 1.0);
  ASSERT_EQ(roots.numRoots, 1);
  ASSERT_NEAR(roots.roots[0], 1.0, 1e-6);
}

TEST(CoplanarSolver, P2MovesTowardsThreeStaticPoint) {
  // fix p1, p3, p4 on the same plane, p2 moves towards the plane
  // p2 is the only point moving
  // this is for guaranteeing the solver works for all points
  Vector<Real, 3> p1{0, 1, 0};
  Vector<Real, 3> p2{0, 0, 1};
  Vector<Real, 3> p3{1, 0, 0};
  Vector<Real, 3> p4{0, 0, 0};
  Vector<Real, 3> v1{0, 0, 0};
  Vector<Real, 3> v2{0, 0, -1};
  Vector<Real, 3> v3{0, 0, 0};
  Vector<Real, 3> v4{0, 0, 0};
  auto query = constructQuery(p1, p2, p3, p4, v1, v2, v3, v4);
  auto roots = solveCoplanarTime(query, 1.0);
  ASSERT_EQ(roots.numRoots, 1);
  ASSERT_NEAR(roots.roots[0], 1.0, 1e-6);
}

TEST(CoplanarSolver, P1P2MoveTowardsStaticP3P4) {
  // fix P3 and P4
  // let P1 and P2 move towards the plane at the same velocity
  Vector<Real, 3> p1{0, 0, 0.8};
  Vector<Real, 3> p2{1, 0, 0.8};
  Vector<Real, 3> p3{1, 0, 0};
  Vector<Real, 3> p4{0, 1, 0};
  Vector<Real, 3> v1{0, 0, -1};
  Vector<Real, 3> v2{0, 0, -1};
  Vector<Real, 3> v3{0, 0, 0};
  Vector<Real, 3> v4{0, 0, 0};
  auto query = constructQuery(p1, p2, p3, p4, v1, v2, v3, v4);
  auto roots = solveCoplanarTime(query, 1.0);
  ASSERT_EQ(roots.numRoots, 1);
  ASSERT_NEAR(roots.roots[0], 0.8, 1e-6);
}
int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}