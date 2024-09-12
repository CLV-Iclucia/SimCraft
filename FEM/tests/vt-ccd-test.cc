//
// Created by creeper on 9/9/24.
//
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

TEST(VertexTriangleCCD, PointMovesTowardsTriangleInside) {
  Vector<Real, 3> p1{0, 0, 1};
  Vector<Real, 3> t1{0, 0, 0};
  Vector<Real, 3> t2{1, 0, 0};
  Vector<Real, 3> t3{0, 1, 0};
  Vector<Real, 3> v1{0, 0, -1};
  Vector<Real, 3> v2{0, 0, 0};
  Vector<Real, 3> v3{0, 0, 0};
  auto query = constructQuery(p1, t1, t2, t3, v1, v2, v3, v3);
  auto contact = vtCCD(query, 1.0);
  ASSERT_TRUE(contact.has_value());
  ASSERT_NEAR(contact->t, 1.0, 1e-6);
  ASSERT_NEAR(contact->pos[0], 0.0, 1e-6);
  ASSERT_NEAR(contact->pos[1], 0.0, 1e-6);
  ASSERT_NEAR(contact->pos[2], 0.0, 1e-6);
}

TEST(VertexTriangleCCD, PointMovesTowardsTriangleOutside) {
  Vector<Real, 3> p1{0, 0, 1};
  Vector<Real, 3> t1{0, 0, 0};
  Vector<Real, 3> t2{1, 0, 0};
  Vector<Real, 3> t3{0, 1, 0};
  Vector<Real, 3> v1{0, 0, -1};
  Vector<Real, 3> v2{0, 0, 0};
  Vector<Real, 3> v3{0, 0, 0};
  auto query = constructQuery(p1, t1, t2, t3, v1, v2, v3, v3);
  auto contact = vtCCD(query, 0.5);
  ASSERT_FALSE(contact.has_value());
}

TEST(VertexTriangleCCD, CoplanarMovingParallel) {
  // the point and the triangle are on the same plane
  // they move on the plane
  // so they are always coplanar
  // but the point is not inside the triangle
  Vector<Real, 3> p1{3, 0, 0};
  Vector<Real, 3> t1{0, 0, 0};
  Vector<Real, 3> t2{1, 0, 0};
  Vector<Real, 3> t3{0, 1, 0};
  Vector<Real, 3> vp1{-1, 0, 0};
  Vector<Real, 3> v1{-1, 0, 0};
  Vector<Real, 3> v2{-1, 0, 0};
  Vector<Real, 3> v3{-1, 0, 0};
  auto query = constructQuery(p1, t1, t2, t3, vp1, v1, v2, v3);
  auto contact = vtCCD(query, 1.0);
  ASSERT_FALSE(contact.has_value());
}

int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}