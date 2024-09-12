//
// Created by creeper on 9/9/24.
//
#include <fem/ipc/collision-detector.h>
#include <gtest/gtest.h>

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

TEST(EdgeEdgeCCD, TwoVerticalEdgesIntersect) {
  // two vertical edges moving towards each other and intersect at time 0.5
  Vector<Real, 3> p1{0, 0, 0};
  Vector<Real, 3> p2{0, 0, 1};
  Vector<Real, 3> p3{1, 0, 0};
  Vector<Real, 3> p4{1, 1, 0};
  Vector<Real, 3> v1{1, 0, 0};
  Vector<Real, 3> v2{1, 0, 0};
  Vector<Real, 3> v3{-1, 0, 0};
  Vector<Real, 3> v4{-1, 0, 0};
  auto query = constructQuery(p1, p2, p3, p4, v1, v2, v3, v4);
  auto contact = eeCCD(query, 1.0);
  ASSERT_TRUE(contact.has_value());
  ASSERT_NEAR(contact->t, 0.5, 1e-6);
  ASSERT_NEAR(contact->pos[0], 0.5, 1e-6);
  ASSERT_NEAR(contact->pos[1], 0.0, 1e-6);
  ASSERT_NEAR(contact->pos[2], 0.0, 1e-6);
}

TEST(EdgeEdgeCCD, TwoVerticalEdgesNotIntersect) {
  // two vertical edges moving towards each other and coplanar at time 0.5, but no intersection
  Vector<Real, 3> p1{0, 0, 0};
  Vector<Real, 3> p2{0, 0, 1};
  Vector<Real, 3> p3{1, 10, 0};
  Vector<Real, 3> p4{1, 11, 0};
  Vector<Real, 3> v1{1, 0, 0};
  Vector<Real, 3> v2{1, 0, 0};
  Vector<Real, 3> v3{-1, 0, 0};
  Vector<Real, 3> v4{-1, 0, 0};
  auto query = constructQuery(p1, p2, p3, p4, v1, v2, v3, v4);
  auto contact = eeCCD(query, 1.0);
  ASSERT_FALSE(contact.has_value());
}

int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}