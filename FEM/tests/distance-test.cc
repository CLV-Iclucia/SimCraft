//
// Created by creeper on 9/15/24.
//
#include <fem/ipc/distances.h>
#include <gtest/gtest.h>
using namespace fem;

TEST(PointPointDistanceTest, Test1) {
  Vec3d p1 = {0, 0, 0};
  Vec3d p2 = {1, 1, 1};
  Real distance = ipc::distanceSqrPointPoint(p1, p2);
  EXPECT_NEAR(distance, 3, 1e-6);
}

TEST(PointPointDistanceTest, Test2) {
  Vec3d p1 = {0, 0, 0};
  Vec3d p2 = {0, 0, 0};
  Real distance = ipc::distanceSqrPointPoint(p1, p2);
  EXPECT_NEAR(distance, 0, 1e-6);
}

TEST(PointPointDistanceTest, Test3) {
  Vec3d p1 = {0, 0, 0};
  Vec3d p2 = {1, 0, 0};
  Real distance = ipc::distanceSqrPointPoint(p1, p2);
  EXPECT_NEAR(distance, 1, 1e-6);
}

TEST(PointLineDistanceTest, Test1) {
  Vec3d p = {0, 0, 0};
  Vec3d p1 = {1, 0, 0};
  Vec3d p2 = {0, 1, 0};
  Real distance = ipc::distanceSqrPointLine(p, p1, p2);
  EXPECT_NEAR(distance, 0.5, 1e-6);
}

TEST(PointLineDistanceTest, Test2) {
  Vec3d p = {0, 3, 0};
  Vec3d p1 = {1, 0, 0};
  Vec3d p2 = {0, 1, 0};
  Real distance = ipc::distanceSqrPointLine(p, p1, p2);
  EXPECT_NEAR(distance, 2, 1e-6);
}

TEST(PointPlaneDistanceTest, Test1) {
  Vec3d p = {0, 0, 0};
  Vec3d p1 = {1, 0, 0};
  Vec3d p2 = {0, 1, 0};
  Vec3d p3 = {0, 0, 1};
  Real distance = ipc::distanceSqrPointPlane(p, p1, p2, p3);
  EXPECT_NEAR(distance, 1.0 / 3.0, 1e-6);
}

TEST(LineLineDistanceTest, Test1) {
  Vec3d p1 = {0, 0, 0};
  Vec3d p2 = {1, 0, 0};
  Vec3d q1 = {0, 1, 0};
  Vec3d q2 = {1, 1, 0};
  Real distance = ipc::distanceSqrLineLine(p1, p2, q1, q2);
  EXPECT_NEAR(distance, 1, 1e-6);
}

int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}