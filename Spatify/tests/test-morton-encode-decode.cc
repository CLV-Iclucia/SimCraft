//
// Created by creeper on 6/27/24.
//
#include <Spatify/mortons.h>
#include <gtest/gtest.h>
#include <random>
using namespace spatify;

static int randint() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 4096);
  return dis(gen);
}

TEST(MortonEncodeDecode21Bit, TestEncodeDecode21Bit) {
  for (int i = 0; i < 1000; i++) {
    auto x = randint();
    auto y = randint();
    auto z = randint();
    auto code = encodeMorton21bit(x, y, z);
    auto [x1, y1, z1] = decodeMorton21bit(code);
    ASSERT_EQ(x, x1);
    ASSERT_EQ(y, y1);
    ASSERT_EQ(z, z1);
  }
}

TEST(MortonEncodeDecode10Bit, TestEncodeDecode10Bit) {
  for (int x = 0; x < 1024; x++) {
    for (int y = 0; y < 1024; y++) {
      for (int z = 0; z < 1024; z++) {
        auto code = encodeMorton10bit(x, y, z);
        auto [x1, y1, z1] = decodeMorton10bit(code);
        ASSERT_EQ(x, x1);
        ASSERT_EQ(y, y1);
        ASSERT_EQ(z, z1);
      }
    }
  }
}

int main() {
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}