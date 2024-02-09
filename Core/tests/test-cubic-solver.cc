#include <Core/utils.h>
#include <gtest/gtest.h>
#include <random>
#include <format>

using core::Real;
TEST(CubicEquationSolverTest, RandomEquations) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0); // 生成范围在 -10 到 10 之间的随机数

  std::cout << "Start testing cubic equation solver\n" << std::endl;
  for (int i = 0; i < 1000; ++i) {
    Real a = dis(gen);
    Real b = dis(gen);
    Real c = dis(gen);
    Real d = dis(gen);
    Real l = dis(gen);
    Real r = dis(gen);
    if (l > r) std::swap(l, r);
    Real root = core::cubicSolve(a, b, c, d, l, r, 1e-6);
    if (std::isnan(root))
      continue;
    std::cout << std::format("Test {}, root = {}, f(root) = {}", i, root,
                             a * core::cubic(root) + b * core::sqr(root) + c *
                             root + d) << std::endl;
    ASSERT_NEAR(a * core::cubic(root) + b * core::sqr(root) + c * root + d, 0,
                1e-6);
  }
}

int main() {
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}