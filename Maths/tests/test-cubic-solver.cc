#include <Maths/equations.h>
#include <gtest/gtest.h>
#include <random>
#include <format>

using maths::Real;
TEST(CubicEquationSolverTest, RandomEquations) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0);

  std::cout << "Start testing cubic equation solver\n" << std::endl;
  for (int i = 0; i < 1000; ++i) {
    Real a = dis(gen);
    Real b = dis(gen);
    Real c = dis(gen);
    Real d = dis(gen);
    Real l = dis(gen);
    Real r = dis(gen);
    if (l > r) std::swap(l, r);
    auto roots = maths::clampedCubicSolve({a, b, c, d}, l, r, 1e-10);
    if (!roots.num_roots) continue;
    for (int j = 0; j < roots.num_roots; j++) {
      Real root = roots.roots[j];
      std::cout << std::format("Test {}, root = {}, f(root) = {}", i, root,
                               a * root * root * root + b * root * root + c * root + d)
                << std::endl;
      ASSERT_LE(l, root);
      ASSERT_GE(r, root);
      ASSERT_NEAR(a * root * root * root + b * root * root + c * root + d, 0,
                  1e-6);
    }
  }
}

int main() {
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}