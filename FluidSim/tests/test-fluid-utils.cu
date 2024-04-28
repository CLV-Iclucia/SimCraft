#include <gtest/gtest.h>
#include <FluidSim/cuda/utils.h>
#include <random>
using uint = unsigned int;
using namespace fluid::cuda;

TEST(SaxpyTest, RandomEquations) {
  int3 resolution = {10, 10, 10};
  float alpha = 10;
  float ref_x[10][10][10], ref_y[10][10][10], ref_r[10][10][10], tempArr[10][10][10];
  uint8_t ref_active[10][10][10];
  std::unique_ptr<CudaSurface<float>> x;
  std::unique_ptr<CudaSurface<float>> y;
  std::unique_ptr<CudaSurface<uint8_t>> active;
  x = std::make_unique<CudaSurface<float>>(make_uint3(10, 10, 10));
  y = std::make_unique<CudaSurface<float>>(make_uint3(10, 10, 10));
  active = std::make_unique<CudaSurface<uint8_t>>(make_uint3(10, 10, 10));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0); // 生成范围在 -10 到 10 之间的随机数
  std::bernoulli_distribution dis_b(0.5);
  //  Generate reference
  for (int i = 0; i < 10; i++){
    for (int j = 0; j < 10; j++){
      for (int k = 0; k < 10; k++){
        ref_x[i][j][k] = dis(gen);
        tempArr[i][j][k] = ref_x[i][j][k];
        ref_y[i][j][k] = dis(gen);
        ref_active[i][j][k] = dis_b(gen);
        if (ref_active[i][j][k]){
          ref_r[i][j][k] = ref_x[i][j][k] + alpha * ref_y[i][j][k];
        }else{
          ref_r[i][j][k] = 0.0;
        }
      }
    }
  }
  x->copyFrom(&ref_x[0][0][0]);
  y->copyFrom(&ref_y[0][0][0]);
  active->copyFrom(&ref_active[0][0][0]);
//  检查用代码
  float temp = ref_x[0][0][0];
  x->copyTo(&ref_x[0][0][0]);
  printf("cuda:%f host:%f\n", ref_x[0][0][0], temp);

  fluid::cuda::saxpy(*x, *y, alpha, *active, resolution);
  x->copyTo(&ref_x[0][0][0]);
  printf("cuda:%f host:%f\n", ref_x[0][0][0], temp);
//  cudaMemcpy(ref_x, x->cuda_array, sizeof(ref_x), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++){
    for (int j = 0; j < 10; j++){
      for (int k = 0; k < 10; k++){
        printf("Test for index: %d %d %d\n", i, j, k);
        printf("x is %f, y is %f, active is %u\n", tempArr[i][j][k], ref_y[i][j][k], ref_active[i][j][k]);
        EXPECT_NEAR(ref_x[i][j][k], ref_r[i][j][k], 1e-3);
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}