#include <gtest/gtest.h>
#include <FluidSim/cuda/utils.h>
#include <random>

using namespace fluid::cuda;

TEST(SaxpyTest, RandomEquations) {
  int3 resolution = {10, 10, 10};
  float alpha = 0.5;
  float ref_x[10][10][10], ref_y[10][10][10], ref_r[10][10][10];
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
  cudaMemcpy(ref_x, x->cuda_array, sizeof(ref_x), cudaMemcpyHostToDevice);
  cudaMemcpy(ref_y, y->cuda_array, sizeof(ref_y), cudaMemcpyHostToDevice);
  cudaMemcpy(ref_active, active->cuda_array, sizeof(ref_active), cudaMemcpyHostToDevice);
  fluid::cuda::saxpy(*x, *y, alpha, *active, resolution);
  cudaMemcpy(ref_x, x->cuda_array, sizeof(ref_x), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++){
    for (int j = 0; j < 10; j++){
      for (int k = 0; k < 10; k++){
        EXPECT_EQ(ref_x[i][j][k], ref_r[i][j][k]);
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}