#include <gtest/gtest.h>
#include <FluidSim/cuda/utils.h>
#include <random>
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
          ref_r[i][j][k] = ref_x[i][j][k];
        }
      }
    }
  }
  x->copyFrom(&ref_x[0][0][0]);
  y->copyFrom(&ref_y[0][0][0]);
  active->copyFrom(&ref_active[0][0][0]);

  fluid::cuda::saxpy(*x, *y, alpha, *active, resolution);
  x->copyTo(&tempArr[0][0][0]);
  std::cout << "Test begin for Saxpy" << std::endl;
  for (int i = 0; i < 10; i++){
    for (int j = 0; j < 10; j++){
      for (int k = 0; k < 10; k++){
//        printf("Test for index: %d %d %d\n", i, j, k);
//        printf("x is %f, y is %f, active is %u\n", ref_x[i][j][k], ref_y[i][j][k], ref_active[i][j][k]);
        EXPECT_NEAR(tempArr[i][j][k], ref_r[i][j][k], 1e-3);
      }
    }
  }
  std::cout << "Test end" << std::endl;
}

TEST(ScaleAndAddTest, RandomEquations) {
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
          ref_r[i][j][k] = ref_x[i][j][k];
        }
      }
    }
  }
  x->copyFrom(&ref_x[0][0][0]);
  y->copyFrom(&ref_y[0][0][0]);
  active->copyFrom(&ref_active[0][0][0]);
  fluid::cuda::scaleAndAdd(*x, *y, alpha, *active, resolution);
  x->copyTo(&tempArr[0][0][0]);
  std::cout << "Test begin for ScaleAndAdd" << std::endl;
  for (int i = 0; i < 10; i++){
    for (int j = 0; j < 10; j++){
      for (int k = 0; k < 10; k++){
        EXPECT_NEAR(tempArr[i][j][k], ref_r[i][j][k], 1e-3);
      }
    }
  }
  std::cout << "Test end" << std::endl;
}

TEST(DotProductTest, RandomEquations) {
  int3 resolution = {10, 10, 10};
  float ref_x[10][10][10], ref_y[10][10][10];
  double ref_r;
  int num_block_x = (resolution.x + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_block_y = (resolution.y + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_block_z = (resolution.z + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_blocks = num_block_x * num_block_y * num_block_z;
  uint8_t ref_active[10][10][10];
  std::unique_ptr<CudaSurface<float>> x;
  std::unique_ptr<CudaSurface<float>> y;
  std::unique_ptr<CudaSurface<uint8_t>> active;
  DeviceArray<double> device_reduce_buffer(num_blocks);
  std::vector<double> host_reduce_buffer(num_blocks);
  x = std::make_unique<CudaSurface<float>>(make_uint3(10, 10, 10));
  y = std::make_unique<CudaSurface<float>>(make_uint3(10, 10, 10));
  active = std::make_unique<CudaSurface<uint8_t>>(make_uint3(10, 10, 10));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0); // 生成范围在 -10 到 10 之间的随机数
  std::bernoulli_distribution dis_b(0.5);
  //  Generate reference
  int iters = 1000;
  std::cout << "Test begin for DotProduct: abs_error=0.05" << std::endl;
  for(int iter = 0; iter < iters; iter++){
    ref_r = 0.0;
    for (int i = 0; i < 10; i++){
      for (int j = 0; j < 10; j++){
        for (int k = 0; k < 10; k++){
          ref_x[i][j][k] = dis(gen);
          ref_y[i][j][k] = dis(gen);
          ref_active[i][j][k] = dis_b(gen);
          ref_r += ref_x[i][j][k] * ref_y[i][j][k] * float(ref_active[i][j][k]);
        }
      }
    }
    x->copyFrom(&ref_x[0][0][0]);
    y->copyFrom(&ref_y[0][0][0]);
    active->copyFrom(&ref_active[0][0][0]);
    float result = fluid::cuda::dotProduct(*x, *y, *active, device_reduce_buffer, host_reduce_buffer, resolution);
    EXPECT_NEAR(result, ref_r, 0.05);
  }
  std::cout << "Test end" << std::endl;
}

//please disable TEST(DotProductTest, RandomEquations) before testing LinfNorm
TEST(LinfNormTest, RandomEquations) {
  int3 resolution = {10, 10, 10};
  float ref_x[10][10][10];
  float ref_r;
  int num_block_x = (resolution.x + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_block_y = (resolution.y + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_block_z = (resolution.z + kThreadBlockSize3D - 1) / kThreadBlockSize3D;
  int num_blocks = num_block_x * num_block_y * num_block_z;
  uint8_t ref_active[10][10][10];
  std::unique_ptr<CudaSurface<float>> x;
  std::unique_ptr<CudaSurface<float>> y;
  std::unique_ptr<CudaSurface<uint8_t>> active;
  DeviceArray<double> device_reduce_buffer(num_blocks);
  std::vector<double> host_reduce_buffer(num_blocks);
  x = std::make_unique<CudaSurface<float>>(make_uint3(10, 10, 10));
  active = std::make_unique<CudaSurface<uint8_t>>(make_uint3(10, 10, 10));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0); // 生成范围在 -10 到 10 之间的随机数
  std::bernoulli_distribution dis_b(0.5);
  //  Generate reference
  int iters = 1000;
  std::cout << "Test begin for LinfNorm" << std::endl;
  for(int iter = 0; iter < iters; iter++){
    ref_r = 0.0;
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        for (int k = 0; k < 10; k++) {
          ref_x[i][j][k] = dis(gen);
          ref_active[i][j][k] = dis_b(gen);
          if (ref_active[i][j][k]){
            if (fabs(ref_x[i][j][k]) > ref_r){
              ref_r = fabs(ref_x[i][j][k]);
            }
          }
        }
      }
    }
    x->copyFrom(&ref_x[0][0][0]);
    active->copyFrom(&ref_active[0][0][0]);
    float result = fluid::cuda::LinfNorm(*x, *active, device_reduce_buffer, host_reduce_buffer, resolution);
    EXPECT_NEAR(result, ref_r, 1e-2);
  }
  std::cout << "Test end" << std::endl;
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}