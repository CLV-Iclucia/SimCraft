//
// Created by creeper on 4/25/24.
//
#include <FluidSim/cuda/mgpcg.cuh>

namespace fluid::cuda {
__constant__ float kTransferWights[4][4][4];
// 0 stands for solid, 1 stands for fluid
__global__ void PrecomputeDownSampleKernel(CudaSurfaceAccessor<uint8_t> surf,
                                           CudaSurfaceAccessor<uint8_t> surf_nxt, uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  uint8_t val_1 = surf.read<cudaBoundaryModeZero>(x * 2, y * 2, z * 2);
  uint8_t val_2 = surf.read<cudaBoundaryModeZero>(x * 2 + 1, y * 2, z * 2);
  uint8_t val_3 = surf.read<cudaBoundaryModeZero>(x * 2, y * 2 + 1, z * 2);
  uint8_t val_4 = surf.read<cudaBoundaryModeZero>(x * 2 + 1, y * 2 + 1, z * 2);
  uint8_t val_5 = surf.read<cudaBoundaryModeZero>(x * 2, y * 2, z * 2 + 1);
  uint8_t val_6 = surf.read<cudaBoundaryModeZero>(x * 2 + 1, y * 2, z * 2 + 1);
  uint8_t val_7 = surf.read<cudaBoundaryModeZero>(x * 2, y * 2 + 1, z * 2 + 1);
  uint8_t val_8 = surf.read<cudaBoundaryModeZero>(x * 2 + 1, y * 2 + 1, z * 2 + 1);
  surf_nxt.write(val_1 && val_2 && val_3 && val_4 && val_5 && val_6 && val_7 && val_8, x, y, z);
}
__global__ void RestrictKernel(CudaSurfaceAccessor<float> u,
                               CudaSurfaceAccessor<float> uc, uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  float sum = 0.0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++)
        sum += kTransferWights[i][j][k] * u.read<cudaBoundaryModeZero>(x * 2 + i - 1, y * 2 + j - 1, z * 2 + k - 1);
  uc.write(sum, x, y, z);
}
__global__ void ProlongateKernel(CudaSurfaceAccessor<float> uc,
                                 CudaSurfaceAccessor<float> u, uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);

}
void prepareWeights() {
  float weights[4][4][4] = {

  };
  cudaMemcpyToSymbol(kTransferWights, weights, sizeof(weights));
}
}