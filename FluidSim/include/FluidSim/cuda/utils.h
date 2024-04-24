//
// Created by creeper on 24-3-22.
//

#ifndef SIM_CRAFT_CUDA_UTILS_H
#define SIM_CRAFT_CUDA_UTILS_H
#include <Core/cuda-utils.h>

namespace fluid::cuda {
CUDA_FORCEINLINE CUDA_CALLABLE double3 getCellCentre(int x, int y, int z) {
  return make_double3(x + 0.5, y + 0.5, z + 0.5);
}

CUDA_FORCEINLINE CUDA_CALLABLE double3 getFaceCentre(
    int x, int y, int z, int axis) {
  return make_double3(x + (axis == 0), y + (axis == 1), z + (axis == 2));
}

CUDA_FORCEINLINE CUDA_CALLABLE int3 getCellIndex(double3 p, double h) {
  return make_int3(static_cast<int>(p.x / h), static_cast<int>(p.y / h),
                   static_cast<int>(p.z / h));
}

CUDA_FORCEINLINE CUDA_DEVICE double3 grad(CudaTextureAccessor<float> field,
                                          const float3& pos, int3 resolution, Real h) {

}
}
#endif //SIM_CRAFT_CUDA_UTILS_H