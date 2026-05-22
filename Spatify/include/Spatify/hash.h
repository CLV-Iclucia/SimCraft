//
// Created by creeper on 12/14/23.
//

#ifndef HASH_H
#define HASH_H

#include <Spatify/types.h>

#include <functional>

namespace spatify {
enum class Axis {
  X,
  Y,
  Z
};

struct LinearHashXYZ {
  LinearHashXYZ() = default;
  LinearHashXYZ(int sizex, int sizey, int sizez)
    : sizex(sizex), sizey(sizey), sizez(sizez) {}
  int operator()(int x, int y, int z) const {
    assert(x >= 0 && x < sizex && y >= 0 && y < sizey && z >= 0 && z < sizez);
    return z + y * sizez + x * sizey * sizez;
  }
  int sizex{}, sizey{}, sizez{};
};

template <typename T>
concept SpatialHashFunction3D = requires(const T& func, int x, int y, int z) {
  { func(x, y, z) } -> std::convertible_to<int>;
};
}

#endif //HASH_H