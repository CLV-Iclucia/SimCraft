//
// Created by creeper on 5/23/24.
//
#include <Spatify/spatial-hash.h>
#include <Spatify/sh-pruning.h>
#include <Spatify/bbox.h>
#include <iostream>
using namespace spatify;

struct ShEdge {
  using CoordType = Real;
  Real x0, y0, z0, x1, y1, z1;
  [[nodiscard]] BBox<Real, 3> bbox() const {
    return BBox<Real, 3>{{std::min(x0, x1), std::min(y0, y1), std::min(z0, z1)},
                         {std::max(x0, x1), std::max(y0, y1), std::max(z0, z1)}};
  }
  [[nodiscard]] BBoxIterator<Real> pruningIterator(const SingleCell<Real> &cell) const {
    return BBoxIterator<Real>{bbox(), cell};
  }
};

int main() {
  SpatialHash<ShEdge> sh;
  std::vector<ShEdge> accessor;
  accessor.push_back({0.0, 0.0, 0.0, 2.0, 2.0, 2.0});
  accessor.push_back({1.0, 1.0, 1.0, 2.0, 2.0, 2.0});
  accessor.push_back({2.0, 2.0, 2.0, 3.0, 3.0, 3.0});
  accessor.push_back({3.0, 3.0, 3.0, 4.0, 4.0, 4.0});
  accessor.push_back({4.0, 4.0, 4.0, 5.0, 5.0, 5.0});
  sh.build(accessor, 1.0);
  auto resolution = sh.resolution();
  for (int i = 0; i < resolution.x; i++)
    for (int j = 0; j < resolution.y; j++)
      for (int k = 0; k < resolution.z; k++) {
        std::cout << std::format("Cell ({}, {}, {}): ", i, j, k);
        sh.forProbablePrimitivesInCell(i, j, k, [&](int idx) {
          std::cout << idx << " ";
        });
        std::cout << "\n";
      }
}