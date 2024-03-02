//
// Created by creeper on 2/20/24.
//

#ifndef COLLISION_INCLUDE_COLLISION_CPU_LBVH_H
#define COLLISION_INCLUDE_COLLISION_CPU_LBVH_H
#include <Core/properties.h>
#include <Core/aabb.h>
#include <tbb/tbb.h>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <array>
#include <vector>

namespace collision {
using core::AABB;

template <typename T, int Dim>
uint64_t encodeAABB(const AABB<T, Dim>& aabb, int idx);

template <typename T, int Dim>
class LBVH : core::NonCopyable {
public:
  explicit LBVH(int n) : nPrs(n), aabbs((n << 1) - 1), fa((n << 1) - 1),
                         mortons(n), lch(n - 1), rch(n - 1), idx(n) {
  }

  [[nodiscard]] bool isLeaf(int idx) const {
    return idx >= nPrs - 1;
  }

  void refit(const std::vector<AABB<T, Dim>>& updatedAABBs);
  void queryOverlap(std::vector<int>& overlaps, const AABB<T, Dim>& queryAABB) const;

  AABB<T, Dim> aabb(int idx) const {
    return aabbs[idx];
  }

  int lchild(int idx) const {
    return lch[idx];
  }

  int rchild(int idx) const {
    return rch[idx];
  }

private:
  [[nodiscard]] int findSplit(int l, int r) const;
  [[nodiscard]] int delta(int i, int j) const;
  int nPrs;
  std::vector<AABB<T, Dim>> aabbs{};
  std::vector<uint64_t> mortons;
  std::vector<int> fa;
  std::vector<int> lch;
  std::vector<int> rch;
  std::vector<int> idx;
};
}

#include "lbvh.inl"
#endif //COLLISION_INCLUDE_COLLISION_CPU_LBVH_H