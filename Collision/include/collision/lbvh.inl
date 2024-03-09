#include <collision/lbvh.h>
#include <Core/core.h>
#include <Core/aabb.h>

namespace collision {
using core::Vector;

static int clz(uint64_t x) {
  return __builtin_clzll(x);
}

static unsigned int expandBits(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

static inline std::string toBinary(uint64_t n) {
  // the output must be 64 bytes
  std::string r;
  for (int i = 63; i >= 0; --i) {
    r += (n & (1ULL << i)) ? '1' : '0';
  }
  return r;
}

static int lcp(uint64_t a, uint64_t b) {
  // fast longest common prefix
  return clz(a ^ b);
}

template <typename T, int Dim>
int LBVH<T, Dim>::delta(int i, int j) const {
  if (j < 0 || j > nPrs - 1) return -1;
  return lcp(mortons[i], mortons[j]);
}

template <typename T, int Dim>
uint32_t mortonCode(const Vector<T, Dim>& p) {
  if constexpr (Dim == 2) {
    unsigned int x = std::min(std::max(p[0] * 1024.0, 0.0), 1023.0);
    unsigned int y = std::min(std::max(p[1] * 1024.0, 0.0), 1023.0);
    return (expandBits(x) << 1) | expandBits(y);
  } else {
    unsigned int x = std::min(std::max(p[0] * 1024.0, 0.0), 1023.0);
    unsigned int y = std::min(std::max(p[1] * 1024.0, 0.0), 1023.0);
    unsigned int z = std::min(std::max(p[2] * 1024.0, 0.0), 1023.0);
    return (expandBits(x) << 2) | (expandBits(y) << 1) | expandBits(z);
  }
}

template <typename T, int Dim>
uint64_t encodeAABB(const AABB<T, Dim>& aabb, int idx) {
  uint32_t morton = mortonCode<T, Dim>(aabb.centre());
  return (static_cast<uint64_t>(morton) << 32) | idx;
}

template <typename T, int Dim>
void LBVH<T, Dim>::queryOverlap(std::vector<int>& overlaps,
                                const AABB<T, Dim>& queryAABB) const {
  std::array<int, 64> stack{};
  int top{};
  int nodeIdx = 0;
  if (!queryAABB.overlap(aabbs[nodeIdx]))
    return;
  while (true) {
    // it is a leaf node
    if (isLeaf(nodeIdx)) {
      overlaps.emplace_back(idx[nodeIdx - nPrs + 1]);
      if (!top) return;
      nodeIdx = stack[--top];
      continue;
    }
    int lc = lch[nodeIdx];
    int rc = rch[nodeIdx];
    bool intl = queryAABB.overlap(aabbs[lc]);
    bool intr = queryAABB.overlap(aabbs[rc]);
    if (!intl && !intr) {
      if (!top) return;
      nodeIdx = stack[--top];
      continue;
    }
    if (intl && !intr) nodeIdx = lc;
    else if (!intl) nodeIdx = rc;
    else {
      nodeIdx = lc;
      stack[top++] = rc;
    }
  }
}

template <typename T, int Dim>
int LBVH<T, Dim>::findSplit(int l, int r) const {
  if (mortons[l] == mortons[r])
    return (l + r) >> 1;
  int commonPrefix = lcp(mortons[l], mortons[r]);
  int search = l;
  int step = r - l;
  do {
    step = (step + 1) >> 1;
    if (int newSearch = search + step; newSearch < r) {
      uint64_t splitCode = mortons[newSearch];
      if (lcp(mortons[l], splitCode) > commonPrefix)
        search = newSearch;
    }
  } while (step > 1);
  return search;
}

template <typename T, int Dim>
void LBVH<T, Dim>::refit(const std::vector<AABB<T, Dim>>& updatedAABBs) {
  if (updatedAABBs.size() != nPrs) {
    std::cerr << "LBVH::refit: updatedAABBs.size() != nPrs" << std::endl;
    exit(1);
  }
  tbb::parallel_for(0, nPrs, [this, &updatedAABBs](int i) {
    mortons[i] = encodeAABB(updatedAABBs[i], i);
    idx[i] = i;
  });
  auto mortons_copy = mortons;
  tbb::parallel_sort(idx.begin(), idx.end(), [this](int a, int b) {
    return mortons[a] < mortons[b];
  });
  tbb::parallel_for(0, nPrs, [this, &mortons_copy](int i) {
    mortons[i] = mortons_copy[idx[i]];
  });
  fa[0] = -1;
  tbb::parallel_for(0, nPrs - 1, [this](int i) {
    int dir = delta(i, i + 1) > delta(i, i - 1) ? 1 : -1;
    int min_delta = delta(i, i - dir);
    int lmax = 2;
    while (delta(i, i + lmax * dir) > min_delta) lmax <<= 1;
    int len = 0;
    for (int t = lmax >> 1; t; t >>= 1) {
      if (delta(i, i + (len | t) * dir) > min_delta)
        len |= t;
    }
    int l = std::min(i, i + len * dir);
    int r = std::max(i, i + len * dir);
    int split = findSplit(l, r);
    if (l == split)
      lch[i] = nPrs - 1 + split;
    else lch[i] = split;
    if (r == split + 1)
      rch[i] = nPrs + split;
    else rch[i] = split + 1;
    fa[rch[i]] = fa[lch[i]] = i;
  });
  std::vector<std::atomic<bool>> processed(nPrs - 1);
  tbb::parallel_for(0, nPrs, [this, &updatedAABBs, &processed](int i) {
    int node_idx = nPrs + i - 1;
    aabbs[node_idx] = updatedAABBs[idx[i]];
    while (fa[node_idx] != -1) {
      int parent = fa[node_idx];
      if (!processed[parent].exchange(true)) return;
      aabbs[parent] = aabbs[lch[parent]].merge(
          aabbs[rch[parent]]);
      node_idx = parent;
    }
  });
}
}
