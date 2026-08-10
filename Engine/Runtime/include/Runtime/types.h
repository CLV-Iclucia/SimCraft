#pragma once

#include <Core/rtti.h>

#include <cstddef>

namespace ksk::runtime {

using Real = double;

using SubsystemIdx = int;
using PointIdx = int;
using EdgeIdx = int;
using TriangleIdx = int;
using PhysicalMaterialId = int;

using ObjectId = int;
using ObjectTypeId = RF::TypeId;

// Sparse matrix entry shared by the global solver and the per-subsystem solver
// APIs (row/col are DOF-scalar indices into the global DOF space).
struct SparseMatrixEntry {
  int row = 0;
  int col = 0;
  double value = 0.0;
};

inline constexpr PhysicalMaterialId kDefaultPhysicalMaterial = 0;

template <typename T>
constexpr ObjectTypeId elementTypeId() noexcept
{
  return RF::TypeOf<T>();
}

struct SubsystemRange {
  int first = 0;
  int count = 0;
};

}  // namespace ksk::runtime
