#pragma once

#include <Core/rtti.h>

#include <cstddef>

namespace ksk::runtime {

using Real = double;

using SubsystemIdx = int;
using PointIdx = int;

using ObjectId = int;
using ObjectTypeId = RF::TypeId;

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
