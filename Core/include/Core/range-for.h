//
// Created by creeper on 23-8-19.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_RANGE_ACCESSOR_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_RANGE_ACCESSOR_H_
#include <Core/core.h>
namespace core {
// RangeAccessor enables the user to perform on indices
template <int Dim> using Range = std::tuple<Vector<int, Dim>, Vector<int, Dim>>;

// another forRange
// the difference is that this one takes a function that takes a vector
template <typename Func, int Dim>
inline void forRange(const Range<Dim> &range, Func &&func) {
  const Vector<int, Dim> &start = std::get<0>(range);
  const Vector<int, Dim> &end = std::get<1>(range);
  if constexpr (Dim == 2) {
    for (int i = start[0]; i < end[0]; i++)
      for (int j = start[1]; j < end[1]; j++)
        func(Vector<int, 2>{i, j});
  }
  if constexpr (Dim == 3) {
    for (int i = start[0]; i < end[0]; i++)
      for (int j = start[1]; j < end[1]; j++)
        for (int k = start[2]; k < end[2]; k++)
          func(Vector<int, 3>{i, j, k});
  }
}

template <typename Func>
inline void forRange(int i_begin, int i_end, int j_begin, int j_end,
                     Func &&func) {
  for (int i = i_begin; i < i_end; i++)
    for (int j = j_begin; j < j_end; j++)
      func(i, j);
}
// a 3D version
template <typename Func>
inline void forRange(int i_begin, int i_end, int j_begin, int j_end,
                     int k_begin, int k_end, Func &&func) {
  for (int i = i_begin; i < i_end; i++)
    for (int j = j_begin; j < j_end; j++)
      for (int k = k_begin; k < k_end; k++)
        func(i, j, k);
}
// another kind of for range
} // namespace core
#endif // SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_RANGE_ACCESSOR_H_
