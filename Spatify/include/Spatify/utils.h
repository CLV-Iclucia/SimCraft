//
// Created by creeper on 6/8/24.
//

#ifndef SIMCRAFT_SPATIFY_INCLUDE_SPATIFY_UTILS_H_
#define SIMCRAFT_SPATIFY_INCLUDE_SPATIFY_UTILS_H_
#include <Spatify/types.h>
namespace spatify {
template<typename T, int Dim>
Vector<T, Dim> cwiseMin(const Vector<T, Dim> &a, const Vector<T, Dim> &b) {
  Vector<T, Dim> result;
  for (int i = 0; i < Dim; i++)
    result[i] = std::min(a[i], b[i]);
  return result;
}

template<typename T, int Dim>
Vector<T, Dim> cwiseMax(const Vector<T, Dim> &a, const Vector<T, Dim> &b) {
  Vector<T, Dim> result;
  for (int i = 0; i < Dim; i++)
    result[i] = std::max(a[i], b[i]);
  return result;
}
}
#endif //SIMCRAFT_SPATIFY_INCLUDE_SPATIFY_UTILS_H_
