//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_LINALG_UTILS_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_LINALG_UTILS_H_
#include <Maths/types.h>
#include <cmath>
namespace maths {
template <typename T>
Matrix<T, 3, 3> skewt(const Vector<T, 3>& v) {
  Matrix<T, 3, 3> A;
  A << 0, -v(2), v(1),
      v(2), 0, -v(0),
      -v(1), v(0), 0;
  return A;
}

template <typename T>
T mixedProduct(const Vector<T, 3>& a, const Vector<T, 3>& b, const Vector<T, 3>& c) {
  return a.cross(b).dot(c);
}

template <typename T>
Matrix<T, 3, 3> constructFrame(const Vector<T, 3>& n) {
  Vector<T, 3> x = n;
  if (std::abs(x(0)) < std::abs(x(1)))
    x(0) = 1;
  else
    x(1) = 1;
  x = (x - n.dot(x) * n).normalized();
  Vector<T, 3> y = n.cross(x);
  Matrix<T, 3, 3> R;
  R << x, y, n;
  return R;
}
}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_LINALG_UTILS_H_
