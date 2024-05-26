//
// Created by creeper on 5/25/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_TENSOR_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_TENSOR_H_
#include <Maths/types.h>
namespace maths {
template<typename T, int N>
Eigen::Map<Vector<T, N * N>> vectorize(const Matrix<T, N, N> &A) {
  return Eigen::Map<Vector<T, N * N>>(A.data());
}
template <typename T, int N>
using FourthOrderTensor = Matrix<T, N * N, N * N>;

template <typename T, int N>
auto doubleContract(const FourthOrderTensor<T, N>& tensor, const Matrix<T, N, N>& A) {
  return tensor.transpose() * vectorize(A);
}
template <typename T, int N, int M>
using ThirdOrderTensor = Matrix<T, N * N, M>;
template <typename T>
Matrix<T, 3, 3> skewt(const Vector<T, 3>& v) {
  Matrix<T, 3, 3> A;
  A << 0, -v(2), v(1),
      v(2), 0, -v(0),
      -v(1), v(0), 0;
  return A;
}}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_TENSOR_H_
