//
// Created by creeper on 23-8-14.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_TENSOR_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_TENSOR_H_
#include <Core/core.h>
#include <array>
namespace core {
// TODO: this may be implemented and extended in the future if needed
template <typename T, int Order, int... Dims> class Tensor {};

// For now, all I need is a fourth order tensor of shape 3x3x3x3 or 2x2x2x2
// this will be useful for computing Hessian when doing implicit integration
template <typename T, int Dim> class Tensor<T, 4, Dim, Dim, Dim, Dim> {
  static_assert(std::is_scalar<T>::value,
                "Tensor element type must be a scalar");

public:
  // implement all the methods
  Tensor() = default;
  inline T &operator()(int i, int j, int k, int l) {
    return m_data[i * Dim * Dim * Dim + j * Dim * Dim + k * Dim + l];
  }
  inline const T &operator()(int i, int j, int k, int l) const {
    return m_data[i * Dim * Dim * Dim + j * Dim * Dim + k * Dim + l];
  }
  inline T &operator()(const Vector<int, 4> &idx) {
    return m_data[idx[0] * Dim * Dim * Dim + idx[1] * Dim * Dim + idx[2] * Dim +
                  idx[3]];
  }
  inline const T &operator()(const Vector<int, 4> &idx) const {
    return m_data[idx[0] * Dim * Dim * Dim + idx[1] * Dim * Dim + idx[2] * Dim +
                  idx[3]];
  }
  // some arithmetic operations
  inline Tensor<T, 4, Dim, Dim, Dim, Dim>
  operator+(const Tensor<T, 4, Dim, Dim, Dim, Dim> &rhs) const {
    Tensor<T, 4, Dim, Dim, Dim, Dim> ret;
    for (int i = 0; i < Dim * Dim * Dim * Dim; i++)
      ret.m_data[i] = m_data[i] + rhs.m_data[i];
  }
  inline Tensor<T, 4, Dim, Dim, Dim, Dim>
  operator-(const Tensor<T, 4, Dim, Dim, Dim, Dim> &rhs) const {
    Tensor<T, 4, Dim, Dim, Dim, Dim> ret;
    for (int i = 0; i < Dim * Dim * Dim * Dim; i++)
      ret.m_data[i] = m_data[i] - rhs.m_data[i];
  }
private:
  // use one std::array to store all the data
  std::array<T, Dim * Dim * Dim * Dim> m_data;
};
template <typename T, int Dim>
inline Matrix<T, Dim> tensorProduct(const Vector<T, Dim> &a,
                                    const Vector<T, Dim> &b) {
  Matrix<T, Dim> ret;
  for (int i = 0; i < Dim; i++)
    for (int j = 0; j < Dim; j++)
      ret(i, j) = a[i] * b[j];
  return ret;
}
template <typename T, int Dim>
using FourthOrderTensor = Tensor<T, 4, Dim, Dim, Dim, Dim>;
} // namespace core
#endif // SIMCRAFT_CORE_INCLUDE_CORE_TENSOR_H_
