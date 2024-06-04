//
// Created by creeper on 5/25/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_SVD_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_SVD_H_
#include <Maths/types.h>
namespace maths {
template <typename T>
void svd3x3(const Matrix<T, 3, 3> &A, Matrix<T, 3, 3> &U, Vector<T, 3> &S, Matrix<T, 3, 3> &V) {
  Eigen::HouseholderQR<Matrix<T, 3, 3>> qr(A);
  U = qr.householderQ();
  auto R = qr.matrixQR().template triangularView<Eigen::Upper>();
  S = R.diagonal();
  V = R.array().colwise() / S.array();
}
}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_SVD_H_
