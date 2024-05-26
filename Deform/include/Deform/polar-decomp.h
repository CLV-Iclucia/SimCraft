//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_DEFORM_INCLUDE_DEFORM_POLAR_DECOMP_H_
#define SIMCRAFT_DEFORM_INCLUDE_DEFORM_POLAR_DECOMP_H_
#include <Deform/types.h>
namespace deform {
enum PolarDecompositionOptions {
  ComputeFullR = 0x01,
  ComputeFullS = 0x02
};
template<typename T, int Option = ComputeFullR | ComputeFullS>
void polarDecomposition(const Matrix<T, 3> &F, Matrix<T, 3> &R, Matrix<T, 3> &S) {
  Eigen::JacobiSVD<Matrix<T, 3>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (Option & ComputeFullR)
    R = svd.matrixU() * svd.matrixV().transpose();
  if (Option & ComputeFullS)
    S = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
}
}
#endif //SIMCRAFT_DEFORM_INCLUDE_DEFORM_POLAR_DECOMP_H_
