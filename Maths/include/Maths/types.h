//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_TYPES_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_TYPES_H_
#include <Eigen/Eigen>
#include <Eigen/Sparse>
namespace maths {
using Eigen::Dynamic;
template <typename T, int Rows, int Cols>
using Matrix = Eigen::Matrix<T, Rows, Cols>;
template <typename T, int Dim>
using Vector = Eigen::Matrix<T, Dim, 1>;
template <typename T, int Dim>
using VecView = Eigen::Map<Vector<T, Dim>>;
template <typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;
using Real = double;
}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_TYPES_H_
