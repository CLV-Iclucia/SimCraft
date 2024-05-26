//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_TYPES_H_
#define SIMCRAFT_FEM_INCLUDE_TYPES_H_
#include <Maths/types.h>
#include <Spatify/bbox.h>
namespace fem {
using maths::Dynamic;
template <typename T, int Rows, int Cols>
using Matrix = maths::Matrix<T, Rows, Cols>;
template <typename T, int Dim>
using Vector = maths::Vector<T, Dim>;
template <typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;
using Real = double;
using Vec3d = Vector<Real, 3>;
using Vec3i = Vector<int, 3>;
using Vec4i = Vector<int, 4>;
using VecXd = Vector<Real, Dynamic>;
using spatify::BBox;
}
#endif //SIMCRAFT_FEM_INCLUDE_TYPES_H_
