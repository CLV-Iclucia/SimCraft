#ifndef SIMCRAFT_FEM_INCLUDE_TYPES_H_
#define SIMCRAFT_FEM_INCLUDE_TYPES_H_
#include <Maths/types.h>
#include <Maths/tensor.h>
#include <Spatify/bbox.h>
#include <Core/zip.h>
#include <Core/reflection.h>

namespace sim::fem {
using maths::Dynamic;
using maths::Matrix;
using maths::Vector;
template <typename T>
using SubVector = maths::VecView<T, Eigen::Dynamic>;
template <typename T>
using CSubVector = maths::CVecView<T, Eigen::Dynamic>;
using Real = double;
using Index = uint32_t;
using Vec3d = Vector<Real, 3>;
using Vec3i = Vector<int, 3>;
using Vec4i = Vector<int, 4>;
using VecXd = Vector<Real, Dynamic>;
using spatify::BBox;
using core::zip;
} // namespace sim::fem
#endif //SIMCRAFT_FEM_INCLUDE_TYPES_H_
