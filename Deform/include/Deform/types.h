//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_DEFORM_TYPES_H_
#define SIMCRAFT_DEFORM_TYPES_H_
#include <Maths/types.h>
#include <Maths/tensor.h>
namespace deform {
template <typename T, int Dim>
using Matrix = maths::Matrix<T, Dim, Dim>;
using maths::Vector;
using maths::FourthOrderTensor;
using maths::Real;
}
#endif //SIMCRAFT_DEFORM_TYPES_H_
