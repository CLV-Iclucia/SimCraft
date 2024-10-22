//
// Created by creeper on 10/22/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_PRIMITIVES_PRIMITIVE_BASE_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_PRIMITIVES_PRIMITIVE_BASE_H_
#include <fem/types.h>
namespace fem {
struct PrimitiveBase {
  VecView<Real, Eigen::Dynamic> dofView;
  VecView<Real, Eigen::Dynamic> surfaceView;
  VecView<Real, Eigen::Dynamic> restStateView;
  VecView<Real, Eigen::Dynamic> xdotView;
};
}

#endif //SIMCRAFT_FEM_INCLUDE_FEM_PRIMITIVES_PRIMITIVE_BASE_H_
