//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_BARRIER_FUNCTIONS_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_BARRIER_FUNCTIONS_H_
#include <fem/types.h>
#include <cmath>
namespace fem::ipc {
struct BarrierFunction {
  Real d_hat{1e-5};
  Real operator()(Real d) const {
    assert(d > 0);
    return d >= d_hat ? 0.0 : (d - d_hat) * (d - d_hat) * log(d_hat / d);
  }
  Real distanceGradient(Real d) const {
    assert(d > 0);
    return d >= d_hat ? 0.0 : 2 * (d - d_hat) * log(d_hat / d) - (d - d_hat);
  }
  Real distanceHessian(Real d) const {
    assert(d > 0);
    return d >= d_hat ? 0.0 : 2 * log(d_hat / d) - 2 + 2 * (d - d_hat) / (d * d);
  }
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_BARRIER_FUNCTIONS_H_
