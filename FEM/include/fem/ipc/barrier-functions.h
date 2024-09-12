//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_BARRIER_FUNCTIONS_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_BARRIER_FUNCTIONS_H_
#include <fem/types.h>
#include <cmath>
namespace fem::ipc {
template <typename Derived>
struct BarrierFunction {
  Real d_hat{};
  [[nodiscard]] const Derived& derived() const {
    return *static_cast<const Derived*>(this);
  }
  virtual Real operator()(Real d) const {
    return derived()(d);
  }
  [[nodiscard]] virtual Real distanceGradient(Real d) const {
    return derived().distanceGradient(d);
  }
  [[nodiscard]] virtual Real distanceHessian(Real d) const {
    return derived().distanceHessian(d);
  }
};

struct LogBarrier : BarrierFunction<LogBarrier> {
  Real operator()(Real d) const override {
    assert(d > 0);
    return d >= d_hat ? 0.0 : (d - d_hat) * (d - d_hat) * log(d_hat / d);
  }
  [[nodiscard]] Real distanceGradient(Real d) const override {
    assert(d > 0);
    return d >= d_hat ? 0.0 : 2 * (d - d_hat) * log(d_hat / d) - (d - d_hat);
  }
  [[nodiscard]] Real distanceHessian(Real d) const override {
    assert(d > 0);
    return d >= d_hat ? 0.0 : 2 * log(d_hat / d) - 2 + 2 * (d - d_hat) / (d * d);
  }
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_BARRIER_FUNCTIONS_H_
