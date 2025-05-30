//
// Created by creeper on 10/24/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_BOUNDARY_CONDITIONS_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_BOUNDARY_CONDITIONS_H_

#include <variant>
#include <functional>
#include <fem/types.h>
#include <Core/json.h>
namespace sim::fem {
struct DirichletBoundaryCondition {
  Index element{};
  uint8_t dofIdx{};
  std::variant<Real, std::function<Real(Real)>> constraint;
  [[nodiscard]] Real value(Real t) const {
    if (std::holds_alternative<Real>(constraint))
      return std::get<Real>(constraint);
    return std::get<std::function<Real(Real)>>(constraint)(t);
  }
};

}// namespace fem
#endif //SIMCRAFT_FEM_INCLUDE_FEM_BOUNDARY_CONDITIONS_H_
