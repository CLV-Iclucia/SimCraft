//
// Created by creeper on 10/21/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_PRIMITIVE_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_PRIMITIVE_H_
#include <memory>
#include <variant>
#include <fem/primitives/elastic-tri-mesh.h>
#include <fem/primitives/elastic-tet-mesh.h>
#include <fem/primitives/elastic-string.h>
#include <fem/primitives/elastic-sphere.h>
#include <fem/primitives/custom-primitive.h>
#include <Maths/sparse-matrix-builder.h>
namespace fem {
struct Primitive {
  std::variant<ElasticTetMesh> data;
  void assembleEnergyGradient(VecXd& globalGrad) const {
    std::visit([&](auto&& arg) {
      arg.assembleEnergyGradient(globalGrad);
    }, data);
  }
  size_t dofDim() const {
    return std::visit([](auto&& arg) {
      return arg.dofDim();
    }, data);
  }
  void assembleEnergyHessian(maths::SparseMatrixBuilder<Real>& globalHessian) const {
    std::visit([&](auto&& arg) {
      arg.assembleEnergyHessian(globalHessian);
    }, data);
  }
  void assembleMassMatrix(maths::SparseMatrixBuilder<Real>& globalMass) const {
    std::visit([&](auto&& arg) {
      arg.assembleMassMatrix(globalMass);
    }, data);
  }
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_PRIMITIVE_H_
