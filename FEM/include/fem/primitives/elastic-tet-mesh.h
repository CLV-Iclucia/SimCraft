//
// Created by creeper on 10/22/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_PRIMITIVES_TET_MESH_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_PRIMITIVES_TET_MESH_H_
#include <memory>
#include <fem/types.h>
#include <fem/primitives/primitive-base.h>
#include <Deform/strain-energy-density.h>
#include <Maths/sparse-matrix-builder.h>
namespace fem {
struct ElasticTetMesh : PrimitiveBase {
  std::vector<std::array<Index, 4>> tets{};
  std::unique_ptr<deform::StrainEnergyDensity<Real>> energy{};
  void assembleEnergyGradient(VecXd &globalGrad) const {
  }
  [[nodiscard]] size_t dofDim() const {
    return dofView.size();
  }
  [[nodiscard]] Real potentialEnergy() const {
    return 0.0;
  }
  void assembleEnergyHessian(maths::SparseMatrixBuilder<Real> &globalHessian) const {
  }
  void assembleMassMatrix(maths::SparseMatrixBuilder<Real> &globalMass) const {
  }
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_PRIMITIVES_TET_MESH_H_
