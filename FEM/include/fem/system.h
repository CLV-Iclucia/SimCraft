//
// Created by creeper on 5/25/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_SYSTEM_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_SYSTEM_H_
#include <fem/types.h>
#include <fem/tet-mesh.h>
#include <Maths/geometry.h>
#include <Core/debug.h>
#include <Core/zip.h>
#include <Deform/strain-energy-density.h>
#include <Maths/sparse-matrix-builder.h>
namespace fem {
using maths::HalfPlane;
using maths::vectorize;
using deform::DeformationGradient;
using deform::StrainEnergyDensity;
struct System {
  auto currentPos(int i) const {
    return x.segment<3>(i * 3);
  }
  auto referencePos(int i) const {
    return X.segment<3>(i * 3);
  }
 private:
  void updateDeformationGradient() const;
  HalfPlane wall;
  VecXd x;
  mutable bool E_dirty{false}, dg_dirty{false}, E_grad_dirty{false};
  mutable VecXd psi_grad;
  int num_triangles, num_tets;
  mutable Real E_cache;
  struct Primitives {
    Matrix<int, 3, Dynamic> surfaces;
    Matrix<int, 4, Dynamic> tets;
    std::vector<Real> ref_volumes;
    mutable std::vector<DeformationGradient<Real, 3>> dgs;
    std::vector<int> energy_indices;
    std::vector<std::unique_ptr<StrainEnergyDensity<Real>>> energies;
  } primitives;
  enum class State : uint8_t {
    Initialization,
    Simulation,
  };
  State state{State::Initialization};
 public:
  struct Config {
  } config;
  const Matrix<int, 3, Dynamic> &surfaces() const {
    return primitives.surfaces;
  }
  VecXd xdot, X, f_ext;
  explicit System(const Config &config) : config(config) {}
  void spdProjectHessian(maths::SparseMatrixBuilder<Real> &builder) const;
  [[nodiscard]] Real deformationEnergy() const;
  System &updateCurrentConfig(const VecXd &x_nxt);
  [[nodiscard]] const VecXd &deformationEnergyGradient() const;
  [[nodiscard]] int numTets() const {
    return num_tets;
  }
  [[nodiscard]] int numTriangles() const {
    return num_triangles;
  }
  System &startSimulationPhase() {
    state = State::Simulation;
    return *this;
  }
  const VecXd &currentConfig() const {
    return x;
  }
  System &addPrimitive(TetMesh &&mesh, std::unique_ptr<StrainEnergyDensity<Real>>&& energy) {
    if (state != State::Initialization)
      core::ERROR("Cannot add tet mesh after initialization");
    if (X.size() == 0) {
      X = mesh.vertices;
      return *this;
    }
    assert(X.cols() % 3 == 0);
    int current_num_vertices = static_cast<int>(X.cols() / 3);
    int current_num_tets = static_cast<int>(primitives.tets.cols());
    int current_num_triangles = static_cast<int>(primitives.surfaces.cols());
    X.conservativeResize(Eigen::NoChange, X.cols() + mesh.vertices.cols());
    // copy the positions of the vertices to the end of the X vector
    X.block(0, current_num_vertices * 3, 3, mesh.vertices.cols()) = mesh.vertices;
    primitives.tets.conservativeResize(Eigen::NoChange, current_num_tets + mesh.tets.cols());
    primitives.surfaces.conservativeResize(Eigen::NoChange, current_num_triangles + mesh.surfaces.cols());
    for (int i = 0; i < mesh.tets.cols(); i++)
      primitives.tets.col(current_num_tets + i) = mesh.tets.col(i) + Vec4i::Constant(current_num_vertices);
    for (int i = 0; i < mesh.surfaces.cols(); i++)
      primitives.surfaces.col(current_num_triangles + i) = mesh.surfaces.col(i) + Vec3i::Constant(current_num_vertices);
    primitives.dgs.reserve(primitives.tets.cols());
    primitives.ref_volumes.reserve(primitives.tets.cols());
    for (int i = current_num_tets; i < primitives.tets.cols(); i++) {
      Matrix<Real, 3, 3> local_X;
      for (int j = 0; j < 3; j++)
        local_X.col(j) = referencePos(primitives.tets(j, i)) - referencePos(primitives.tets(0, i));
      primitives.dgs.emplace_back(local_X);
      primitives.ref_volumes.emplace_back(std::abs(local_X.determinant()) / 6.0);
    }
    primitives.energies.push_back(std::move(energy));
    for (int i = 0; i < mesh.tets.cols(); i++)
      primitives.energy_indices.push_back(static_cast<int>(primitives.energies.size()) - 1);
    num_tets = static_cast<int>(primitives.tets.cols());
    num_triangles = static_cast<int>(primitives.surfaces.cols());
    return *this;
  }
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_SYSTEM_H_