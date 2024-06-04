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
namespace fem {
using maths::HalfPlane;
using deform::DeformationGradient;
using deform::StrainEnergyDensity;
struct System {
  struct Config {
  } config;
  enum class State : uint8_t {
    Initialization,
    Simulation,
  };
  System(const Config &config) : config(config) {

  }
  [[nodiscard]] Real deformationEnergy() {
    if (state != State::Simulation)
      core::ERROR("Cannot compute deformation energy in states other than simulation");
    Real sum = 0.0;
    computeDeformationGradient();
    for (auto &[mesh, dgs, energy] : zip(primitives.meshes, primitives.dgs, primitives.energies))
      for (auto &[tet, dg] : zip(mesh.tets, dgs))
        sum += energy->computeEnergyDensity(dg);
    return sum;
  }
  [[nodiscard]] const VecXd &deformationEnergyGradient() const {
    for (auto &[mesh, dgs, energy] : zip(primitives.meshes, primitives.dgs, primitives.energies)) {
      for (auto &[tet, dg] : zip(mesh.tets, dgs)) {
        auto p_psi_p_F = energy->computeEnergyGradient(dg);
        auto p_F_p_x = dg.gradient();
        auto local_force = maths::doubleContract(p_psi_p_F, p_F_p_x);
        addGlobal(psi_grad, local_force, tet);
      }
    }
  }

  [[nodiscard]] int numTets() const {
    return num_tets;
  }
  [[nodiscard]] int numTriangles() const {
    return num_triangles;
  }
  void startSimulationPhase() {
    state = State::Simulation;
  }
  void addTetMesh(TetMesh &&mesh) {
    if (state != State::Initialization)
      core::ERROR("Cannot add tet mesh after initialization");
    if (X.size() == 0) {
      X = mesh.vertices;
      return;
    }
    assert(X.cols() % 3 == 0);
    int current_num_vertices = static_cast<int>(X.cols() / 3);
    X.conservativeResize(Eigen::NoChange, X.cols() + mesh.vertices.cols());
    // copy the positions of the vertices to the end of the X vector
    X.block(0, current_num_vertices * 3, 3, mesh.vertices.cols()) = mesh.vertices;
    primitives.meshes.push_back(std::move(mesh.tets));
    for (auto &tet : primitives.meshes.back().tets)
      for (int i = 0; i < 4; i++)
        tet[i] += current_num_vertices;
    for (auto &tri : primitives.meshes.back().surface)
      for (int i = 0; i < 3; i++)
        tri[i] += current_num_vertices;
    num_tets += static_cast<int>(primitives.meshes.back().tets.size());
    num_triangles += static_cast<int>(primitives.meshes.back().tets.size());
  }
 private:
  void addGlobal(VecXd &global, const VecXd &local, const TetrahedronTopology &tet) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 3; j++)
        global(tet(i) * 3 + j) += local(i * 3 + j);
  }
  auto currentPos(int i) const {
    return x.segment<3>(i * 3);
  }
  auto referencePos(int i) const {
    return X.segment<3>(i * 3);
  }
  void computeDeformationGradient() {
    for (auto &[mesh, dgs] : zip(primitives.meshes, primitives.dgs)) {
      for (auto &[tet, dg] : zip(mesh.tets, dgs)) {
        Matrix<Real, 3, 3> local_x;
        Matrix<Real, 3, 3> local_X;
        for (int i = 1; i < 4; i++) {
          local_x.col(i) = currentPos(tet[i]) - currentPos(tet[0]);
          local_X.col(i) = referencePos(tet[i]) - referencePos(tet[0]);
        }
        dg = DeformationGradient<Real, 3>(local_x, local_X);
      }
    }
  }
  HalfPlane wall;
  VecXd x, xdot, X, f_ext, psi_grad;
  Sparse
  int num_triangles, num_tets;
  struct Primitives {
    std::vector<TetMeshTopology> meshes;
    std::vector<std::vector<Real>> referenceVolumes;
    std::vector<std::vector<DeformationGradient < Real, 3>>> dgs;
    std::vector<std::unique_ptr<StrainEnergyDensity < Real>>> energies;
  } primitives;
  State state{State::Initialization};
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_SYSTEM_H_