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

struct PrimitiveConfig {
  std::unique_ptr<TetMesh> mesh{};
  Matrix<Real, 3, Dynamic> velocities{};
  std::unique_ptr<StrainEnergyDensity<Real>> energy{};
  Real density{};
};

struct System {
  auto currentPos(int i) const {
    return x.segment<3>(i * 3);
  }

  auto referencePos(int i) const {
    return X.segment<3>(i * 3);
  }

 private:
  VecXd x;
  mutable bool E_dirty{false}, dg_dirty{false}, E_grad_dirty{false};
  mutable VecXd psi_grad;
  mutable Real E_cache{};
  struct Primitives {
    Matrix<int, 3, Dynamic> surfaces;
    Matrix<int, 4, Dynamic> tets;
    Matrix<int, 2, Dynamic> edges;
    std::vector<Real> tetRefVolumes;
    mutable std::vector<DeformationGradient<Real, 3>> tetDeformationGradients;
    std::vector<int> tetMeshIDs;
    std::vector<std::unique_ptr<StrainEnergyDensity<Real>>> meshEnergies;
    std::vector<Real> meshDensities;
  } primitives;
  enum class State : uint8_t {
    Initialization,
    Simulation,
  };
  State state{State::Initialization};
  std::vector<int> m_vertices{};
  Eigen::SparseMatrix<Real> m_mass;
  Eigen::SimplicialLDLT<SparseMatrix<Real>> m_massLDLT;


  void updateDeformationGradient() const;

  void buildMassMatrix(maths::SparseMatrixBuilder<Real> &builder) const;
 public:
  VecXd xdot{}, X{}, f_ext{};

  const std::vector<int> &vertices() const {
    return m_vertices;
  }

  const Matrix<int, 3, Dynamic> &surfaces() const {
    return primitives.surfaces;
  }

  const Matrix<int, 2, Dynamic> &edges() const {
    return primitives.edges;
  }

  const SparseMatrix<Real>& mass() const {
    return m_mass;
  }

  const Eigen::SimplicialLDLT<SparseMatrix<Real>>& massLDLT() const {
    return m_massLDLT;
  }

  void spdProjectHessian(maths::SparseMatrixBuilder<Real> &builder) const;

  [[nodiscard]] Real deformationEnergy() const;

  System &updateCurrentConfig(const VecXd &x_nxt);

  [[nodiscard]] const VecXd &deformationEnergyGradient() const;

  [[nodiscard]] int numTets() const {
    return static_cast<int>(primitives.tets.cols());
  }

  [[nodiscard]] int numTriangles() const {
    return static_cast<int>(primitives.surfaces.cols());
  }

  int triangleVertexIndex(int triangle_index, int vertex_id) const {
    assert(vertex_id >= 0 && vertex_id < 3);
    return surfaces()(vertex_id, triangle_index);
  }

  int edgeVertexIndex(int edge_index, int vertex_id) const {
    assert(vertex_id >= 0 && vertex_id < 2);
    return edges()(vertex_id, edge_index);
  }

  size_t dof() const {
    return 3 * m_vertices.size();
  }

  bool checkEdgeAdjacent(int ia, int ib) const {
    if (edgeVertexIndex(ia, 0) == edgeVertexIndex(ib, 0) || edgeVertexIndex(ia, 0) == edgeVertexIndex(ib, 1))
      return true;
    if (edgeVertexIndex(ia, 1) == edgeVertexIndex(ib, 0) || edgeVertexIndex(ia, 1) == edgeVertexIndex(ib, 1))
      return true;
    return false;
  }

  bool checkTriangleAdjacent(int ia, int ib) const {
    int count = 0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        if (triangleVertexIndex(ia, i) == triangleVertexIndex(ib, j))
          count++;
    return count > 0;
  }

  System &startSimulationPhase() {
    state = State::Simulation;
    auto massBuilder = maths::SparseMatrixBuilder<Real>(dof(), dof());
    buildMassMatrix(massBuilder);
    m_mass = massBuilder.build();
    m_massLDLT.compute(m_mass);
    if (m_massLDLT.info() != Eigen::Success)
      core::ERROR("Failed to compute mass matrix LDLT decomposition");
    return *this;
  }

  const VecXd &currentConfig() const {
    return x;
  }

  System &addPrimitive(PrimitiveConfig&& config);
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_SYSTEM_H_