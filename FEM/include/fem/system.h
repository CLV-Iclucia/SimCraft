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

struct SystemUpdateListener {
  virtual void onSystemUpdate(const System& system) = 0;
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
  mutable bool energyDirty{false}, dgDirty{false}, energyGradientDirty{false};
  mutable VecXd energyGradient;
  mutable Real cachedEnergy{};
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
  Eigen::SparseMatrix<Real> m_mass;
  Eigen::SimplicialLDLT<SparseMatrix<Real>> m_massLDLT;
  Real m_meshLengthScale{std::numeric_limits<Real>::infinity()};

  void updateDeformationGradient() const;

  void buildMassMatrix(maths::SparseMatrixBuilder<Real> &builder) const;
 public:
  VecXd xdot{}, X{}, f_ext{};

  Real meshLengthScale() const {
    return m_meshLengthScale;
  }

  const Matrix<int, 3, Dynamic> &surfaces() const {
    return primitives.surfaces;
  }

  const Matrix<int, 2, Dynamic> &edges() const {
    return primitives.edges;
  }

  const SparseMatrix<Real> &mass() const {
    return m_mass;
  }

  const Eigen::SimplicialLDLT<SparseMatrix<Real>> &massLDLT() const {
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
    return x.size();
  }

  [[nodiscard]] int numVertices() const {
    assert(x.size() % 3 == 0);
    return static_cast<int>(x.size() / 3);
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
    std::cout << "mass built" << std::endl;
    std::cout << m_mass.rows() << " " << m_mass.cols() << std::endl;
    m_massLDLT.compute(m_mass);
    std::cout << "mass inverse computed" << std::endl;
    if (m_massLDLT.info() != Eigen::Success)
      core::ERROR("Failed to compute mass matrix LDLT decomposition");
    f_ext.resize(static_cast<Eigen::Index>(dof()));
    f_ext.setZero();
    energyGradient.resize(static_cast<Eigen::Index>(dof()));
    dgDirty = true;
    energyDirty = true;
    energyGradientDirty = true;
    return *this;
  }

  Real kineticEnergy() const {
    return 0.5 * xdot.dot(m_mass * xdot);
  }

  Real potentialEnergy() const {
    return deformationEnergy();
  }

  Real totalEnergy() const {
    return kineticEnergy() + potentialEnergy();
  }

  const VecXd &currentConfig() const {
    return x;
  }

  System &addPrimitive(PrimitiveConfig &&config);

  void saveSurfaceObjFile(const std::filesystem::path &path) const;

  friend VecXd symbolicDeformationEnergyGradient(System &system);
  friend VecXd numericalDeformationEnergyGradient(System &system);
};

VecXd symbolicDeformationEnergyGradient(System &system);
VecXd numericalDeformationEnergyGradient(System &system);
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_SYSTEM_H_