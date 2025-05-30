//
// Created by creeper on 5/25/24.
//

#pragma once
#include "external-force.h"
#include "geometry-manager.h"
#include <Core/zip.h>
#include <Deform/strain-energy-density.h>
#include <Maths/sparse-matrix-builder.h>
#include <Spatify/lbvh.h>
#include <fem/colliders.h>
#include <fem/primitive.h>
#include <fem/types.h>

namespace sim {
namespace fem::ipc {
struct CollisionDetector;
}

namespace fem {
using deform::DeformationGradient;
using deform::StrainEnergyDensity;
using maths::vectorize;

// the only thing System do is dispatching tasks to primitives
// and gather the dof to solve the system globally
struct System {
  VecXd xdot{};
  [[nodiscard]] Real meshLengthScale() const { return m_meshLengthScale; }

  [[nodiscard]] const SparseMatrix<Real> &mass() const { return m_mass; }

  void spdProjectHessian(maths::SparseMatrixBuilder<Real> &builder) const;

  void updateDeformationEnergy();

  void updateDeformationEnergyGradient();

  [[nodiscard]] Real deformationEnergy() const;

  System &updateCurrentConfig(const VecXd &x_nxt);

  [[nodiscard]] const VecXd &deformationEnergyGradient() const;

  [[nodiscard]] size_t dof() const { return x.size(); }
  System &init();
  [[nodiscard]] Real kineticEnergy() const {
    return 0.5 * xdot.dot(m_mass * xdot);
  }

  [[nodiscard]] Real potentialEnergy() const { return deformationEnergy(); }

  [[nodiscard]] Real totalEnergy() const {
    return kineticEnergy() + potentialEnergy();
  }

  [[nodiscard]] const VecXd &currentConfig() const { return x; }
  [[nodiscard]] const VecXd &referenceConfig() const { return X; }
  [[nodiscard]] const Primitive &primitive(int id) const { return prs[id]; }
  Primitive &primitive(int id) { return prs[id]; }

  [[nodiscard]] const std::vector<Primitive> &primitives() const { return prs; }

  [[nodiscard]] int numTriangles() const { return nTriangles; }
  [[nodiscard]] int numEdges() const { return nEdges; }
  [[nodiscard]] int numVertices() const { return nVertices; }

  [[nodiscard]] int globalTriangleToPrimitive(int globalIdx) const {
    return m_geometryManager.getTriangleRef(globalIdx).primitiveId;
  }

  [[nodiscard]] int globalEdgeToPrimitive(int globalIdx) const {
    return m_geometryManager.getEdgeRef(globalIdx).primitiveId;
  }

  [[nodiscard]] int globalVertexToPrimitive(int globalIdx) const;

  [[nodiscard]] Triangle getTriangleVertices(int globalIdx) const {
    return m_geometryManager.getGlobalTriangle(globalIdx);
  }

  [[nodiscard]] Edge getGlobalEdge(int globalIdx) const {
    return m_geometryManager.getGlobalEdge(globalIdx);
  }

  [[nodiscard]] bool triangleContainsVertex(int triangleIdx,
                                            int vertexIdx) const {
    return m_geometryManager.triangleContainsVertex(triangleIdx, vertexIdx);
  }

  [[nodiscard]] bool checkEdgeAdjacent(int edgeA, int edgeB) const {
    return m_geometryManager.checkEdgeAdjacent(edgeA, edgeB);
  }

  [[nodiscard]] VecXd computeAcceleration() const;

  template <typename Func> void sequentialDispatch(Func &&func) const {
    for (auto i = 0; i < prs.size(); i++)
      func(prs[i], i);
  }

  template <typename Func> void sequentialDispatch(Func &&func) {
    for (auto i = 0; i < prs.size(); i++)
      func(prs[i], i);
  }

  template <typename Func> void autoDispatch(Func &&func) {
      sequentialDispatch(func);
  }

  template <typename Func> void autoDispatch(Func &&func) const {
      sequentialDispatch(func);
  }

  void initGeometryManager();

  [[nodiscard]] const GeometryManager &geometryManager() const {
    return m_geometryManager;
  }

  friend VecXd symbolicDeformationEnergyGradient(System &system);
  friend VecXd numericalDeformationEnergyGradient(System &system);
  friend struct SystemBuilder;

private:
  void logSystemInfo() const;
  VecXd x;
  VecXd X;
  VecXd energyGradient;
  Real cachedEnergy{};
  bool use_parallel_dispatch{false};

  std::vector<Primitive> prs;
  std::vector<Collider> colliders;
  std::vector<ExternalForce> externalForces;
  std::vector<int> dofStarts;
  GeometryManager m_geometryManager;

  int nTriangles{0};
  int nEdges{0};
  int nVertices{0};

  Eigen::SparseMatrix<Real> m_mass;
  Real m_meshLengthScale{std::numeric_limits<Real>::infinity()};

  void updateDeformationGradient();
  void buildMassMatrix(maths::SparseMatrixBuilder<Real> &builder) const;
};

VecXd symbolicDeformationEnergyGradient(System &system);
VecXd numericalDeformationEnergyGradient(System &system);

struct SystemConfig {
  std::vector<Primitive> primitives{};
  std::vector<Collider> colliders{};
  std::vector<ExternalForce> externalForces{};
  REFLECT(primitives, colliders)
};

struct SystemBuilder {
  System build(const core::JsonNode &json) {
    auto cfg = core::deserialize<SystemConfig>(json);
    System system;
    system.prs = std::move(cfg.primitives);
    system.colliders = std::move(cfg.colliders);
    system.init();
    return system;
  }
};
}
}
