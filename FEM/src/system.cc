//
// Created by creeper on 6/12/24.
//
#include <Core/json.h>
#include <Deform/invariants.h>
#include <fem/system.h>
#include <spdlog/spdlog.h>

namespace sim::fem {
void tetAssembleGlobal(VecXd &global, const Vector<Real, 12> &local,
                       const Vector<int, 4> &tet) {
  for (int i = 0; i < 4; i++)
    global.segment<3>(tet[i] * 3) += local.segment<3>(3 * i);
}

void System::spdProjectHessian(
    maths::SparseMatrixBuilder<Real> &builder) const {
  autoDispatch([this, &builder](const Primitive &pr, int id) {
    auto subMatBuilder = builder.subMatrixBuilder(dofStarts[id], dofStarts[id],
                                                  pr.dofDim(), pr.dofDim());
    pr.assembleEnergyHessian(subMatBuilder);
  });
}

Real System::deformationEnergy() const { return cachedEnergy; }

System &System::updateCurrentConfig(const VecXd &x_nxt) {
  x = x_nxt;
  updateDeformationGradient();
  return *this;
}

const VecXd &System::deformationEnergyGradient() const {
  return energyGradient;
}

void System::updateDeformationGradient() {
  autoDispatch([this](Primitive &pr, int id) {
    pr.updateDeformationEnergyGradient(pr.view(x));
  });
  updateDeformationEnergy();
  updateDeformationEnergyGradient();
}

void System::buildMassMatrix(maths::SparseMatrixBuilder<Real> &builder) const {
  autoDispatch([this, &builder](const Primitive &pr, int id) {
    auto subMatBuilder = builder.subMatrixBuilder(dofStarts[id], dofStarts[id],
                                                  pr.dofDim(), pr.dofDim());
    pr.assembleMassMatrix(subMatBuilder);
  });
}

VecXd symbolicDeformationEnergyGradient(System &system) {
  return system.deformationEnergyGradient();
}

VecXd numericalDeformationEnergyGradient(System &system) {
  VecXd grad(system.dof());
  Real dx = 1e-7;
  VecXd current = system.currentConfig();
  for (int i = 0; i < system.dof(); i++) {
    VecXd x_plus = current;
    x_plus(i) += dx;
    VecXd x_minus = current;
    x_minus(i) -= dx;
    Real E_plus =
        system.updateCurrentConfig(x_plus).deformationEnergy() / (2 * dx);
    Real E_minus =
        system.updateCurrentConfig(x_minus).deformationEnergy() / (2 * dx);
    grad(i) = (E_plus - E_minus);
  }
  system.updateCurrentConfig(current);
  return grad;
}

void System::updateDeformationEnergy() {
  cachedEnergy = 0.0;
  autoDispatch([this](Primitive &pr, int id) {
    cachedEnergy += pr.deformationEnergy();
  });
}
void System::updateDeformationEnergyGradient() {
  energyGradient.setZero();
  autoDispatch([this](Primitive &pr, int dofStart) {
    pr.assembleEnergyGradient(pr.view(energyGradient));
  });
}

System &System::init() {
  size_t dofDim{};
  autoDispatch(
      [this, &dofDim](const Primitive &pr, int id) { dofDim += pr.dofDim(); });
  x.resize(static_cast<Eigen::Index>(dofDim));
  xdot.resize(static_cast<Eigen::Index>(dofDim));
  X.resize(static_cast<Eigen::Index>(dofDim));
  energyGradient.resize(static_cast<Eigen::Index>(dofDim));
  dofStarts.resize(prs.size());
  int dofStart = 0;
  for (int i = 0; i < prs.size(); ++i) {
    dofStarts[i] = dofStart;
    prs[i].setDofStart(dofStart);
    dofStart += static_cast<int>(prs[i].dofDim());
  }
  autoDispatch([this](Primitive &pr, int id) {
    auto x = pr.view(this->x);
    auto X = pr.view(this->X);
    auto xdot = pr.view(this->xdot);
    pr.init(x, xdot, X);
  });
  auto massBuilder = maths::SparseMatrixBuilder<Real>(dof(), dof());
  buildMassMatrix(massBuilder);
  m_mass = massBuilder.build();
  initGeometryManager();

  if (nEdges > 0) {
    Real totalLength = 0.0;
    for (int i = 0; i < nEdges; ++i) {
      auto vertices = getGlobalEdge(i);
      Vector<Real, 3> v0 = x.segment<3>(vertices.x * 3);
      Vector<Real, 3> v1 = x.segment<3>(vertices.y * 3);
      totalLength += (v1 - v0).norm();
    }
    m_meshLengthScale = totalLength / nEdges;
  }
  logSystemInfo();
  return *this;
}

void System::initGeometryManager() {
  m_geometryManager.collectGeometryReferences(prs, colliders);
  nTriangles = m_geometryManager.triangleCount();
  nEdges = m_geometryManager.edgeCount();
  nVertices = m_geometryManager.vertexCount();
}

void System::logSystemInfo() const {
  // log all the system info so that we can know all the information about the
  // system
  spdlog::info("System Info:");
  spdlog::info("Number of primitives: {}", prs.size());
  spdlog::info("Number of colliders: {}", colliders.size());
  spdlog::info("Number of triangles: {}", nTriangles);
  spdlog::info("Number of edges: {}", nEdges);
  spdlog::info("Number of vertices: {}", nVertices);
  spdlog::info("Mesh length scale: {}", m_meshLengthScale);
  for (int i = 0; i < prs.size(); ++i) {
    spdlog::info("Primitive {}: dofDim = {}, dofStart = {}", i, prs[i].dofDim(),
                 dofStarts[i]);
  }
}

int System::globalVertexToPrimitive(int globalIdx) const {
  return m_geometryManager.getVertexRef(globalIdx).primitiveId;
}

VecXd System::computeAcceleration() const { return {}; }
} // namespace sim::fem
