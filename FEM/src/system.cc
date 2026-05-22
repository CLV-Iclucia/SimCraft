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

void System::updateCurrentConfig(const maths::BlockVector<3> &x_nxt) {
  x = x_nxt;
  updateDeformationGradient();
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
  return VecXd(system.deformationEnergyGradient().asEigen());
}

VecXd numericalDeformationEnergyGradient(System &system) {
  VecXd grad(system.dof());
  Real dx = 1e-7;
  maths::BlockVector<3> saved = system.x;
  for (int i = 0; i < system.dof(); i++) {
    system.x.data()[i] += dx;
    system.updateCurrentConfig(system.x);
    Real E_plus = system.deformationEnergy();
    system.x.data()[i] -= 2.0 * dx;
    system.updateCurrentConfig(system.x);
    Real E_minus = system.deformationEnergy();
    system.x.data()[i] += dx; // restore
    grad(i) = (E_plus - E_minus) / (2 * dx);
  }
  system.updateCurrentConfig(saved);
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
  assert(dofDim % 3 == 0 && "DOF dimension must be a multiple of 3 for BlockVector<3>");
  int numBlocks = static_cast<int>(dofDim / 3);
  x.resize(numBlocks);
  xdot.resize(numBlocks);
  X.resize(numBlocks);
  energyGradient.resize(numBlocks);
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
  m_blockMass = maths::BlockSparseMatrix<3>::fromEigen(m_mass);
  initGeometryManager();

  if (nEdges > 0) {
    Real totalLength = 0.0;
    auto xView = x.asEigen();
    for (int i = 0; i < nEdges; ++i) {
      auto vertices = getGlobalEdge(i);
      Vector<Real, 3> v0 = xView.segment<3>(vertices.x * 3);
      Vector<Real, 3> v1 = xView.segment<3>(vertices.y * 3);
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
