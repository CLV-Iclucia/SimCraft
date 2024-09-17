//
// Created by creeper on 6/12/24.
//
#include <fem/system.h>
#include <Deform/invariants.h>
namespace fem {

static void tetAssembleGlobal(VecXd &global, const Vector<Real, 12> &local, const TetrahedronTopology &tet) {
  for (int i = 0; i < 4; i++)
    global.segment<3>(tet(i) * 3) += local.segment<3>(3 * i);
}

void System::spdProjectHessian(maths::SparseMatrixBuilder<Real> &builder) const {
  for (int i = 0; i < numTets(); i++) {
    auto &dg = primitives.tetDeformationGradients[i];
    auto &energy = primitives.meshEnergies[primitives.tetMeshIDs[i]];
    auto hessian_F = energy->filteredEnergyHessian(dg);
    auto p_F_p_x = dg.gradient();
    auto hessian_x = p_F_p_x.transpose() * hessian_F * p_F_p_x * primitives.tetRefVolumes[i];
    builder.assembleBlock<12, 3>(hessian_x,
                                 primitives.tets(0, i),
                                 primitives.tets(1, i),
                                 primitives.tets(2, i),
                                 primitives.tets(3, i));
  }
}

Real System::deformationEnergy() const {
  if (state != State::Simulation)
    core::ERROR("Cannot compute deformation energy in states other than simulation");
  if (dgDirty)
    updateDeformationGradient();
  else if (!energyDirty)
    return cachedEnergy;
  assert(!dgDirty);
  Real sum = 0;
  for (int i = 0; i < numTets(); i++) {
    auto &dg = primitives.tetDeformationGradients[i];
    auto &energy = primitives.meshEnergies[primitives.tetMeshIDs[i]];
    sum += energy->computeEnergyDensity(dg) * primitives.tetRefVolumes[i];
  }
  cachedEnergy = sum;
  energyDirty = false;
  return sum;
}

System &System::updateCurrentConfig(const VecXd &x_nxt) {
  x = x_nxt;
  dgDirty = true;
  energyDirty = true;
  energyGradientDirty = true;
  return *this;
}

const VecXd &System::deformationEnergyGradient() const {
  if (state != State::Simulation)
    core::ERROR("Cannot compute deformation energy gradient in states other than simulation");
  if (dgDirty)
    updateDeformationGradient();
  else if (!energyGradientDirty)
    return energyGradient;
  energyGradient.setZero();
  for (int i = 0; i < numTets(); i++) {
    auto &dg = primitives.tetDeformationGradients[i];
    auto &energy = primitives.meshEnergies[primitives.tetMeshIDs[i]];
    auto grad = energy->computeEnergyGradient(dg);
    auto p_F_p_x = dg.gradient();
    Vector<Real, 12> grad_x = p_F_p_x.transpose() * vectorize(grad) * primitives.tetRefVolumes[i];
    tetAssembleGlobal(energyGradient, grad_x, primitives.tets.col(i));
  }
  energyGradientDirty = false;
  return energyGradient;
}

void System::updateDeformationGradient() const {
  if (!dgDirty) return;
  for (int i = 0; i < numTets(); i++) {
    auto &dg = primitives.tetDeformationGradients[i];
    Matrix<Real, 3, 3> local_x;
    for (int j = 0; j < 3; j++)
      local_x.col(j) = currentPos(primitives.tets(j + 1, i)) - currentPos(primitives.tets(0, i));
    dg.updateCurrentConfig(local_x);
  }
  dgDirty = false;
}

void System::saveSurfaceObjFile(const std::filesystem::path &path) const {
  std::ofstream out(path);
  for (int i = 0; i < numVertices(); i++)
    out << "v " << currentPos(i).transpose() << std::endl;
  for (int i = 0; i < primitives.surfaces.cols(); i++)
    out << "f " << primitives.surfaces(0, i) + 1 << " " << primitives.surfaces(1, i) + 1 << " "
        << primitives.surfaces(2, i) + 1 << std::endl;
  out.close();
}

System &System::addPrimitive(PrimitiveConfig &&config) {
  auto &&mesh = std::move(config.mesh);
  auto &&energy = std::move(config.energy);
  Real density = config.density;
  if (state != State::Initialization) {
    core::ERROR("Cannot add tet mesh after initialization");
    return *this;
  }
  int current_num_vertices = static_cast<int>(X.rows() / 3);
  int current_num_tets = static_cast<int>(primitives.tets.cols());
  int current_num_triangles = static_cast<int>(primitives.surfaces.cols());
  int current_num_edges = static_cast<int>(primitives.edges.cols());
  int added_num_vertices = static_cast<int>(mesh->vertices.cols());
  int added_num_tets = static_cast<int>(mesh->tets.cols());
  int added_num_triangles = static_cast<int>(mesh->surfaces.cols());
  int added_num_edges = static_cast<int>(mesh->surfaceEdges.cols());
  if (X.size() == 0) {
    X = std::move(mesh->vertices).reshaped();
    xdot = std::move(config.velocities).reshaped();
    x = X;
  } else {
    assert(X.rows() % 3 == 0);
    X.conservativeResize(X.rows() + added_num_vertices * 3, Eigen::NoChange);
    X.block(current_num_vertices * 3, 0, added_num_vertices * 3, 1) = std::move(mesh->vertices).reshaped();

    xdot.conservativeResize(xdot.rows() + added_num_vertices * 3, Eigen::NoChange);
    xdot.block(current_num_vertices * 3, 0, added_num_vertices * 3, 1) = std::move(config.velocities).reshaped();

    x.conservativeResize(x.rows() + added_num_vertices * 3, Eigen::NoChange);
    x.block(current_num_vertices * 3, 0, added_num_vertices * 3, 1) =
        X.block(current_num_vertices * 3, 0, added_num_vertices * 3, 1);
  }

  primitives.edges.conservativeResize(Eigen::NoChange, primitives.edges.cols() + added_num_edges);
  for (int i = 0; i < added_num_edges; i++) {
    Real length = (currentPos(mesh->surfaceEdges(0, i)) - currentPos(mesh->surfaceEdges(1, i))).norm();
    m_meshLengthScale = std::min(m_meshLengthScale, length);
    primitives.edges.col(current_num_edges + i) =
        mesh->surfaceEdges.col(i) + Vector<int, 2>::Constant(current_num_vertices);
  }

  primitives.tets.conservativeResize(Eigen::NoChange, current_num_tets + mesh->tets.cols());
  for (int i = 0; i < added_num_tets; i++)
    primitives.tets.col(current_num_tets + i) = mesh->tets.col(i) + Vector<int, 4>::Constant(current_num_vertices);

  primitives.surfaces.conservativeResize(Eigen::NoChange, current_num_triangles + mesh->surfaces.cols());
  for (int i = 0; i < added_num_triangles; i++)
    primitives.surfaces.col(current_num_triangles + i) =
        mesh->surfaces.col(i) + Vector<int, 3>::Constant(current_num_vertices);

  primitives.tetDeformationGradients.reserve(primitives.tets.cols());
  primitives.tetRefVolumes.reserve(primitives.tets.cols());

  for (int i = current_num_tets; i < primitives.tets.cols(); i++) {
    Matrix<Real, 3, 3> local_X;
    for (int j = 0; j < 3; j++)
      local_X.col(j) = referencePos(primitives.tets(j + 1, i)) - referencePos(primitives.tets(0, i));
    primitives.tetDeformationGradients.emplace_back(local_X);
    primitives.tetRefVolumes.emplace_back(std::abs(local_X.determinant()) / 6.0);
  }

  primitives.meshEnergies.emplace_back(std::move(energy));
  primitives.meshDensities.push_back(density);
  for (int i = 0; i < mesh->tets.cols(); i++)
    primitives.tetMeshIDs.push_back(static_cast<int>(primitives.meshEnergies.size()) - 1);
  return *this;
}

void System::buildMassMatrix(maths::SparseMatrixBuilder<Real> &builder) const {
  for (auto i = 0; i < numTets(); i++) {
    Real density = primitives.meshDensities[primitives.tetMeshIDs[i]];
    Real volume = primitives.tetRefVolumes[i];
    Matrix<Real, 12, 12> localMass;
    localMass.setZero();
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++)
        localMass.block<3, 3>(j * 3, k * 3) +=
            density * volume * (j == k ? 0.1 : 0.05) * Matrix<Real, 3, 3>::Identity();
    builder.assembleBlock<12, 3>(localMass,
                                 primitives.tets(0, i),
                                 primitives.tets(1, i),
                                 primitives.tets(2, i),
                                 primitives.tets(3, i));
  }
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
    Real E_plus = system.updateCurrentConfig(x_plus).deformationEnergy() / (2 * dx);
    Real E_minus = system.updateCurrentConfig(x_minus).deformationEnergy() / (2 * dx);
    grad(i) = (E_plus - E_minus);
  }
  system.updateCurrentConfig(current);
  return grad;
}
}