//
// Created by creeper on 6/12/24.
//
#include <fem/system.h>

namespace fem {

static void tetAssembleGlobal(VecXd &global, const VecXd &local, const TetrahedronTopology &tet) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      global(tet(i) * 3 + j) += local(i * 3 + j);
}

void System::spdProjectHessian(maths::SparseMatrixBuilder<Real> &builder) const {
  for (int i = 0; i < numTets(); i++) {
    auto &dg = primitives.tetDeformationGradients[i];
    auto &energy = primitives.meshEnergies[primitives.tetMeshIDs[i]];
    auto hessian_F = energy->filteredEnergyHessian(dg);
    auto p_F_p_x = dg.gradient();
    auto hessian_x = p_F_p_x.transpose() * hessian_F * p_F_p_x;
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
  if (dg_dirty)
    updateDeformationGradient();
  else if (!E_dirty)
    return E_cache;
  assert(!dg_dirty);
  Real sum = 0;
  for (int i = 0; i < numTets(); i++) {
    auto &dg = primitives.tetDeformationGradients[i];
    auto &energy = primitives.meshEnergies[primitives.tetMeshIDs[i]];
    sum += energy->computeEnergyDensity(dg);
  }
  E_cache = sum;
  E_dirty = false;
  return sum;
}

System &System::updateCurrentConfig(const VecXd &x_nxt) {
  x = x_nxt;
  dg_dirty = true;
  E_dirty = true;
  E_grad_dirty = true;
  return *this;
}

const VecXd &System::deformationEnergyGradient() const {
  if (state != State::Simulation)
    core::ERROR("Cannot compute deformation energy gradient in states other than simulation");
  if (dg_dirty)
    updateDeformationGradient();
  else if (!E_grad_dirty)
    return psi_grad;
  psi_grad.setZero();
  for (int i = 0; i < numTets(); i++) {
    auto &dg = primitives.tetDeformationGradients[i];
    auto &energy = primitives.meshEnergies[primitives.tetMeshIDs[i]];
    auto grad = energy->computeEnergyGradient(dg);
    auto p_F_p_x = dg.gradient();
    auto grad_x = p_F_p_x.transpose() * vectorize(grad);
    tetAssembleGlobal(psi_grad, grad_x, primitives.tets.col(i));
  }
  return psi_grad;
}

void System::updateDeformationGradient() const {
  if (!dg_dirty) return;
  for (int i = 0; i < numTets(); i++) {
    auto &dg = primitives.tetDeformationGradients[i];
    Matrix<Real, 3, 3> local_x;
    for (int j = 0; j < 3; j++)
      local_x.col(j) = currentPos(primitives.tets(j + 1, i)) - currentPos(primitives.tets(0, i));
    dg.updateCurrentConfig(local_x);
  }
  dg_dirty = false;
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
  for (int i = 0; i < added_num_edges; i++)
    primitives.edges.col(current_num_edges + i) = mesh->surfaceEdges.col(i) + Vector<int, 2>::Constant(current_num_edges);

  primitives.tets.conservativeResize(Eigen::NoChange, current_num_tets + mesh->tets.cols());
  for (int i = 0; i < added_num_tets; i++)
    primitives.tets.col(current_num_tets + i) = mesh->tets.col(i) + Vector<int, 4>::Constant(current_num_vertices);

  primitives.surfaces.conservativeResize(Eigen::NoChange, current_num_triangles + mesh->surfaces.cols());
  for (int i = 0; i < added_num_triangles; i++)
    primitives.surfaces.col(current_num_triangles + i) = mesh->surfaces.col(i) + Vector<int, 3>::Constant(current_num_vertices);

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
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++)
        localMass.block<3, 3>(j * 3, k * 3) = density * volume * (j == k ? 0.1 : 0.05) * Matrix<Real, 3, 3>::Identity();
    builder.assembleBlock<12, 3>(localMass,
                                  primitives.tets(0, i),
                                  primitives.tets(1, i),
                                  primitives.tets(2, i),
                                  primitives.tets(3, i));
  }
}

}