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
  for (int i = 0; i < num_tets; i++) {
    auto &dg = primitives.dgs[i];
    auto &energy = primitives.energies[primitives.energy_indices[i]];
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
  for (int i = 0; i < num_tets; i++) {
    auto &dg = primitives.dgs[i];
    auto &energy = primitives.energies[primitives.energy_indices[i]];
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
  for (int i = 0; i < num_tets; i++) {
    auto &dg = primitives.dgs[i];
    auto &energy = primitives.energies[primitives.energy_indices[i]];
    auto grad = energy->computeEnergyGradient(dg);
    auto p_F_p_x = dg.gradient();
    auto grad_x = p_F_p_x.transpose() * vectorize(grad);
    tetAssembleGlobal(psi_grad, grad_x, primitives.tets.col(i));
  }
  return psi_grad;
}
void System::updateDeformationGradient() const {
  if (!dg_dirty) return;
  for (int i = 0; i < num_tets; i++) {
    auto &dg = primitives.dgs[i];
    Matrix<Real, 3, 3> local_x;
    for (int j = 0; j < 3; j++)
      local_x.col(j) = currentPos(primitives.tets(j, i)) - currentPos(primitives.tets(0, i));
    dg.updateCurrentConfig(local_x);
  }
  dg_dirty = false;
}
}