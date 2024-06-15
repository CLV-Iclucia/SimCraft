//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
#include <fem/integrator.h>
#include <fem/ipc/constraint.h>
#include <tbb/concurrent_vector.h>
#include <fem/ipc/barrier-functions.h>
#include <Core/log.h>
#include <Core/debug.h>
#include <Core/timer.h>
#include <Maths/sparse-matrix-builder.h>
namespace fem {
class IpcIntegrator : public Integrator {
 public:
  struct Config {
    Real dHat;
    Real eps;
  } config;
 protected:
  tbb::concurrent_vector<ipc::Constraint> active_constraints;
};
inline Real unconstrainedLinfNorm(const VecXd &p, const std::vector<bool> &constrained) {
  Real maxv = 0.0;
  for (int i = 0; i < p.size(); i++)
    if (!constrained[i])
      maxv = std::max(maxv, std::abs(p[i]));
  return maxv;
}
class IpcImplicitEuler : public IpcIntegrator {
 public:
  void step(Real dt) override;
 private:
  SparseMatrix<Real> spdProjectHessian();
  Real incrementalPotentialEnergy(const VecXd &x_t, Real h) const;
  // for now, do not consider dissipative friction forces
  Real barrierAugmentedIncrementalPotentialEnergy(const VecXd &x_t,
                                                  const tbb::concurrent_vector<ipc::Constraint> &active_constraints,
                                                  Real h);
  auto incrementalPotentialEnergyGradient(const VecXd &x_t, Real h) {
    const auto &v_t = system().xdot;
    const auto &f_e = system().f_ext;
    auto x_hat = x_t + h * v_t + h * h * M.inverse() * f_e;
    return M * (system().currentConfig() - x_hat) + h * h * system().deformationEnergyGradient();
  }
  VecXd barrierAugmentedIncrementalPotentialEnergyGradient(const VecXd &x_t,
                                                           const tbb::concurrent_vector<ipc::Constraint> &active_constraints,
                                                           Real h) {
    VecXd barrier_energy_gradient = VecXd::Zero(system().currentConfig().size());
    for (const auto &c : active_constraints) {

    }
    return incrementalPotentialEnergyGradient(x_t, h) + kappa * barrier_energy_gradient;
  }
  void updateCandidateSolution(const VecXd &x) {
    system().updateCurrentConfig(x);
    constraints_dirty = true;
  }
  Real computeStepSizeUpperBound(const VecXd &p,
                                 const tbb::concurrent_vector<ipc::Constraint> &active_constraints) {

  }
  void computeConstraintSet(const VecXd &x, Real d_hat) {
    if (!constraints_dirty)
      return;

    constraints_dirty = false;
  }
  void adaptStiffness() {

  }
  maths::SparseMatrixBuilder<Real> hessian_builder;
  Eigen::SimplicialLDLT<SparseMatrix<Real>> ldlt;
  Eigen::DiagonalMatrix<Real, Eigen::Dynamic> M;
  ipc::BarrierFunction barrier;
  Real kappa;
  core::Logger logger;
  VecXd x_prev, x_t_buf;
  VecXd g;
  std::vector<bool> constrained_cache;
  bool constraints_dirty{false};
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
