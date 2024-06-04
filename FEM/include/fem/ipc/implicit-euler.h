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
inline Real unconstrainedLinfNorm(const VecXd &p, const std::vector<bool>& constrained) {
  Real maxv = 0.0;
  for (int i = 0; i < p.size(); i++)
    if (!constrained[i])
      maxv = std::max(maxv, std::abs(p[i]));
  return maxv;
}
class IpcImplicitEuler : public IpcIntegrator {
 public:
  void step(System &system, Real dt) override;
 private:
  SparseMatrix<Real> spdProjectHessian(const System &system) {
    hessian_builder.reset();

  }
  Real incrementalPotentialEnergy(const System &system, const VecXd& x, const VecXd& x_t, Real h) {
    const auto &v_t = system.xdot;
    const auto &f_e = system.f_ext;
    auto x_hat = x_t + h * v_t + h * h * M.inverse() * f_e;
    return 0.5 * (x - x_hat).transpose() * M * (x - x_hat) + h * h * system.deformationEnergy();
  }
  // for now, do not consider dissipative friction forces
  Real barrierAugmentedIncrementalPotentialEnergy(const System& system,
                                                  const VecXd &x,
                                                  const VecXd &x_t,
                                                  const tbb::concurrent_vector<ipc::Constraint> &active_constraints,
                                                  Real h) {
    Real barrier_energy = 0.0;
    for (const auto &c : active_constraints) {

    }
    return incrementalPotentialEnergy(system, x, x_t, h) + kappa * barrier_energy;
  }
  auto incrementalPotentialEnergyGradient(const System &system, const VecXd&x, const VecXd& x_t, Real h) {
    const auto &v_t = system.xdot;
    const auto &f_e = system.f_ext;
    auto x_hat = x_t + h * v_t + h * h * M.inverse() * f_e;
    return M * (x - x_hat) + h * h * system.deformationEnergyGradient(x);
  }
  VecXd barrierAugmentedIncrementalPotentialEnergyGradient(const System &system,
                                                           const VecXd &x,
                                                           const VecXd &x_t,
                                                           const tbb::concurrent_vector<ipc::Constraint> &active_constraints,
                                                           Real h) {
    VecXd barrier_energy_gradient = VecXd::Zero(system.x.size());
    for (const auto &c : active_constraints) {

    }
    return incrementalPotentialEnergyGradient(system, x, x_t, h) + kappa * barrier_energy_gradient;
  }
  Real computeStepSizeUpperBound(const VecXd &x,
                                 const VecXd &p,
                                 const tbb::concurrent_vector<ipc::Constraint> &active_constraints) {

  }
  void computeConstraintSet(const VecXd &x, Real d_hat) {

  }
  void adaptStiffness() {

  }
  maths::SparseMatrixBuilder<Real> hessian_builder;
  Eigen::SimplicialLDLT<SparseMatrix<Real>> ldlt;
  Eigen::DiagonalMatrix<Real, Eigen::Dynamic> M;
  std::vector<bool> constrained;
  ipc::BarrierFunction barrier;
  Real kappa;
  core::Logger logger;
  VecXd x_prev;
  VecXd x_t_buf;
  VecXd p_buf, g;
};

}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_IMPLICIT_EULER_H_
