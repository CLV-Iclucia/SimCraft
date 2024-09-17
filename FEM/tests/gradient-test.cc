//
// Created by creeper on 9/13/24.
//
#include <Deform/strain-energy-density.h>
#include <fem/tet-mesh.h>
#include <fem/system.h>
#include <fem/ipc/implicit-euler.h>
#include <ogl-render/window.h>
#include <Maths/svd.h>
using namespace fem;
using namespace deform;

struct Material {
  ElasticityParameters<Real> params{1e7, 0.45};
  Real density{1150.0};
};

PrimitiveConfig alicePrimitive() {
  std::unique_ptr<StrainEnergyDensity<Real>> energy = std::make_unique<StableNeoHookean<Real>>(Material{}.params);
  std::cout << "loading..." << std::endl;
  auto alice = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube50x50.tobj");
  std::cout << "load done" << std::endl;
  for (int i = 0; i < alice->vertices.cols(); ++i)
    alice->vertices.col(i) -= fem::Vector<Real, 3>(0.25, 0.0, 0.0);
  fem::Matrix<Real, 3, Eigen::Dynamic> velocities(3, alice->vertices.cols());
  for (int i = 0; i < velocities.cols(); ++i)
    velocities.col(i) = fem::Vector<Real, 3>(10.0, 0.0, 0.0);
  return {std::move(alice), std::move(velocities), std::move(energy), Material{}.density};
}

void checkGradient(System &system) {
  std::cout << "start checking" << std::endl;
  VecXd current = system.currentConfig();
  auto symGrad = fem::symbolicDeformationEnergyGradient(system);
  auto numGrad = fem::numericalDeformationEnergyGradient(system);
  if ((numGrad - symGrad).template lpNorm<Eigen::Infinity>() > 1e-3) {
    std::cerr << "Symbolic and numerical gradients do not match\n";
    std::cerr << "Symbolic gradient:\n" << symGrad.transpose() << std::endl;
    std::cerr << "Numerical gradient:\n" << numGrad.transpose() << std::endl;
    exit(1);
  }
  VecXd update = VecXd::Random(system.dof()).normalized() * 1e-2;
  system.updateCurrentConfig(current + update);
  std::cout << update.transpose() << std::endl;
  auto symGrad2 = fem::symbolicDeformationEnergyGradient(system);
  auto numGrad2 = fem::numericalDeformationEnergyGradient(system);
  if ((numGrad2 - symGrad2).lpNorm<Eigen::Infinity>() > 1e-3) {
    std::cerr << "Symbolic and numerical gradients do not match after updating\n";
    std::cerr << "Symbolic gradient:\n" << symGrad2.transpose() << std::endl;
    std::cerr << "Numerical gradient:\n" << numGrad2.transpose() << std::endl;
    for (int i = 0; i < system.dof(); i++) {
      Real diff = symGrad2(i) - numGrad2(i);
      if (std::abs(diff) > 1e-3) {
        std::cerr << "Difference at " << i << ": " << symGrad2(i) - numGrad2(i) << std::endl;
        std::cout << "Possibly vertex ID: " << i / 3 << std::endl;
      }
    }
    exit(1);
  }
}

void checkIntegratorGradient(System &system, IpcImplicitEuler &integrator) {
}

int main() {
  auto system = std::make_unique<System>();
  system->addPrimitive(std::move(alicePrimitive()));
  std::cout << "add done" << std::endl;
  system->startSimulationPhase();
  auto integrator = std::make_unique<IpcImplicitEuler>(*system);
  checkGradient(*system);
  checkIntegratorGradient(*system, *integrator);
}