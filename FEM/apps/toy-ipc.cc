#include <Deform/strain-energy-density.h>
#include <fem/tet-mesh.h>
#include <fem/system.h>
#include <fem/ipc/implicit-euler.h>
#include <ogl-render/window.h>
#include <Maths/svd.h>
using namespace fem;
using namespace deform;

template<int Dim>
bool checkSPD(const Eigen::Matrix<Real, Dim, Dim> &A) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, Dim, Dim>> es(A);
  return es.eigenvalues().minCoeff() >= 0.0;
}
template<int Dim>
bool checkSymmetric(const Eigen::Matrix<Real, Dim, Dim> &A) {
  return (A - A.transpose()).norm() < 1e-6;
}

struct Material {
  ElasticityParameters<Real> params{1e7, 0.45};
  Real density{1150.0};
};

TetPrimitiveConfig alicePrimitive() {
  std::unique_ptr<StrainEnergyDensity<Real>> energy = std::make_unique<StableNeoHookean<Real>>(Material{}.params);
  auto alice = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube10x10.tobj");
  fem::Matrix<Real, 3, Eigen::Dynamic> velocities(3, alice->vertices.cols());
  for (int i = 0; i < velocities.cols(); ++i)
    velocities.col(i) = fem::Vector<Real, 3>(10.0, 0.0, 0.0);
  return {std::move(alice), std::move(velocities), std::move(energy), Material{}.density};
}

TetPrimitiveConfig bobPrimitive() {
  std::unique_ptr<StrainEnergyDensity<Real>> energy = std::make_unique<StableNeoHookean<Real>>(Material{}.params);
  auto bob = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube10x10.tobj");
  for (int i = 0; i < bob->vertices.cols(); ++i)
    bob->vertices.col(i) += fem::Vector<Real, 3>(2.5, 0.0, 0.0);
  fem::Matrix<Real, 3, Eigen::Dynamic> velocities(3, bob->vertices.cols());
  for (int i = 0; i < velocities.cols(); ++i)
    velocities.col(i) = fem::Vector<Real, 3>(-10.0, 0.0, 0.0);
  return {std::move(bob), std::move(velocities), std::move(energy), Material{}.density};
}

int main() {
  auto system = std::make_unique<System>();
  system->addPrimitive(alicePrimitive());
  system->addPrimitive(bobPrimitive());
  system->startSimulationPhase();
  auto integrator = std::make_unique<IpcImplicitEuler>(*system);
  Real t = 0.0, dt = 0.001;
  std::cout << std::format("total energy: {:f} = {:f} (T) + {:f} (V)\n", system->totalEnergy(), system->kineticEnergy(), system->potentialEnergy());
  int frame = 0;
  while (t < 0.5) {
    integrator->step(dt);
    t += dt;
    frame++;
    std::cout << std::format("----------------------- t = {:f} -----------------\n", t);
    std::cout << std::format("total energy: {:f} = {:f} (T) + {:f} (V)\n", system->totalEnergy(), system->kineticEnergy(), system->potentialEnergy());
    if (frame % 10 == 0) {
      system->saveSurfaceObjFile(std::format("frame{:04d}.obj", frame));
    }
  }
}