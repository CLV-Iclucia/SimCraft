#include <FluidSim/cpu/fluid-simulator.h>
#include <FluidSim/cuda/fluid-simulator.h>
#include <Core/debug.h>
namespace fluid {
void FluidSimulator::setBackend(Backend backend_type) {
  if (backend_type == Backend::CPU) {
    backend = std::make_unique<cpu::FluidSimulator>(config.nParticles, config.size, config.resolution);
  } else if (backend_type == Backend::CUDA) {
    backend = std::make_unique<cuda::FluidSimulator>(config.nParticles, config.size, config.resolution);
  } else
    ERROR("Invalid backend");
}
void cuda::FluidSimulator::setInitialFluid(const Mesh &fluid_mesh) {

}
}