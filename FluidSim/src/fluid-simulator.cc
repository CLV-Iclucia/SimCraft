#include <FluidSim/cpu/fluid-simulator.h>
#include <FluidSim/cuda/fluid-simulator.h>
#include <Core/debug.h>
namespace fluid {
void FluidSimulator::setBackend(const std::string& backend_name) {
  if (backend_name == "cpu") {
    backend = std::make_unique<cpu::FluidSimulator>(config.nParticles, config.size, config.resolution);
  } else if (backend_name == "cuda") {
    backend = std::make_unique<cuda::FluidSimulator>();
  } else
    ERROR("Invalid backend name: ", backend_name);
}
}