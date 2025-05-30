#include <FluidSim/cpu/fluid-simulator.h>
#include <Core/debug.h>
namespace fluid {
void FluidSimulator::setBackend(Backend backend_type) {
  if (backend_type == Backend::CPU) {
    backend = std::make_unique<cpu::FluidSimulator>(config.nParticles, config.size, config.resolution);
  } else
    ERROR("Invalid backend");
}
}