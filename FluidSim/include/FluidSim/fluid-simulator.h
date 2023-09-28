//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
#include <Core/animation.h>
#include <Core/timer.h>
#include <FluidSim/fluid-sim.h>
namespace fluid {

class FluidSimulator2D : public core::Animation {
public:
  FluidSimulator2D() = default;
  virtual ~FluidSimulator2D() = default;
  virtual void init() = 0;
};

class ApicSimulator2D : public FluidSimulator2D {
public:
  core::Timer timer;
  ~ApicSimulator2D() override = default;
  void init() override;
  void step(core::Frame &frame) override;

private:
  int nParticles = 0;
  struct ParticleList {
  } m_particles;
};
class ApicSimulator3D : public FluidSimulator2D {
 public:
  core::Timer timer;
  ~ApicSimulator3D() override = default;
  void init() override;
  void step(core::Frame &frame) override;

 private:
  int nParticles = 0;
  struct ParticleList {
  } m_particles;
};
} // namespace fluid
#endif // SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_FLUID_SIMULATOR_H_
