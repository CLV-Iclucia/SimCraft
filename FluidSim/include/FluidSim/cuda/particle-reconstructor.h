//
// Created by creeper on 24-3-22.
//

#ifndef SIM_CRAFT_PARTICLE_RECONSTRUCTOR_H
#define SIM_CRAFT_PARTICLE_RECONSTRUCTOR_H

#include <FluidSim/cuda/particle-system.h>
#include <FluidSim/cuda/gpu-arrays.h>

namespace fluid::cuda {
struct ParticleSystemReconstructor {
  ParticleSystemReconstructor() = default;
  virtual ~ParticleSystemReconstructor() = default;
  virtual void reconstruct(int nParticles, PosAccessor pos, Real radius,
                           CudaSurfaceAccessor<double> dest_sdf) const = 0;
};

struct NaiveReconstructor {
  void reconstruct(int nParticles, PosAccessor pos, Real radius,
                   CudaSurfaceAccessor<double> dest_sdf) const {

  }
};
}

#endif //SDF_RECONSTRUCTOR_H