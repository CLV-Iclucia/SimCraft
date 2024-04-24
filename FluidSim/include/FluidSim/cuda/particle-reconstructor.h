//
// Created by creeper on 24-3-22.
//

#ifndef SIM_CRAFT_PARTICLE_RECONSTRUCTOR_H
#define SIM_CRAFT_PARTICLE_RECONSTRUCTOR_H

#include <FluidSim/cuda/particle-system.h>
#include <FluidSim/cuda/gpu-arrays.cuh>

namespace fluid::cuda {
struct ParticleSystemReconstructor {
  ParticleSystemReconstructor() = default;
  virtual ~ParticleSystemReconstructor() = default;
  virtual void reconstruct(const ParticleSystem &partiles, Real radius,
                           CudaSurface<Real> &sdf, CudaSurface<uint8_t> &sdf_valid, int3 resolution, Real h) = 0;
};

struct NeighbourSearcher {
  NeighbourSearcher(int n, int level, const double3 &size)
      : resolution(1 << level),
        spacing(size.x / resolution, size.y / resolution, size.z / resolution) {
    particle_idx_mapping->resize(n);
    cell_begin_idx->resize(resolution * resolution * resolution);
    cell_end_idx->resize(resolution * resolution * resolution);
  }
  void update(const ParticleSystem &particles);

  int resolution;
  double3 spacing;
  double3 origin{};
  std::unique_ptr<DeviceArray<int>> particle_idx_mapping;
  std::unique_ptr<DeviceArray<uint32_t>> particle_cell_mapping;
  std::unique_ptr<DeviceArray<int>> cell_begin_idx;
  std::unique_ptr<DeviceArray<int>> cell_end_idx;
};

struct NaiveReconstructor : ParticleSystemReconstructor {
  void reconstruct(const ParticleSystem &particles, Real radius,
                   CudaSurface<Real> &sdf, CudaSurface<uint8_t> &sdf_valid, int3 resolution, Real h) override;
  std::unique_ptr<NeighbourSearcher> ns;
};
}

#endif //SIM_CRAFT_PARTICLE_RECONSTRUCTOR_H