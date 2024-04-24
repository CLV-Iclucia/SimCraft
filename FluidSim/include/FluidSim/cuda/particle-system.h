//
// Created by creeper on 24-3-22.
//

#ifndef SIM_CRAFT_PARTICLE_SYSTEM_H
#define SIM_CRAFT_PARTICLE_SYSTEM_H
#include <memory>
#include <FluidSim/cuda/gpu-arrays.cuh>
#include <FluidSim/fluid-sim.h>
namespace fluid::cuda {
struct VelAccessor {
  Accessor<DeviceArray<Real>> vx;
  Accessor<DeviceArray<Real>> vy;
  Accessor<DeviceArray<Real>> vz;

  CUDA_DEVICE CUDA_FORCEINLINE
  double3 read(int idx) const {
    return make_double3(vx[idx], vy[idx], vz[idx]);
  }

  CUDA_DEVICE CUDA_FORCEINLINE
  void write(int idx, const double3& val) {
    vx[idx] = val.x;
    vy[idx] = val.y;
    vz[idx] = val.z;
  }
};

struct PosAccessor {
  Accessor<DeviceArray<Real>> px;
  Accessor<DeviceArray<Real>> py;
  Accessor<DeviceArray<Real>> pz;

  CUDA_DEVICE CUDA_FORCEINLINE
  double3 read(int idx) const {
    return make_double3(px[idx], py[idx], pz[idx]);
  }

  CUDA_DEVICE CUDA_FORCEINLINE
  void write(int idx, const double3& val) {
    px[idx] = val.x;
    py[idx] = val.y;
    pz[idx] = val.z;
  }
};

// Basic particle system, contains particle positions and velocities
struct ParticleSystem {
  int size() const { return vx->size(); }
  std::unique_ptr<DeviceArray<Real>> vx;
  std::unique_ptr<DeviceArray<Real>> vy;
  std::unique_ptr<DeviceArray<Real>> vz;
  std::unique_ptr<DeviceArray<Real>> px;
  std::unique_ptr<DeviceArray<Real>> py;
  std::unique_ptr<DeviceArray<Real>> pz;
  [[nodiscard]] VelAccessor velAccessor() const {
    return {vx->accessor(), vy->accessor(), vz->accessor()};
  }

  [[nodiscard]] PosAccessor posAccessor() const {
    return {px->accessor(), py->accessor(), pz->accessor()};
  }
};

// Derive from this class to implement particle systems with custom data
}
#endif //SIM_CRAFT_PARTICLE_SYSTEM_H