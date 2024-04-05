//
// Created by creeper on 24-3-22.
//

#ifndef SIM_CRAFT_PARTICLE_SYSTEM_H
#define SIM_CRAFT_PARTICLE_SYSTEM_H
#include <memory>
#include <FluidSim/cuda/gpu-arrays.h>

namespace fluid::cuda {
struct VelAccessor {
  DeviceArrayAccessor<double> vx;
  DeviceArrayAccessor<double> vy;
  DeviceArrayAccessor<double> vz;

  [[nodiscard]] double3 read(int idx) const {
    return make_double3(vx[idx], vy[idx], vz[idx]);
  }

  void write(int idx, const double3& val) {
    vx[idx] = val.x;
    vy[idx] = val.y;
    vz[idx] = val.z;
  }
};

struct PosAccessor {
  DeviceArrayAccessor<double> px;
  DeviceArrayAccessor<double> py;
  DeviceArrayAccessor<double> pz;

  [[nodiscard]] double3 read(int idx) const {
    return make_double3(px[idx], py[idx], pz[idx]);
  }

  void write(int idx, const double3& val) {
    px[idx] = val.x;
    py[idx] = val.y;
    pz[idx] = val.z;
  }
};

// Basic particle system, contains particle positions and velocities
struct ParticleSystem {
  int size() const { return vx->size(); }
  std::unique_ptr<DeviceArray<double>> vx;
  std::unique_ptr<DeviceArray<double>> vy;
  std::unique_ptr<DeviceArray<double>> vz;
  std::unique_ptr<DeviceArray<double>> px;
  std::unique_ptr<DeviceArray<double>> py;
  std::unique_ptr<DeviceArray<double>> pz;

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