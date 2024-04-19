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
  virtual void reconstruct(const ParticleSystem &partiles, Real radius,
                           CudaSurface<Real> &dest_sdf, int3 resolution, Real h) const = 0;
};

class NeighbourSearcher {
 public:
  NeighbourSearcher(int n, int w, int h, int d, const Vector<T, 3> &size)
      : width(w), height(h), depth(d),
        spacing(size.x / w, size.y / h, size.z / d) {
    particle_idx_mapping.resize(n);
    cell_begin_idx.resize(width * height * depth);
    cell_end_idx.resize(width * height * depth);
  }
  void resetGrid(int w, int h, int d, const Vector<T, 3> &s) {
    width = w;
    height = h;
    depth = d;
    spacing = s;
    cell_begin_idx.resize(width * height * depth);
    cell_end_idx.resize(width * height * depth);
  }
  void update(const ParticleSystem& particles) {
    int n = particles.size();
    particle_idx_mapping.resize(n);
    particle_cell_mapping.resize(n);
    for (int i = 0; i < n; i++) {
      particle_idx_mapping[i] = i;
      int idx_x = std::floor(particles[i].x / spacing.x);
      if (idx_x < 0) idx_x = 0;
      if (idx_x >= width) idx_x = width - 1;
      int idx_y = std::floor(particles[i].y / spacing.y);
      if (idx_y < 0) idx_y = 0;
      if (idx_y >= height) idx_y = height - 1;
      int idx_z = std::floor(particles[i].z / spacing.z);
      if (idx_z < 0) idx_z = 0;
      if (idx_z >= depth) idx_z = depth - 1;
      int cell_idx = static_cast<int>(idx_x * height * depth + idx_y * depth + idx_z);
      particle_cell_mapping[i] = cell_idx;
      assert(cell_idx >= 0 && cell_idx < width * height * depth);
    }
    std::ranges::sort(particle_cell_mapping);
    for (int i = 0; i < cell_begin_idx.size(); i++)
      cell_begin_idx[i] = cell_end_idx[i] = -1;
    for (int i = 0; i < n; i++) {
      int grid_idx = particle_cell_mapping[i];
      if (i == 0 || particle_cell_mapping[i - 1] != grid_idx)
        cell_begin_idx[grid_idx] = i;
      if (i == n - 1 || particle_cell_mapping[i + 1] != grid_idx)
        cell_end_idx[grid_idx] = i;
    }
  }

  template<typename Func>
  void forNeighbours(const Vec3d &pos, std::span<Vec3d> positions, Real r,
                     Func &&f) {
    int x_min = std::max(
        0, static_cast<int>(std::floor((pos.x - r) / spacing.x)));
    int x_max = std::min(width - 1,
                         static_cast<int>(std::ceil(
                             (pos.x + r) / spacing.x)));
    int y_min = std::max(
        0, static_cast<int>(std::floor((pos.y - r) / spacing.y)));
    int y_max = std::min(height - 1,
                         static_cast<int>(std::ceil(
                             (pos.y + r) / spacing.y)));
    int z_min = std::max(
        0, static_cast<int>(std::floor((pos.z - r) / spacing.z)));
    int z_max = std::min(depth - 1,
                         static_cast<int>(std::ceil(
                             (pos.z + r) / spacing.z)));
    for (int i = x_min; i <= x_max; i++) {
      for (int j = y_min; j <= y_max; j++) {
        for (int k = z_min; k <= z_max; k++) {
          int cell_idx = i * height * depth + j * depth + k;
          assert(cell_begin_idx[cell_idx] <= cell_end_idx[cell_idx]);
          for (int l = cell_begin_idx[cell_idx]; l < cell_end_idx[cell_idx]; l++)
            if (glm::distance(positions[particle_idx_mapping[l]], pos) <= r)
              f(particle_idx_mapping[l]);
        }
      }
    }
  }

 private:
  int width = 0, height = 0, depth = 0;
  double3 spacing;
  double3 origin{};
  std::unique_ptr<DeviceArray<int>> particle_idx_mapping;
  std::unique_ptr<DeviceArray<int>> particle_cell_mapping;
  std::unique_ptr<DeviceArray<int>> cell_begin_idx;
  std::unique_ptr<DeviceArray<int>> cell_end_idx;
};

struct NaiveReconstructor : ParticleSystemReconstructor {

  void reconstruct(const ParticleSystem &particles, Real radius,
                   CudaSurface<Real> &dest_sdf, int3 resolution, Real h) const override {
    ns.update(particles);
    sdf.grid.fill(1e9);
    sdfValid.fill(false);
    sdf.grid.parallelForEach([&](int i, int j, int k) {
      Vector<T, 3> p = sdf.grid.indexToCoord(i, j, k);
      ns.forNeighbours(p, particles, 4 * radius, [&](int idx) {
        Real dis = glm::distance(p, particles[idx]);
        assert(dis < 4 * radius);
        sdf(i, j, k) = std::min(sdf(i, j, k), dis - radius);
        sdfValid(i, j, k) = true;
      });
    });
  }
  std::unique_ptr<NeighbourSearcher> ns;
};
}

#endif //SDF_RECONSTRUCTOR_H