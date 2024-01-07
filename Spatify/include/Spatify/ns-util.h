//
// Created by creeper on 10/25/23.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_NS_UTIL_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_NS_UTIL_H_

#include <Spatify/types.h>
#include <Spatify/arrays.h>
#include <algorithm>
#include <cstring>

namespace spatify {
template <typename T, int Dim>
class NeighbourSearcher;

template <typename T>
class NeighbourSearcher<T, 2> {
  public:
    void init(int n_particles, int width_, int height_,
              const Vector<T, 2>& spacing_) {
      width = width_;
      height = height_;
      spacing = spacing_;
      particle_idx_mapping.resize(n_particles);
      cell_begin_idx.resize(width * height);
      cell_end_idx.resize(width * height);
    }
    void resize(int width_, int height_) {
      width = width_;
      height = height_;
      cell_begin_idx.resize(width * height);
      cell_end_idx.resize(width * height);
    }
    void update(int n_particles, Vector<T, 2>* particles) {
      particle_idx_mapping.reserve(n_particles);
      particle_cell_mapping.reserve(n_particles);
      for (int i = 0; i < n_particles; i++) {
        particle_idx_mapping[i] = i;
        particle_cell_mapping[i] =
            std::floor(particles[i].x / spacing.x) * height + std::floor(
                particles[i].x / spacing.y);
      }
      std::sort(particle_idx_mapping.begin(), particle_idx_mapping.end(),
                [this](int i, int j) {
                  return particle_cell_mapping[i] < particle_cell_mapping[j];
                });
      std::sort(particle_cell_mapping.begin(), particle_cell_mapping.end());
      for (int i = 0; i < n_particles; i++) {
        if (i > 0 && particle_cell_mapping[i - 1] != particle_cell_mapping[i]) {
          cell_begin_idx[particle_cell_mapping[i]] = i;
          cell_end_idx[particle_cell_mapping[i - 1]] = i;
        }
        if (i < n_particles - 1 && particle_cell_mapping[i + 1] !=
            particle_cell_mapping[i]) {
          cell_begin_idx[particle_cell_mapping[i]] = i;
          cell_end_idx[particle_cell_mapping[i + 1]] = i;
        }
      }
    }
    template <typename Func>
    void forNeighbours(const Vec2d& pos, Real r, Func&& f) {
      int x = std::floor(pos.x / spacing.x);
      int y = std::floor(pos.y / spacing.y);
      int x_min = std::max(0, x - 1);
      int x_max = std::min(width - 1, x + 1);
      int y_min = std::max(0, y - 1);
      int y_max = std::min(height - 1, y + 1);
      for (int i = x_min; i <= x_max; i++) {
        for (int j = y_min; j <= y_max; j++) {
          int cell_idx = i * height + j;
          for (int k = cell_begin_idx[cell_idx]; k < cell_end_idx[cell_idx];
               k++) {
            f(particle_idx_mapping[k]);
          }
        }
      }
    }

  private:
    int width = 0, height = 0;
    Vector<T, 2> spacing;
    std::vector<int> particle_idx_mapping;
    std::vector<int> particle_cell_mapping;
    std::vector<int> cell_begin_idx;
    std::vector<int> cell_end_idx;
};

template <typename T>
class NeighbourSearcher<T, 3> {
  public:
    NeighbourSearcher(int n, int w, int h, int d, const Vector<T, 3>& size)
      : width(w), height(h), depth(d), spacing(size.x / w, size.y / h, size.z / d) {
      particle_idx_mapping.resize(n);
      cell_begin_idx.resize(width * height * depth);
      cell_end_idx.resize(width * height * depth);
    }
    void resetGrid(int w, int h, int d, const Vector<T, 3>& s) {
      width = w;
      height = h;
      depth = d;
      spacing = s;
      cell_begin_idx.resize(width * height * depth);
      cell_end_idx.resize(width * height * depth);
    }
    void update(std::span<Vector<T, 3>> particles) {
      int n = particles.size();
      particle_idx_mapping.resize(n);
      particle_cell_mapping.resize(n);
      for (int i = 0; i < n; i++) {
        particle_idx_mapping[i] = i;
        particle_cell_mapping[i] =
            std::floor(particles[i].x / spacing.x) * height + std::floor(
                particles[i].x / spacing.y);
      }
      std::sort(particle_idx_mapping.begin(), particle_idx_mapping.end(),
                [this](int i, int j) {
                  return particle_cell_mapping[i] < particle_cell_mapping[j];
                });
      std::sort(particle_cell_mapping.begin(), particle_cell_mapping.end());
      for (int i = 0; i < n; i++) {
        if (i > 0 && particle_cell_mapping[i - 1] != particle_cell_mapping[i]) {
          cell_begin_idx[particle_cell_mapping[i]] = i;
          cell_end_idx[particle_cell_mapping[i - 1]] = i;
        }
        if (i < n - 1 && particle_cell_mapping[i + 1] !=
            particle_cell_mapping[i]) {
          cell_begin_idx[particle_cell_mapping[i]] = i;
          cell_end_idx[particle_cell_mapping[i + 1]] = i;
        }
      }
    }
    template <typename Func>
    void forNeighbours(const Vec3d& pos, Real r, Func&& f) {
      int x = std::floor(pos.x / spacing.x);
      int y = std::floor(pos.y / spacing.y);
      int z = std::floor(pos.z / spacing.z);
      int x_min = std::max(0, x - 1);
      int x_max = std::min(width - 1, x + 1);
      int y_min = std::max(0, y - 1);
      int y_max = std::min(height - 1, y + 1);
      int z_min = std::max(0, z - 1);
      int z_max = std::min(depth - 1, z + 1);
      for (int i = x_min; i <= x_max; i++) {
        for (int j = y_min; j <= y_max; j++) {
          for (int k = z_min; k <= z_max; k++) {
            int cell_idx = i * height * depth + j * depth + k;
            for (int l = cell_begin_idx[cell_idx]; l < cell_end_idx[cell_idx];
                 l++)
              f(particle_idx_mapping[l]);
          }
        }
      }
    }

  private:
    int width = 0, height = 0, depth = 0;
    Vector<T, 3> spacing;
    std::vector<int> particle_idx_mapping;
    std::vector<int> particle_cell_mapping;
    std::vector<int> cell_begin_idx;
    std::vector<int> cell_end_idx;
};
}
#endif //SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_NS_UTIL_H_