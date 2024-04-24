//
// Created by creeper on 10/25/23.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_NS_UTIL_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_NS_UTIL_H_

#include <Spatify/types.h>
#include <Spatify/arrays.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <span>

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
      : width(w), height(h), depth(d),
        spacing(size.x / w, size.y / h, size.z / d) {
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
      std::sort(particle_idx_mapping.begin(), particle_idx_mapping.end(),
                [this](int i, int j) {
                  return particle_cell_mapping[i] < particle_cell_mapping[j];
                });
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
    template <typename Func>
    void forNeighbours(const Vec3d& pos, std::span<Vec3d> positions, Real r,
                       Func&& f) {
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
    Vector<T, 3> spacing;
    Vector<T, 3> origin{};
    std::vector<int> particle_idx_mapping;
    std::vector<int> particle_cell_mapping;
    std::vector<int> cell_begin_idx;
    std::vector<int> cell_end_idx;
};
}
#endif //SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_NS_UTIL_H_