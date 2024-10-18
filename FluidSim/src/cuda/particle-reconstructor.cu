//
// Created by creeper on 4/22/24.
//
#include <FluidSim/cuda/particle-reconstructor.h>
#include <FluidSim/cuda/vec-op.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
namespace fluid::cuda {

static uint32_t CUDA_DEVICE CUDA_FORCEINLINE morton3D(uint32_t x, uint32_t y, uint32_t z) {
  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8)) & 0x0300F00F;
  x = (x | (x << 4)) & 0x030C30C3;
  x = (x | (x << 2)) & 0x09249249;

  y = (y | (y << 16)) & 0x030000FF;
  y = (y | (y << 8)) & 0x0300F00F;
  y = (y | (y << 4)) & 0x030C30C3;
  y = (y | (y << 2)) & 0x09249249;

  z = (z | (z << 16)) & 0x030000FF;
  z = (z | (z << 8)) & 0x0300F00F;
  z = (z | (z << 4)) & 0x030C30C3;
  z = (z | (z << 2)) & 0x09249249;

  return x | (y << 1) | (z << 2);
}
static void CUDA_GLOBAL kernelInitParticleMapping(int n,
                                                  int resolution,
                                                  double3 spacing,
                                                  PosAccessor positions,
                                                  DeviceArrayAccessor<int> particle_idx_mapping,
                                                  DeviceArrayAccessor<uint32_t> particle_cell_mapping) {
  get_and_restrict_tid(idx, n);
  particle_idx_mapping[idx] = idx;
  int idx_x = floor(positions.px[idx] / spacing.x);
  idx_x = max(min(idx_x, resolution - 1), 0);
  int idx_y = floor(positions.py[idx] / spacing.y);
  idx_y = max(min(idx_y, resolution - 1), 0);
  int idx_z = std::floor(positions.pz[idx] / spacing.z);
  idx_z = max(min(idx_z, resolution - 1), 0);
  uint32_t cell_idx = morton3D(idx_x, idx_y, idx_z);
  particle_cell_mapping[idx] = cell_idx;
}

static void CUDA_GLOBAL kernelMarkCellBeginEnd(int n,
                                               DeviceArrayAccessor<uint32_t> particle_cell_mapping,
                                               DeviceArrayAccessor<int> cell_begin_idx,
                                               DeviceArrayAccessor<int> cell_end_idx) {
  get_and_restrict_tid(idx, n);
  uint32_t grid_idx = particle_cell_mapping[idx];
  if (idx == 0 || particle_cell_mapping[idx] != grid_idx)
    cell_begin_idx[grid_idx] = idx;
  if (idx == n - 1 || particle_cell_mapping[idx] != grid_idx)
    cell_end_idx[grid_idx] = idx;
}

void NeighbourSearcher::update(const fluid::cuda::ParticleSystem &particles) {
  int n = particles.size();
  particle_idx_mapping->resize(n);
  particle_cell_mapping->resize(n);
  cudaSafeCheck((kernelInitParticleMapping<<<LAUNCH_THREADS(n)>>>(
      n,
      resolution,
      spacing,
      particles.posAccessor(),
      particle_idx_mapping->accessor(),
      particle_cell_mapping->accessor())));
  auto key_begin = thrust::device_ptr<uint32_t>(particle_cell_mapping->begin());
  auto key_end = key_begin + n;
  thrust::sort_by_key(key_begin, key_end, particle_idx_mapping->begin());
  thrust::fill(cell_begin_idx->begin(), cell_begin_idx->end(), -1);
  thrust::fill(cell_end_idx->begin(), cell_end_idx->end(), -1);
  cudaSafeCheck((kernelMarkCellBeginEnd<<<LAUNCH_THREADS(n)>>>(
      n,
      particle_cell_mapping->accessor(),
      cell_begin_idx->accessor(),
      cell_end_idx->accessor())));
}

static void CUDA_GLOBAL kernelReconstructSdf(int3 sdfResolution,
                                             int resolution,
                                             double3 spacing,
                                             float h,
                                             float r,
                                             PosAccessor positions,
                                             DeviceArrayAccessor<int> particle_idx_mapping,
                                             DeviceArrayAccessor<int> cell_begin_idx,
                                             DeviceArrayAccessor<int> cell_end_idx,
                                             CudaSurfaceAccessor<float> sdf,
                                             CudaSurfaceAccessor<uint8_t> sdf_valid) {
  get_and_restrict_tid_3d(i, j, k, sdfResolution.x, sdfResolution.y, sdfResolution.z);
  auto p = make_double3((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h);
  int x_min = max(0, static_cast<int>(floor((p.x - r) / spacing.x)));
  int x_max = min(resolution - 1, static_cast<int>(ceil((p.x + r) / spacing.x)));
  int y_min = max(0, static_cast<int>(floor((p.y - r) / spacing.y)));
  int y_max = min(resolution - 1, static_cast<int>(ceil((p.y + r) / spacing.y)));
  int z_min = max(0, static_cast<int>(floor((p.z - r) / spacing.z)));
  int z_max = min(resolution - 1, static_cast<int>(ceil((p.z + r) / spacing.z)));
  float val = 1e9;
  bool flag = false;
  for (int x = x_min; x <= x_max; x++) {
    for (int y = y_min; y <= y_max; y++) {
      for (int z = z_min; z <= z_max; z++) {
        uint32_t cell_idx = morton3D(x, y, z);
        if (cell_begin_idx[cell_idx] == -1) continue;
        for (int idx = cell_begin_idx[cell_idx]; idx < cell_end_idx[cell_idx]; idx++) {
          int particle_idx = particle_idx_mapping[idx];
          float dis = distance(p, positions.read(particle_idx));
          val = min(val, dis - r);
          flag |= dis < r;
        }
      }
    }
  }
  sdf.write(val, i, j, k);
  sdf_valid.write(flag, i, j, k);
}

void NaiveReconstructor::reconstruct(const fluid::cuda::ParticleSystem &particles,
                                     float radius,
                                     CudaSurface<float> &sdf,
                                     CudaSurface<uint8_t> &sdf_valid,
                                     int3 sdfResolution,
                                     float h) {
  ns->update(particles);
  kernelReconstructSdf<<<LAUNCH_THREADS_3D(sdfResolution.x, sdfResolution.y, sdfResolution.z)>>>(
      sdfResolution,
      ns->resolution,
      ns->spacing,
      h,
      radius,
      particles.posAccessor(),
      ns->particle_idx_mapping->accessor(),
      ns->cell_begin_idx->accessor(),
      ns->cell_end_idx->accessor(),
      sdf.surfaceAccessor(),
      sdf_valid.surfaceAccessor());
}
}