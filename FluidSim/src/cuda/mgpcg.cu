//
// Created by creeper on 4/25/24.
//
#include <FluidSim/cuda/mgpcg.cuh>
#include <FluidSim/cuda/utils.h>
#include <Core/debug.h>
namespace fluid::cuda {
__constant__ double kTransferWights[4][4][4];
// 0 stands for solid, 1 stands for fluid
__global__ void PrecomputeDownSampleKernel(CudaSurfaceAccessor<uint8_t> surf,
                                           CudaSurfaceAccessor<uint8_t> surf_nxt, uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  uint8_t val_1 = surf.read<cudaBoundaryModeZero>(x * 2, y * 2, z * 2);
  uint8_t val_2 = surf.read<cudaBoundaryModeZero>(x * 2 + 1, y * 2, z * 2);
  uint8_t val_3 = surf.read<cudaBoundaryModeZero>(x * 2, y * 2 + 1, z * 2);
  uint8_t val_4 = surf.read<cudaBoundaryModeZero>(x * 2 + 1, y * 2 + 1, z * 2);
  uint8_t val_5 = surf.read<cudaBoundaryModeZero>(x * 2, y * 2, z * 2 + 1);
  uint8_t val_6 = surf.read<cudaBoundaryModeZero>(x * 2 + 1, y * 2, z * 2 + 1);
  uint8_t val_7 = surf.read<cudaBoundaryModeZero>(x * 2, y * 2 + 1, z * 2 + 1);
  uint8_t val_8 = surf.read<cudaBoundaryModeZero>(x * 2 + 1, y * 2 + 1, z * 2 + 1);
  surf_nxt.write(val_1 && val_2 && val_3 && val_4 && val_5 && val_6 && val_7 && val_8, x, y, z);
}
static __global__ void ComputeResidualKernel(CudaSurfaceAccessor<float> u,
                                             CudaSurfaceAccessor<float> b,
                                             CudaSurfaceAccessor<float> r,
                                             CudaSurfaceAccessor<uint8_t> active, uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z)) return;
  float u_old = u.read(x, y, z);
  uint8_t axp = active.read<cudaBoundaryModeZero>(x - 1, y, z);
  uint8_t axn = active.read<cudaBoundaryModeZero>(x + 1, y, z);
  uint8_t ayp = active.read<cudaBoundaryModeZero>(x, y - 1, z);
  uint8_t ayn = active.read<cudaBoundaryModeZero>(x, y + 1, z);
  uint8_t azp = active.read<cudaBoundaryModeZero>(x, y, z - 1);
  uint8_t azn = active.read<cudaBoundaryModeZero>(x, y, z + 1);
  auto cnt = static_cast<double>(axp + axn + ayp + ayn + azp + azn);
  float pxp = static_cast<float>(axp) * u.read<cudaBoundaryModeClamp>(x - 1, y, z);
  float pxn = static_cast<float>(axn) * u.read<cudaBoundaryModeClamp>(x + 1, y, z);
  float pyp = static_cast<float>(ayp) * u.read<cudaBoundaryModeClamp>(x, y - 1, z);
  float pyn = static_cast<float>(ayn) * u.read<cudaBoundaryModeClamp>(x, y + 1, z);
  float pzp = static_cast<float>(azp) * u.read<cudaBoundaryModeClamp>(x, y, z - 1);
  float pzn = static_cast<float>(azn) * u.read<cudaBoundaryModeClamp>(x, y, z + 1);
  r.write(b.read(x, y, z) - (pxp + pxn + pyp + pyn + pzp + pzn), x, y, z);
}
__global__ void RestrictKernel(CudaSurfaceAccessor<float> u,
                               CudaSurfaceAccessor<float> uc, uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  double sum = 0.0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++)
        sum += kTransferWights[i][j][k] * u.read<cudaBoundaryModeZero>(x * 2 + i - 1, y * 2 + j - 1, z * 2 + k - 1);
  uc.write(static_cast<float>(sum), x, y, z);
}
__global__ void ProlongateKernel(CudaSurfaceAccessor<float> uc,
                                 CudaSurfaceAccessor<uint8_t> active,
                                 CudaSurfaceAccessor<float> u, uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  if (!active.read(x, y, z)) return;
  double sum = u.read(x, y, z);
  // use trilinear interpolation
  int x0 = (x - 1) / 2;
  int y0 = (y - 1) / 2;
  int z0 = (z - 1) / 2;
  auto active_000 = active.read<cudaBoundaryModeZero>(x0, y0, z0);
  auto active_100 = active.read<cudaBoundaryModeZero>(x0 + 1, y0, z0);
  auto active_010 = active.read<cudaBoundaryModeZero>(x0, y0 + 1, z0);
  auto active_110 = active.read<cudaBoundaryModeZero>(x0 + 1, y0 + 1, z0);
  auto active_001 = active.read<cudaBoundaryModeZero>(x0, y0, z0 + 1);
  auto active_101 = active.read<cudaBoundaryModeZero>(x0 + 1, y0, z0 + 1);
  auto active_011 = active.read<cudaBoundaryModeZero>(x0, y0 + 1, z0 + 1);
  auto active_111 = active.read<cudaBoundaryModeZero>(x0 + 1, y0 + 1, z0 + 1);
  auto tx = x * 0.5 - x0 - 0.25;
  auto ty = y * 0.5 - y0 - 0.25;
  auto tz = z * 0.5 - z0 - 0.25;
  auto w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz);
  auto w100 = tx * (1.0 - ty) * (1.0 - tz);
  auto w010 = (1.0 - tx) * ty * (1.0 - tz);
  auto w110 = tx * ty * (1.0 - tz);
  auto w001 = (1.0 - tx) * (1.0 - ty) * tz;
  auto w101 = tx * (1.0 - ty) * tz;
  auto w011 = (1.0 - tx) * ty * tz;
  auto w111 = tx * ty * tz;
  sum += w000 * uc.read<cudaBoundaryModeZero>(x0, y0, z0) * active_000;
  sum += w100 * uc.read<cudaBoundaryModeZero>(x0 + 1, y0, z0) * active_100;
  sum += w010 * uc.read<cudaBoundaryModeZero>(x0, y0 + 1, z0) * active_010;
  sum += w110 * uc.read<cudaBoundaryModeZero>(x0 + 1, y0 + 1, z0) * active_110;
  sum += w001 * uc.read<cudaBoundaryModeZero>(x0, y0, z0 + 1) * active_001;
  sum += w101 * uc.read<cudaBoundaryModeZero>(x0 + 1, y0, z0 + 1) * active_101;
  sum += w011 * uc.read<cudaBoundaryModeZero>(x0, y0 + 1, z0 + 1) * active_011;
  sum += w111 * uc.read<cudaBoundaryModeZero>(x0 + 1, y0 + 1, z0 + 1) * active_111;
  u.write(static_cast<float>(sum), x, y, z);
}
__global__ void DampedJacobiKernel(CudaSurfaceAccessor<float> u,
                                   CudaSurfaceAccessor<float> u_buf,
                                   CudaSurfaceAccessor<uint8_t> active,
                                   CudaSurfaceAccessor<float> f, uint n) {
  get_and_restrict_tid_3d(x, y, z, n, n, n);
  float u_old = u.read(x, y, z);
  uint8_t axp = active.read<cudaBoundaryModeZero>(x - 1, y, z);
  uint8_t axn = active.read<cudaBoundaryModeZero>(x + 1, y, z);
  uint8_t ayp = active.read<cudaBoundaryModeZero>(x, y - 1, z);
  uint8_t ayn = active.read<cudaBoundaryModeZero>(x, y + 1, z);
  uint8_t azp = active.read<cudaBoundaryModeZero>(x, y, z - 1);
  uint8_t azn = active.read<cudaBoundaryModeZero>(x, y, z + 1);
  auto cnt = static_cast<double>(axp + axn + ayp + ayn + azp + azn);
  float pxp = static_cast<float>(axp) * u.read<cudaBoundaryModeClamp>(x - 1, y, z);
  float pxn = static_cast<float>(axn) * u.read<cudaBoundaryModeClamp>(x + 1, y, z);
  float pyp = static_cast<float>(ayp) * u.read<cudaBoundaryModeClamp>(x, y - 1, z);
  float pyn = static_cast<float>(ayn) * u.read<cudaBoundaryModeClamp>(x, y + 1, z);
  float pzp = static_cast<float>(azp) * u.read<cudaBoundaryModeClamp>(x, y, z - 1);
  float pzn = static_cast<float>(azn) * u.read<cudaBoundaryModeClamp>(x, y, z + 1);
  float div = f.read(x, y, z);
  u_buf.write(
      (1.0 - kDampedJacobiOmega) * static_cast<double>(u_old) +
          kDampedJacobiOmega * static_cast<double>((pxp + pxn + pyp + pyn + pzp + pzn - div) / cnt),
      x, y, z);
}

static void smooth(const std::unique_ptr<CudaSurface<uint8_t>> &active,
                   std::unique_ptr<CudaSurface<float>> &u,
                   std::unique_ptr<CudaSurface<float>> &uBuf,
                   std::unique_ptr<CudaSurface<float>> &b,
                   int n) {
  for (int iter = 0; iter < kSmoothingIters; iter++) {
    DampedJacobiKernel<<<LAUNCH_THREADS_3D(n, n, n)>>>(u->surfaceAccessor(), uBuf->surfaceAccessor(),
                                                       active->surfaceAccessor(), b->surfaceAccessor(), n);
    std::swap(u, uBuf);
  }
}
static __global__ void BottomSolveKernel(CudaSurfaceAccessor<float> u,
                                         CudaSurfaceAccessor<float> b,
                                         CudaSurfaceAccessor<uint8_t> active,
                                         uint n) {
  int tid = ktid(x);
  int x = tid / n;
  int y = (tid - x * n) / n;
  int z = tid % n;
  __shared__ float u_shared[2][8][8][8];
  __shared__ float b_shared[8][8][8];
  __shared__ uint8_t active_shared[8][8][8];
  uint8_t cur = 0;
  u_shared[cur][x][y][z] = u.read(x, y, z);
  b_shared[x][y][z] = b.read(x, y, z);
  active_shared[x][y][z] = active.read(x, y, z);
  __syncthreads();
  for (int i = 0; i < kBottomSolveIters; i++) {
    float u_old = u.read(x, y, z);
    uint8_t axp = active_shared[max(x - 1, 0)][y][z];
    uint8_t axn = active_shared[min(x + 1, n - 1)][y][z];
    uint8_t ayp = active_shared[x][max(y - 1, 0)][z];
    uint8_t ayn = active_shared[x][min(y + 1, n - 1)][z];
    uint8_t azp = active_shared[x][y][max(z - 1, 0)];
    uint8_t azn = active_shared[x][y][min(z + 1, n - 1)];
    auto cnt = static_cast<double>(axp + axn + ayp + ayn + azp + azn);
    float pxp = static_cast<float>(axp) * u_shared[cur][max(x - 1, 0)][y][z];
    float pxn = static_cast<float>(axn) * u_shared[cur][min(x + 1, n - 1)][y][z];
    float pyp = static_cast<float>(ayp) * u_shared[cur][x][max(y - 1, 0)][z];
    float pyn = static_cast<float>(ayn) * u_shared[cur][x][min(y + 1, n - 1)][z];
    float pzp = static_cast<float>(azp) * u_shared[cur][x][y][max(z - 1, 0)];
    float pzn = static_cast<float>(azn) * u_shared[cur][x][y][min(z + 1, n - 1)];
    float div = b_shared[x][y][z];
    u_shared[cur ^ 1][x][y][z] = (1.0 - kDampedJacobiOmega) * static_cast<double>(u_old) +
        kDampedJacobiOmega * static_cast<double>((pxp + pxn + pyp + pyn + pzp + pzn - div) / cnt);
    cur ^= 1;
    __syncthreads();
  }
  u.write(u_shared[cur][x][y][z], x, y, z);
}
// assume: n is the power of 2
// then for a bottom solve which is small enough, we can fit all the data into the shared memory
// and solve them using iterations in one kernel with a warp
static void bottomSolve(const std::unique_ptr<CudaSurface<uint8_t>> &active,
                        const std::unique_ptr<CudaSurface<float>> &u,
                        const std::unique_ptr<CudaSurface<float>> &b,
                        int n) {
  if (n > 8)
    ERROR("bottom solve with n > 8 is not supported yet");
  BottomSolveKernel<<<1, n * n * n>>>(u->surfaceAccessor(), b->surfaceAccessor(),
                                      active->surfaceAccessor(), n);
}
void vCycle(std::array<std::unique_ptr<CudaSurface<uint8_t >>, kVcycleLevel> &active,
            std::array<std::unique_ptr<CudaSurface<float >>, kVcycleLevel> &u,
            std::array<std::unique_ptr<CudaSurface<float >>, kVcycleLevel> &uBuf,
            std::array<std::unique_ptr<CudaSurface<float >>, kVcycleLevel> &b,
            int n) {
  for (int l = 0; l < kVcycleLevel; l++) {
    int N = n >> l;
    smooth(active[l], u[l], uBuf[l], b[l], N);
    ComputeResidualKernel<<<LAUNCH_THREADS_3D(N, N, N)>>>(u[l]->surfaceAccessor(), b[l]->surfaceAccessor(),
                                                          uBuf[l]->surfaceAccessor(), active[l]->surfaceAccessor(), n);
    RestrictKernel<<<LAUNCH_THREADS_3D(N >> 1, N >> 1, N >> 1)>>>(uBuf[l]->surfaceAccessor(),
                                                                  uBuf[l + 1]->surfaceAccessor(),
                                                                  n);
  }
  bottomSolve(active[kVcycleLevel], u[kVcycleLevel], b[kVcycleLevel], n >> kVcycleLevel);
  for (int l = kVcycleLevel - 1; l >= 0; l--) {
    int N = n >> l;
    ProlongateKernel<<<LAUNCH_THREADS_3D(N, N, N)>>>(u[l + 1]->surfaceAccessor(),
                                                     active[l]->surfaceAccessor(),
                                                     u[l]->surfaceAccessor(), n);
    smooth(active[l], u[l], uBuf[l], b[l], N);
  }
}

void prepareWeights() {
  double weights[4][4][4];
  for (auto &wi : weights)
    for (auto &wij : wi)
      for (double &wijk : wij)
        wijk = 1.0;
  for (int i = 0; i < 4; i++)
    for (auto &wij : weights[i])
      for (auto &wijk : wij) {
        if (i == 0 || i == 3) wijk *= 0.125;
        else if (i == 1 || i == 2) wijk *= 0.375;
      }
  for (auto &wi : weights)
    for (int j = 0; j < 4; j++)
      for (auto &wijk : wi[j]) {
        if (j == 0 || j == 3) wijk *= 0.125;
        else if (j == 1 || j == 2) wijk *= 0.375;
      }
  for (auto &wi : weights)
    for (auto &wij : wi)
      for (int k = 0; k < 4; k++) {
        if (k == 0 || k == 3) wij[k] *= 0.125;
        else if (k == 1 || k == 2) wij[k] *= 0.375;
      }
  cudaMemcpyToSymbol(kTransferWights, weights, sizeof(weights));
}
}