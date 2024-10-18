//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_GPU_ARRAYS_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_GPU_ARRAYS_H_

#include <Core/properties.h>
#include <Core/cuda-utils.h>
#include <Core/properties.h>
#include <Core/core.h>
#include <Core/debug.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace fluid::cuda {

template<typename T>
struct CudaSurfaceAccessor {
  cudaSurfaceObject_t cuda_surf;

  template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  CUDA_DEVICE CUDA_FORCEINLINE T read(int x, int y, int z) {
    return surf3Dread<T>(cuda_surf, x * sizeof(T), y, z, mode);
  }

  template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  CUDA_DEVICE CUDA_FORCEINLINE T read(const int3 &idx) {
    return surf3Dread<T>(cuda_surf, idx.x * sizeof(T), idx.y, idx.z, mode);
  }

  template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  CUDA_DEVICE CUDA_FORCEINLINE void write(T val, int x, int y, int z) {
//    if constexpr (std::is_floating_point_v<T>)
//      assert(!isnan(val) && !isinf(val));
    surf3Dwrite<T>(val, cuda_surf, x * sizeof(T), y, z, mode);
  }
};

template<typename T>
__global__ void kernelFill(CudaSurfaceAccessor<T> surf, T val, uint n) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  if (x >= n || y >= n || z >= n)
    return;
  surf.write(val, x, y, z);
}

template<typename T>
__global__ void kernelCopy(CudaSurfaceAccessor<T> dst, CudaSurfaceAccessor<T> src, uint nx, uint ny, uint nz) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int z = threadIdx.z + blockDim.z * blockIdx.z;
  if (x >= nx || y >= ny || z >= nz)
    return;
  dst.write(src.read(x, y, z), x, y, z);
}

template<typename T>
struct DeviceArray;
template<typename T>
struct DeviceArrayAccessor {
  T *ptr;

  CUDA_DEVICE CUDA_FORCEINLINE const T &operator[](size_t idx) const {
    return ptr[idx];
  }

  CUDA_DEVICE CUDA_FORCEINLINE T &operator[](size_t idx) { return ptr[idx]; }
};

template<typename T>
struct ConstDeviceArrayAccessor {
  T *ptr;

  CUDA_DEVICE CUDA_FORCEINLINE const T &operator[](size_t idx) const {
    return ptr[idx];
  }
};
} // namespace core
namespace fluid::cuda {

template<typename T>
struct CudaArray3D : core::NonCopyable {
  cudaArray *cuda_array{};
  uint3 dim;

  explicit CudaArray3D(const uint3 &dim_)
      : dim(dim_) {
    cudaExtent extent = make_cudaExtent(dim.x, dim.y, dim.z);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    cudaMalloc3DArray(&cuda_array, &channel_desc, extent,
                      cudaArraySurfaceLoadStore);
  }
  CudaArray3D(CudaArray3D &&other) noexcept
      : cuda_array(other.cuda_array), dim(other.dim) {
    other.cuda_array = nullptr;
  }
  CudaArray3D &operator=(CudaArray3D &&other) noexcept {
    if (this != &other) {
      cuda_array = other.cuda_array;
      dim = other.dim;
      other.cuda_array = nullptr;
    }
    return *this;
  }
  void copyFrom(T *data) {
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcPtr =
        make_cudaPitchedPtr(static_cast<void *>(data), dim.x * sizeof(T), dim.x, dim.y);
    copy3DParams.dstArray = cuda_array;
    copy3DParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
    copy3DParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy3DParams);
  }
  void copyFrom(const std::vector<T> &vec) {
    copyFrom(vec.data());
  }
  void copyFrom(const CudaArray3D<T> &other) {
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcArray = other.cuda_array;
    copy3DParams.dstArray = cuda_array;
    copy3DParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
    copy3DParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copy3DParams);
  }
  void copyTo(T *data) {
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcArray = cuda_array;
    copy3DParams.dstPtr =
        make_cudaPitchedPtr(static_cast<void *>(data), dim.x * sizeof(T), dim.x,
                            dim.y);
    copy3DParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
    copy3DParams.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copy3DParams);
  }
  void copyTo(std::vector<T> &vec) {
    vec.resize(dim.x * dim.y * dim.z);
    copyTo(vec.data());
  }

  [[nodiscard]] cudaArray *Array() const { return cuda_array; }

  ~CudaArray3D() { cudaFreeArray(cuda_array); }
};

template<typename T>
struct CudaSurface : CudaArray3D<T> {
  using CudaArray3D<T>::dim;
  cudaSurfaceObject_t cuda_surf{};

  explicit CudaSurface(const uint3 &dim_)
      : CudaArray3D<T>(dim_) {
    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = CudaArray3D<T>::Array();
    cudaCreateSurfaceObject(&cuda_surf, &res_desc);
  }

  [[nodiscard]] cudaSurfaceObject_t surface() const { return cuda_surf; }
  void fill(const T &val) {
    kernelFill<<<dim3((dim.x + 7) / 8, (dim.y + 7) / 8, (dim.z + 7) / 8),
    dim3(8, 8, 8)>>>(surfaceAccessor(), val, dim.x);
  }
  void copyFrom(const CudaSurface<T> &data) {
    if (dim.x != data.dim.x || dim.y != data.dim.y || dim.z != data.dim.z)
      ERROR("Dimension mismatch");
    kernelCopy<<<dim3((dim.x + 7) / 8, (dim.y + 7) / 8, (dim.z + 7) / 8),
    dim3(8, 8, 8)>>>(surfaceAccessor(), data.surfaceAccessor(), dim.x, dim.y, dim.z);
  }
  CudaSurfaceAccessor<T> surfaceAccessor() const { return {cuda_surf}; }
  ~CudaSurface() { cudaDestroySurfaceObject(cuda_surf); }
};

template<class T>
struct CudaTextureAccessor {
  cudaTextureObject_t m_cuTex;
  __device__ __forceinline__ T sample(double x, double y, double z) const {
    return tex3D<T>(m_cuTex, static_cast<float>(x), static_cast<float>(y),
                    static_cast<float>(z));
  }
  __device__ __forceinline__ T sample(float x, float y, float z) const {
    return tex3D<T>(m_cuTex, x, y, z);
  }
  __device__ __forceinline__ T sample(const float3 &pos) const {
    return tex3D<T>(m_cuTex, pos.x, pos.y, pos.z);
  }
};

template<class T>
struct CudaTexture : CudaSurface<T> {
  struct Parameters {
    cudaTextureAddressMode addressMode{cudaAddressModeClamp};
    cudaTextureFilterMode filterMode{cudaFilterModeLinear};
    cudaTextureReadMode readMode{cudaReadModeElementType};
    bool normalizedCoords{false};
  };

  cudaTextureObject_t cuda_tex{};

  explicit CudaTexture(uint3 const &_dim, Parameters const &_args = {})
      : CudaSurface<T>(_dim) {
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = CudaSurface<T>::Array();

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = _args.addressMode;
    texDesc.addressMode[1] = _args.addressMode;
    texDesc.addressMode[2] = _args.addressMode;
    texDesc.filterMode = _args.filterMode;
    texDesc.readMode = _args.readMode;
    texDesc.normalizedCoords = _args.normalizedCoords;

    cudaCreateTextureObject(&cuda_tex, &resDesc, &texDesc, nullptr);
  }

  [[nodiscard]] cudaTextureObject_t texture() const { return cuda_tex; }

  CudaTextureAccessor<T> texAccessor() const { return {cuda_tex}; }

  ~CudaTexture() { cudaDestroyTextureObject(cuda_tex); }
};

template<typename T>
struct DeviceArray;

using core::Accessor;
using core::ConstAccessor;
template<typename T>
struct DeviceArray {
  DeviceArray() = default;

  CUDA_CALLABLE DeviceArray &operator=(DeviceArray &&other) noexcept {
    if (this != &other) {
      m_size = other.m_size;
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  CUDA_CALLABLE DeviceArray(DeviceArray &&other) noexcept
      : m_size(other.m_size), ptr(other.ptr) {
    other.ptr = nullptr;
  }

  // constructor
  CUDA_CALLABLE explicit DeviceArray(size_t size_) : m_size(size_) {
    cudaMalloc(&ptr, m_size * sizeof(T));
  }

  // construct from vector
  explicit DeviceArray(const std::vector<T> &vec)
      : m_size(vec.size()), ptr(nullptr) {
    cudaMalloc(&ptr, m_size * sizeof(T));
    cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
  }

  // construct from std::array
  template<size_t N>
  explicit DeviceArray(const std::array<T, N> &arr)
      : m_size(N), ptr(nullptr) {
    cudaMalloc(&ptr, m_size * sizeof(T));
    cudaMemcpy(ptr, arr.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
  }

  CUDA_CALLABLE~DeviceArray() { cudaFree(ptr); }

  [[nodiscard]] T *data() const { return ptr; }

  [[nodiscard]] size_t size() const { return m_size; }

  void copyFrom(const T *data) {
    cudaMemcpy(ptr, data, m_size * sizeof(T), cudaMemcpyHostToDevice);
  }

  void copyFrom(const std::vector<T> &vec) {
    m_size = vec.size();
    if (ptr)
      cudaFree(ptr);
    cudaMalloc(&ptr, m_size * sizeof(T));
    cudaMemcpy(ptr, vec.data(), m_size * sizeof(T), cudaMemcpyHostToDevice);
  }

  void copyTo(T *data) {
    cudaMemcpy(data, ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void copyTo(std::vector<T> &vec) const {
    vec.resize(m_size);
    cudaMemcpy(vec.data(), ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void resize(size_t size_) {
    if (m_size == size_) return;
    if (ptr)
      cudaFree(ptr);
    m_size = size_;
    cudaMalloc(&ptr, m_size * sizeof(T));
  }

  CUDA_HOST CUDA_FORCEINLINE DeviceArrayAccessor<T> accessor() const {
    return {ptr};
  }

  CUDA_HOST CUDA_FORCEINLINE ConstDeviceArrayAccessor<T> constAccessor() const {
    return {ptr};
  }

  CUDA_DEVICE CUDA_FORCEINLINE T &operator[](size_t idx) { return ptr[idx]; }

  CUDA_DEVICE CUDA_FORCEINLINE const T &operator[](size_t idx) const {
    return ptr[idx];
  }

  T *begin() { return ptr; }

  T *end() { return ptr + m_size; }

 private:
  uint32_t m_size{};
  T *ptr{};
};
} // namespace fluid
#endif // SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_GPU_ARRAYS_H_