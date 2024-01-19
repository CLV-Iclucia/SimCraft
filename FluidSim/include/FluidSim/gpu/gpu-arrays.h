//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_GPU_ARRAYS_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_GPU_ARRAYS_H_
#include <Core/properties.h>
#include <FluidSim/fluid-sim.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace fluid {
template <typename T>
struct CudaArray : core::NonCopyable {
  cudaArray* cuda_array{};
  uint3 dim;

  explicit CudaArray(const uint3& dim_)
    : dim(dim_) {
    cudaExtent extent = make_cudaExtent(dim.x, dim.y, dim.z);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
    cudaMalloc3DArray(&cuda_array, &channel_desc, extent,
                      cudaArraySurfaceLoadStore);
  }

  void copyFrom(const T* data) {
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcPtr =
        make_cudaPitchedPtr(static_cast<void*>(data), dim.x * sizeof(T), dim.x,
                            dim.y);
    copy3DParams.dstArray = cuda_array;
    copy3DParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
    copy3DParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy3DParams);
  }
  void copyTo(T* data) {
    cudaMemcpy3DParms copy3DParams{};
    copy3DParams.srcArray = cuda_array;
    copy3DParams.dstPtr =
        make_cudaPitchedPtr(static_cast<void*>(data), dim.x * sizeof(T), dim.x,
                            dim.y);
    copy3DParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
    copy3DParams.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copy3DParams);
  }
  [[nodiscard]] cudaArray* Array() const { return cuda_array; }
  cudaArray_t* ArrayPtr() { return &cuda_array; }
  ~CudaArray() { cudaFreeArray(cuda_array); }
};

template <typename T>
struct CudaSurfaceAccessor {
  cudaSurfaceObject_t cuda_surf;
  template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  __device__ __forceinline__ T read(int x, int y, int z) {
    return surf3Dread<T>(cuda_surf, x * sizeof(T), y, z, mode);
  }
  template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
  __device__ __forceinline__ void write(T val, int x, int y, int z) {
    surf3Dwrite<T>(val, cuda_surf, x * sizeof(T), y, z, mode);
  }
};

template <typename T>
struct CudaSurface : CudaArray<T> {
  cudaSurfaceObject_t cuda_surf{};
  explicit CudaSurface(const uint3& dim_)
    : CudaArray<T>(dim_) {
    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = CudaArray<T>::Array();
    cudaCreateSurfaceObject(&cuda_surf, &res_desc);
  }
  [[nodiscard]] cudaSurfaceObject_t surface() const { return cuda_surf; }
  CudaSurfaceAccessor<T> surfAccessor() const { return {cuda_surf}; }
  ~CudaSurface() { cudaDestroySurfaceObject(cuda_surf); }
};

template <class T>
struct CudaTextureAccessor {
  cudaTextureObject_t m_cuTex;

  __device__ __forceinline__ T sample(float x, float y, float z) const {
    return tex3D<T>(m_cuTex, x, y, z);
  }
};

template <class T>
struct CudaTexture : CudaSurface<T> {
  struct Parameters {
    cudaTextureAddressMode addressMode{cudaAddressModeClamp};
    cudaTextureFilterMode filterMode{cudaFilterModeLinear};
    cudaTextureReadMode readMode{cudaReadModeElementType};
    bool normalizedCoords{false};
  };

  cudaTextureObject_t cuda_tex{};

  explicit CudaTexture(uint3 const& _dim, Parameters const& _args = {})
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
} // namespace fluid
#endif // SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_GPU_ARRAYS_H_