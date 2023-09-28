//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_GPU_ARRAYS_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_GPU_ARRAYS_H_
#include <FluidSim/fluid-sim.h>
#include <cuda_runtime.h>
#include <type_traits>
namespace fluid {

template <typename T> struct GpuArray1D : DisableCopy {
  explicit GpuArray1D(uint dim_) : dim(dim_) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaMallocArray(&m_array, &channelDesc, dim, 1);
  }
  void CopyFromHost(const T *data) {
    cudaMemcpyToArray(m_array, 0, 0, data, dim * sizeof(T),
                      cudaMemcpyHostToDevice);
  }
  void CopyFromDevice(const T *data) {
    cudaMemcpyToArray(m_array, 0, 0, data, dim * sizeof(T),
                      cudaMemcpyDeviceToDevice);
  }
  cudaArray *Array() const { return m_array; }
  ~GpuArray1D() { cudaFreeArray(m_array); }
  cudaArray *m_array = nullptr;
  uint dim;
};

template <typename T> struct GpuArray2D : DisableCopy {
  explicit GpuArray2D(uint2 dim_) : dim(dim_) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaMallocArray(&m_array, &channelDesc, dim.x, dim.y);
  }
  GpuArray2D(uint x, uint y) : dim(make_uint2(x, y)) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaMallocArray(&m_array, &channelDesc, dim.x, dim.y);
  }
  void CopyFromHost(const T *data) {
    cudaMemcpyToArray(m_array, 0, 0, data, dim.x * dim.y * sizeof(T),
                      cudaMemcpyHostToDevice);
  }
  void CopyFromDevice(const T *data) {
    cudaMemcpyToArray(m_array, 0, 0, data, dim.x * dim.y * sizeof(T),
                      cudaMemcpyDeviceToDevice);
  }
  cudaArray *Array() const { return m_array; }
  ~GpuArray2D() { cudaFreeArray(m_array); }
  cudaArray *m_array = nullptr;
  uint2 dim;
};
template <typename T> struct GpuArray3D : DisableCopy {
public:
  explicit GpuArray3D(uint3 dim_) : dim(dim_) {
    cudaExtent extent = make_cudaExtent(dim.x, dim.y, dim.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaMalloc3DArray(&m_array, &channelDesc, extent,
                      cudaArraySurfaceLoadStore);
  }
  GpuArray3D(uint x, uint y, uint z) : dim(make_uint3(x, y, z)) {
    cudaExtent extent = make_cudaExtent(dim.x, dim.y, dim.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaMalloc3DArray(&m_array, &channelDesc, extent,
                      cudaArraySurfaceLoadStore);
  }
  void CopyFromHost(const T *data) {
    cudaMemcpy3DParms copyParams{};
    copyParams.srcPtr =
        make_cudaPitchedPtr((void *)data, dim.x * sizeof(T), dim.x, dim.y);
    copyParams.dstArray = m_array;
    copyParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
  }
  cudaArray *array() const { return m_array; }
  ~GpuArray3D() { cudaFreeArray(m_array); }

private:
  cudaArray *m_array = nullptr;
  uint3 dim;
};

template <typename T, int Dim> struct GpuArrayWrapper {
  static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 2 or 3");
  using type = typename std::conditional<
      Dim == 2, GpuArray2D<T>,
      std::conditional<Dim == 1, GpuArray1D<T>, GpuArray3D<T>>>::type;
};

template <typename T, int Dim>
using GpuArray = typename GpuArrayWrapper<T, Dim>::type;

template <typename T> struct CudaSurfaceAccessor {
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

template <typename T> struct CudaSurface3D : GpuArray3D<T> {
  cudaSurfaceObject_t cuda_surf{};
  explicit CudaSurface3D(const uint3 &dim_) : GpuArray3D<T>(dim_) {
    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res = GpuArray3D<T>::Array();
    cudaCreateSurfaceObject(&cuda_surf, &res_desc);
  }
  cudaSurfaceObject_t Surface() const { return cuda_surf; }
  CudaSurfaceAccessor<T> SurfAccessor() const { return {cuda_surf}; }
  ~CudaSurface3D() { cudaDestroySurfaceObject(cuda_surf); }
};

template <class T> struct CudaTextureAccessor3D {
  cudaTextureObject_t m_cuTex;
  __device__ __forceinline__ T sample(float x, float y, float z) const {
    return tex3D<T>(m_cuTex, x, y, z);
  }
};

template <class T> struct CudaTexture3D : CudaSurface3D<T> {
  struct Parameters {
    cudaTextureAddressMode addressMode{cudaAddressModeClamp};
    cudaTextureFilterMode filterMode{cudaFilterModeLinear};
    cudaTextureReadMode readMode{cudaReadModeElementType};
    bool normalizedCoords{false};
  };

  cudaTextureObject_t cuda_tex{};
  explicit CudaTexture3D(uint3 const &_dim, Parameters const &_args = {})
      : CudaSurface3D<T>(_dim) {
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = CudaSurface3D<T>::getArray();

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = _args.addressMode;
    texDesc.addressMode[1] = _args.addressMode;
    texDesc.addressMode[2] = _args.addressMode;
    texDesc.filterMode = _args.filterMode;
    texDesc.readMode = _args.readMode;
    texDesc.normalizedCoords = _args.normalizedCoords;

    cudaCreateTextureObject(&cuda_tex, &resDesc, &texDesc, nullptr);
  }
  cudaTextureObject_t Texture() const { return cuda_tex; }
  CudaTextureAccessor3D<T> TexAccessor() const { return {cuda_tex}; }
  ~CudaTexture3D() { cudaDestroyTextureObject(cuda_tex); }
};

template <typename T> class CudaSurface2D : public GpuArray2D<T> {
public:
  cudaSurfaceObject_t cuda_surf;
  explicit CudaSurface2D(const uint2 &dim_) : GpuArray2D<T>(dim_) {
    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res = GpuArray2D<T>::Array();
    cudaCreateSurfaceObject(&cuda_surf, &res_desc);
  }
  cudaSurfaceObject_t Surface() const { return cuda_surf; }
  CudaSurfaceAccessor<T> SurfAccessor() const { return {cuda_surf}; }
  ~CudaSurface2D() { cudaDestroySurfaceObject(cuda_surf); }
};
// 2D version of CudaTexture and CudaTextureAccessor
template <typename T> struct CudaTextureAccessor2D {
  cudaTextureObject_t m_cuTex;
  __device__ __forceinline__ T sample(float x, float y) const {
    return tex2D<T>(m_cuTex, x, y);
  }
};
template <typename T> struct CudaTexture2D : CudaSurface2D<T> {
  struct Parameters {
    cudaTextureAddressMode addressMode{cudaAddressModeClamp};
    cudaTextureFilterMode filterMode{cudaFilterModeLinear};
    cudaTextureReadMode readMode{cudaReadModeElementType};
    bool normalizedCoords{false};
  };
  cudaTextureObject_t cuda_tex{};
  explicit CudaTexture2D(uint2 const &_dim, Parameters const &_args = {})
      : CudaSurface2D<T>(_dim) {
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = CudaSurface2D<T>::getArray();

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = _args.addressMode;
    texDesc.addressMode[1] = _args.addressMode;
    texDesc.filterMode = _args.filterMode;
    texDesc.readMode = _args.readMode;
    texDesc.normalizedCoords = _args.normalizedCoords;

    cudaCreateTextureObject(&cuda_tex, &resDesc, &texDesc, nullptr);
  }
  cudaTextureObject_t Texture() const { return cuda_tex; }
  CudaTextureAccessor2D<T> TexAccessor() const { return {cuda_tex}; }
  ~CudaTexture2D() { cudaDestroyTextureObject(cuda_tex); }
};
} // namespace fluid
#endif // SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_GPU_ARRAYS_H_
