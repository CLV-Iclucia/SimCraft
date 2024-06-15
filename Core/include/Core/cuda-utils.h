//
// Created by creeper on 24-3-22.
//

#ifndef CORE_CUDA_UTILS_H
#define CORE_CUDA_UTILS_H

#include <cuda_runtime.h>
#include <Core/properties.h>
#include <memory>
#include <iostream>

static constexpr int kThreadBlockSize = 128;
static constexpr int kThreadBlockSize2D = 16;
static constexpr int kThreadBlockSize3D = 8;

namespace core {
#define CUDA_CALLABLE __host__ __device__
#define CUDA_FORCEINLINE __forceinline__
#define CUDA_INLINE __inline__
#define CUDA_SHARED __shared__

#ifndef NDEBUG
#define cudaSafeCheck(kernel) do { \
  kernel;                          \
  cudaDeviceSynchronize();         \
  cudaError_t error = cudaGetLastError(); \
  if (error != cudaSuccess) { \
    printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
    assert(false); \
  } \
} while (0)
#else
#define cudaSafeCheck(kernel) kernel
#endif
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#define CUDA_GLOBAL __global__
#define CUDA_CONSTANT __constant__

#define ktid(axis) (blockIdx.axis * blockDim.axis + threadIdx.axis)
#define get_tid(tid) int tid = ktid(x)
#define get_tid_2d(tid) int tid = ktid(x); int tid_y = ktid(y)
#define get_tid_3d(tid) int tid = ktid(x); int tid_y = ktid(y); int tid_z = ktid(z)
#define get_and_restrict_tid(tid, max) int tid = ktid(x); do { if (tid >= max) return; } while (0)

#define get_and_restrict_tid_2d(tid_x, tid_y, max_x, max_y) \
  int tid_x = ktid(x); int tid_y = ktid(y); \
  do { if (tid_x >= max_x || tid_y >= max_y) return; } while (0)

#define get_and_restrict_tid_3d(tid_x, tid_y, tid_z, max_x, max_y, max_z) \
  int tid_x = ktid(x); int tid_y = ktid(y); int tid_z = ktid(z); \
  do { if (tid_x >= max_x || tid_y >= max_y || tid_z >= max_z) return; }\
  while (0)

#define cuExit() asm("exit;")

template<typename T>
struct best_return_type_for_const {
  using const_ref = std::add_lvalue_reference_t<std::add_const_t<T>>;
  using type = std::conditional_t<std::is_trivially_copyable_v<T>, T, const_ref>;
};

template<typename T>
using best_return_type_for_const_t = typename best_return_type_for_const<
    T>::type;

template<typename T>
requires std::is_pointer_v<T>
bool checkDevicePtr(T arg) {
  cudaPointerAttributes attr{};
  cudaSafeCall(cudaPointerGetAttributes(&attr, arg));
  return attr.devicePointer != nullptr;
}

#define LAUNCH_THREADS(x) ((x) + kThreadBlockSize - 1) / kThreadBlockSize, kThreadBlockSize
#define LAUNCH_THREADS_3D(x, y, z) \
  dim3(((x) + kThreadBlockSize3D - 1) / kThreadBlockSize3D, \
  ((y) + kThreadBlockSize3D - 1) / kThreadBlockSize3D, \
  ((z) + kThreadBlockSize3D - 1) / kThreadBlockSize3D), \
  dim3(kThreadBlockSize3D, kThreadBlockSize3D, kThreadBlockSize3D)

template<typename T>
struct DeviceAutoPtr {
  T *ptr{};
  DeviceAutoPtr() = default;

  explicit DeviceAutoPtr(T *ptr_) {
    checkDevicePtr(ptr_);
    ptr = ptr_;
  }

  template<typename... Args>
  explicit DeviceAutoPtr(Args &&... args) {
    cudaMalloc(&ptr, sizeof(T));
    new(ptr) T(std::forward<Args>(args)...);
  }

  // disallow copy
  DeviceAutoPtr(const DeviceAutoPtr &) = delete;
  DeviceAutoPtr &operator=(const DeviceAutoPtr &) = delete;
  // move
  DeviceAutoPtr(DeviceAutoPtr &&other) noexcept {
    ptr = other.ptr;
    other.ptr = nullptr;
  }

  DeviceAutoPtr &operator=(DeviceAutoPtr &&other) noexcept {
    if (ptr) {
      ptr->~T();
      cudaFree(ptr);
    }
    ptr = other.ptr;
    other.ptr = nullptr;
    return *this;
  }

  T *get() const { return ptr; }

  ~DeviceAutoPtr() {
    if (ptr) {
      ptr->~T();
      cudaFree(ptr);
    }
  }
};
template<typename Derived>
struct Accessor;
template<typename Derived>
struct ConstAccessor;

template<typename Derived>
struct DeviceMemoryAccessible : NonCopyable {
  const Derived &derived() const {
    return static_cast<const Derived &>(*this);
  }

  Derived &derived() {
    return static_cast<Derived &>(*this);
  }

  Accessor<Derived> accessorInterface() {
    return derived().accessor();
  }

  ConstAccessor<Derived> constAccessorInterface() const {
    return derived().constAccessor();
  }
};

template<typename Derived>
struct DeviceMemoryReadable : DeviceMemoryAccessible<Derived> {
  const Derived &derived() const {
    return static_cast<const Derived &>(*this);
  }

  Derived &derived() {
    return static_cast<Derived &>(*this);
  }

  ConstAccessor<Derived> constAccessorInterface() const {
    return derived().constAccessor();
  }
};

template<typename Derived>
Accessor<Derived> access(std::unique_ptr<Derived> &ptr) {
  return static_cast<DeviceMemoryAccessible<Derived>>(ptr)->accessorInterface();
}

template<typename Derived>
ConstAccessor<Derived> constAccess(const std::unique_ptr<Derived> &ptr) {
  return static_cast<DeviceMemoryAccessible<Derived>>(ptr)->
      constAccessorInterface();
}
}

#endif //CORE_CUDA_UTILS_H