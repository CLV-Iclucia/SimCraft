#ifndef FLUIDSIM_VEC_OP_CUH
#define FLUIDSIM_VEC_OP_CUH

#include <cuda_runtime.h>

namespace fluid {
__device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x_t + b.x_t, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x_t - b.x_t, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 operator*(const float3 &a, float b) {
  return make_float3(a.x_t * b, a.y * b, a.z * b);
}
__device__ __forceinline__ float3 operator/(const float3 &a, float b) {
  return make_float3(a.x_t / b, a.y / b, a.z / b);
}
__device__ __forceinline__ float3 operator*(float a, const float3 &b) {
  return make_float3(a * b.x_t, a * b.y, a * b.z);
}
__device__ __forceinline__ float3 operator/(float a, const float3 &b) {
  return make_float3(a / b.x_t, a / b.y, a / b.z);
}
__device__ __forceinline__ float3 &operator+=(float3 &a, const float3 &b) {
  a.x_t += b.x_t;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__device__ __forceinline__ float4 operator+(const float4 &a, const float4 &b) {
  return make_float4(a.x_t + b.x_t, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 operator-(const float4 &a, const float4 &b) {
  return make_float4(a.x_t - b.x_t, a.y - b.y, a.z - b.z, a.w - b.w);
}
__device__ __forceinline__ float4 operator*(const float4 &a, float b) {
  return make_float4(a.x_t * b, a.y * b, a.z * b, a.w * b);
}
__device__ __forceinline__ float4 operator/(const float4 &a, float b) {
  return make_float4(a.x_t / b, a.y / b, a.z / b, a.w / b);
}
__device__ __forceinline__ float4 operator*(float a, const float4 &b) {
  return make_float4(a * b.x_t, a * b.y, a * b.z, a * b.w);
}
__device__ __forceinline__ float4 operator/(float a, const float4 &b) {
  return make_float4(a / b.x_t, a / b.y, a / b.z, a / b.w);
}
__device__ __forceinline__ float4 &operator+=(float4 &a, const float4 &b) {
  a.x_t += b.x_t;
  a.y += b.y;
  a.z += b.z;
  return a;
}
__device__ __forceinline__ float3 &operator-=(float3 &a, const float3 &b) {
  a.x_t -= b.x_t;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

__device__ __forceinline__ float norm(const float4 &vec) {
  return sqrtf(vec.x_t * vec.x_t + vec.y * vec.y + vec.z * vec.z);
}
__device__ __forceinline__ float distance(const float3 &a, const float3 &b) {
  return sqrtf((a.x_t - b.x_t) * (a.x_t - b.x_t) + (a.y - b.y) * (a.y - b.y) +
              (a.z - b.z) * (a.z - b.z));
}
__device__ __forceinline__ double distance(const double3 &a, const double3 &b) {
  return sqrt((a.x_t - b.x_t) * (a.x_t - b.x_t) + (a.y - b.y) * (a.y - b.y) +
              (a.z - b.z) * (a.z - b.z));
}
__device__ __forceinline__ float norm(const float3 &vec) {
  return sqrtf(vec.x_t * vec.x_t + vec.y * vec.y + vec.z * vec.z);
}
__device__ __forceinline__ float4 normalize(const float4 &vec) {
  float n = norm(vec);
  return make_float4(vec.x_t / n, vec.y / n, vec.z / n, vec.w);
}
__device__ __forceinline__ float3 normalize(const float3 &vec) {
  float n = sqrtf(vec.x_t * vec.x_t + vec.y * vec.y + vec.z * vec.z);
  return make_float3(vec.x_t / n, vec.y / n, vec.z / n);
}
__device__ __forceinline__ float3 cross(const float3 &a, const float3 &b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x_t - a.x_t * b.z,
                     a.x_t * b.y - a.y * b.x_t);
}
// implement double versions
__device__ __forceinline__ double3 operator+(const double3 &a, const double3 &b) {
  return make_double3(a.x_t + b.x_t, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ double3 operator-(const double3 &a, const double3 &b) {
  return make_double3(a.x_t - b.x_t, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ double3 operator*(const double3 &a, double b) {
  return make_double3(a.x_t * b, a.y * b, a.z * b);
}
__device__ __forceinline__ double3 operator/(const double3 &a, double b) {
  return make_double3(a.x_t / b, a.y / b, a.z / b);
}
__device__ __forceinline__ double3 operator*(double a, const double3 &b) {
  return make_double3(a * b.x_t, a * b.y, a * b.z);
}
__device__ __forceinline__ double3 operator/(double a, const double3 &b) {
  return make_double3(a / b.x_t, a / b.y, a / b.z);
}
__device__ __forceinline__ double3 &operator+=(double3 &a, const double3 &b) {
  a.x_t += b.x_t;
  a.y += b.y;
  a.z += b.z;
  return a;
}
__device__ __forceinline__ double3 normalize(const double3 &vec) {
  double n = sqrt(vec.x_t * vec.x_t + vec.y * vec.y + vec.z * vec.z);
  if (n == 0) return make_double3(0, 0, 0);
  return make_double3(vec.x_t / n, vec.y / n, vec.z / n);
}
__device__ __forceinline__ float dot(const float3 &a, const float3 &b) {
  return a.x_t * b.x_t + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float dot(const float4 &a, const float4 &b) {
  return a.x_t * b.x_t + a.y * b.y + a.z * b.z;
}}
#endif //FLUIDSIM_VEC_OP_CUH