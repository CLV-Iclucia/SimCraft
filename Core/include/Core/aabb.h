#ifndef CORE_INCLUDE_CORE_AABB_H
#define CORE_INCLUDE_CORE_AABB_H
#include <Core/properties.h>
#include <Core/utils.h>

namespace core {
template <typename T, int Dim>
struct AABB {
  Vector<T, Dim> lo{};
  Vector<T, Dim> hi{};
  AABB() = default;
  AABB(const Vector<T, Dim>& lo, const Vector<T, Dim>& hi) : lo(lo), hi(hi) {
  }
  AABB merge(const AABB& other) const {
    return {cwiseMin<T, Dim>(lo, other.lo), cwiseMax<T, Dim>(hi, other.hi)};
  }
  AABB dilate(T factor) const {
    return {lo - factor, hi + factor};
  }
  bool overlap(const AABB& other) const {
    return (lo.array() <= other.hi.array()).all() &&
           (hi.array() >= other.lo.array()).all();
  }
  bool inside(const Vector<T, Dim>& point) const {
    return (point.array() >= lo.array()).all() &&
           (point.array() <= hi.array()).all();
  }
  bool contain(const AABB& other) const {
    return (lo.array() <= other.lo.array()).all() &&
           (hi.array() >= other.hi.array()).all();
  }
  bool contain(const Vector<T, Dim>& point) const {
    return (point.array() >= lo.array()).all() &&
           (point.array() <= hi.array()).all();
  }
  Vector<T, Dim> centre() const {
    return (lo + hi) * static_cast<T>(0.5);
  }
};
}
#endif //COLLISION_INCLUDE_COLLISION_CPU_LBVH_H