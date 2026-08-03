#pragma once

#include <Runtime/types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <glm/geometric.hpp>

namespace ksk::runtime::detail {

[[nodiscard]] inline glm::dvec3 safeNormalize(glm::dvec3 value,
                                              glm::dvec3 fallback) {
  const Real length = glm::length(value);
  if (std::isfinite(length) && length > 1.0e-12) {
    return value / length;
  }
  const Real fallback_length = glm::length(fallback);
  if (std::isfinite(fallback_length) && fallback_length > 1.0e-12) {
    return fallback / fallback_length;
  }
  return glm::dvec3(0.0, 1.0, 0.0);
}

[[nodiscard]] inline Real pointEdgeProjectionParameter(const glm::dvec3& p,
                                                       const glm::dvec3& a,
                                                       const glm::dvec3& b) {
  const glm::dvec3 e = b - a;
  const Real e_len2 = glm::dot(e, e);
  if (e_len2 == 0.0) {
    return 0.0;
  }
  return glm::dot(p - a, e) / e_len2;
}

[[nodiscard]] inline Real clampedPointEdgeProjectionParameter(
    const glm::dvec3& p, const glm::dvec3& a, const glm::dvec3& b) {
  return std::clamp(pointEdgeProjectionParameter(p, a, b), 0.0, 1.0);
}

[[nodiscard]] inline std::array<Real, 3> triangleBarycentric(
    const glm::dvec3& p, const glm::dvec3& a, const glm::dvec3& b,
    const glm::dvec3& c) {
  const glm::dvec3 v0 = b - a;
  const glm::dvec3 v1 = c - a;
  const glm::dvec3 v2 = p - a;
  const Real d00 = glm::dot(v0, v0);
  const Real d01 = glm::dot(v0, v1);
  const Real d11 = glm::dot(v1, v1);
  const Real d20 = glm::dot(v2, v0);
  const Real d21 = glm::dot(v2, v1);
  const Real denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) <= 1.0e-18) {
    return {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  }
  Real v = (d11 * d20 - d01 * d21) / denom;
  Real w = (d00 * d21 - d01 * d20) / denom;
  Real u = 1.0 - v - w;
  u = std::clamp(u, 0.0, 1.0);
  v = std::clamp(v, 0.0, 1.0);
  w = std::clamp(w, 0.0, 1.0);
  const Real sum = u + v + w;
  if (sum > 1.0e-18) {
    u /= sum;
    v /= sum;
    w /= sum;
  }
  return {u, v, w};
}

[[nodiscard]] inline std::array<Real, 2> closestSegmentParameters(
    const glm::dvec3& p1, const glm::dvec3& q1, const glm::dvec3& p2,
    const glm::dvec3& q2) {
  const glm::dvec3 d1 = q1 - p1;
  const glm::dvec3 d2 = q2 - p2;
  const glm::dvec3 r = p1 - p2;
  const Real a = glm::dot(d1, d1);
  const Real e = glm::dot(d2, d2);
  const Real f = glm::dot(d2, r);
  Real s = 0.0;
  Real t = 0.0;

  if (a <= 1.0e-18 && e <= 1.0e-18) {
    return {0.0, 0.0};
  }
  if (a <= 1.0e-18) {
    t = std::clamp(f / e, 0.0, 1.0);
  } else {
    const Real c = glm::dot(d1, r);
    if (e <= 1.0e-18) {
      s = std::clamp(-c / a, 0.0, 1.0);
    } else {
      const Real b = glm::dot(d1, d2);
      const Real denom = a * e - b * b;
      if (std::abs(denom) > 1.0e-18) {
        s = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
      }
      t = (b * s + f) / e;
      if (t < 0.0) {
        t = 0.0;
        s = std::clamp(-c / a, 0.0, 1.0);
      } else if (t > 1.0) {
        t = 1.0;
        s = std::clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }
  return {s, t};
}

}  // namespace ksk::runtime::detail
