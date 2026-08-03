#pragma once

#include <Runtime/contact-barrier-energy.h>
#include <Runtime/contact-table.h>

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>

namespace ksk::runtime::detail {

inline int appendPoint(std::vector<PointIdx>& points,
                       std::vector<glm::dvec3>& values,
                       std::unordered_map<PointIdx, int>& offsets,
                       PointIdx point) {
  const auto found = offsets.find(point);
  if (found != offsets.end()) {
    return found->second;
  }

  const int offset = static_cast<int>(points.size());
  offsets.emplace(point, offset);
  points.push_back(point);
  values.push_back(glm::dvec3(0.0));
  return offset;
}

[[nodiscard]] inline bool isNonZero(const glm::dvec3& value) {
  return value.x != 0.0 || value.y != 0.0 || value.z != 0.0;
}

[[nodiscard]] inline bool isNonZero(const glm::dmat3& value) {
  for (int col = 0; col < 3; col++) {
    for (int row = 0; row < 3; row++) {
      if (value[col][row] != 0.0) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] inline std::uint64_t pointPairKey(PointIdx row, PointIdx col) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row)) << 32) |
         static_cast<std::uint32_t>(col);
}

template <typename Stencil>
void scatterContactValues(const Stencil& contact,
                          const std::array<glm::dvec3, 4>& values,
                          std::vector<PointIdx>& points,
                          std::vector<glm::dvec3>& out,
                          std::unordered_map<PointIdx, int>& offsets) {
  for (int i = 0; i < contactPointCount(contact.type); i++) {
    if (!isNonZero(values[static_cast<size_t>(i)])) {
      continue;
    }

    const int offset = appendPoint(points, out, offsets,
                                   contact.geometryIds[static_cast<size_t>(i)]);
    out[static_cast<size_t>(offset)] += values[static_cast<size_t>(i)];
  }
}

template <typename Stencil, typename Hessian>
void scatterContactHessian(const Stencil& contact, const Hessian& hessian,
                           std::vector<ContactGeometryHessianBlock>& blocks,
                           std::unordered_map<std::uint64_t, int>& offsets) {
  const int count = contactPointCount(contact.type);
  for (int row = 0; row < count; row++) {
    const PointIdx row_point = contact.geometryIds[static_cast<size_t>(row)];
    for (int col = 0; col < count; col++) {
      const glm::dmat3& value =
          hessian.blocks[static_cast<size_t>(row)][static_cast<size_t>(col)];
      if (!isNonZero(value)) {
        continue;
      }

      const PointIdx col_point = contact.geometryIds[static_cast<size_t>(col)];
      const std::uint64_t key = pointPairKey(row_point, col_point);
      const auto found = offsets.find(key);
      if (found != offsets.end()) {
        blocks[static_cast<size_t>(found->second)].value += value;
        continue;
      }

      const int offset = static_cast<int>(blocks.size());
      offsets.emplace(key, offset);
      blocks.push_back(ContactGeometryHessianBlock{
          .row = row_point,
          .col = col_point,
          .value = value,
      });
    }
  }
}

}  // namespace ksk::runtime::detail
