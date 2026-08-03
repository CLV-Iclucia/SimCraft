#pragma once

#include <Runtime/buffers.h>
#include <Runtime/contact-table.h>
#include <Runtime/global-geometry-manager.h>

#include <glm/glm.hpp>

#include <array>
#include <span>
#include <vector>

namespace ksk::runtime {

struct ContactPotentialGradient {
  std::vector<PointIdx> points;
  GeometryBuffer gradient;
};

struct ContactGeometryHessianBlock {
  PointIdx row = -1;
  PointIdx col = -1;
  glm::dmat3 value{0.0};
};

struct ContactGeometryHessian {
  std::vector<ContactGeometryHessianBlock> blocks;
};

struct ContactLocalHessian {
  std::array<std::array<glm::dmat3, 4>, 4> blocks;

  ContactLocalHessian()
  {
    for (auto& row : blocks) {
      for (glm::dmat3& block : row) {
        block = glm::dmat3(0.0);
      }
    }
  }
};

[[nodiscard]] double computeContactEnergy(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts);

[[nodiscard]] ContactPotentialGradient computeContactGradient(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts);

[[nodiscard]] ContactPotentialGradient computeContactGradientWrtGeometry(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts);

[[nodiscard]] ContactGeometryHessian computeContactHessianWrtGeometry(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts);

[[nodiscard]] ContactPotentialGradient computeContactHessianProduct(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts,
    ConstGeometryView geometryDirection);

[[nodiscard]] ContactPotentialGradient computeContactHessianProductWrtGeometry(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts,
    ConstGeometryView geometryDirection);

}  // namespace ksk::runtime
