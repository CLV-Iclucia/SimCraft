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

[[nodiscard]] double computeContactFrictionEnergyGradient(
    const ContactStencil& contact,
    std::span<const glm::dvec3> positions,
    std::array<glm::dvec3, 4>* gradient = nullptr);

[[nodiscard]] ContactLocalHessian computeContactFrictionHessian(
    const ContactStencil& contact,
    std::span<const glm::dvec3> positions);

[[nodiscard]] std::array<glm::dvec3, 4> computeContactFrictionHessianProduct(
    const ContactStencil& contact,
    std::span<const glm::dvec3> positions,
    std::span<const glm::dvec3> direction);

[[nodiscard]] double computeContactFrictionEnergy(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts);

[[nodiscard]] ContactPotentialGradient computeContactFrictionGradientWrtGeometry(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts);

[[nodiscard]] ContactGeometryHessian computeContactFrictionHessianWrtGeometry(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts);

[[nodiscard]] ContactPotentialGradient
computeContactFrictionHessianProductWrtGeometry(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts,
    ConstGeometryView geometryDirection);

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
