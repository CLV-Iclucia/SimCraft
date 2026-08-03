#pragma once

#include <Runtime/contact-barrier-energy.h>
#include <Runtime/solver-config.h>

#include <array>
#include <span>

namespace ksk::runtime {

[[nodiscard]] double computeContactFrictionEnergyGradient(
    const FrictionContactStencil& contact,
    std::span<const glm::dvec3> positions,
    std::array<glm::dvec3, 4>* gradient = nullptr);

[[nodiscard]] ContactLocalHessian computeContactFrictionHessian(
    const FrictionContactStencil& contact,
    std::span<const glm::dvec3> positions);

[[nodiscard]] std::array<glm::dvec3, 4> computeContactFrictionHessianProduct(
    const FrictionContactStencil& contact,
    std::span<const glm::dvec3> positions,
    std::span<const glm::dvec3> direction);

[[nodiscard]] FrictionContactStencils buildFrictionContactStencils(
    const GlobalGeometryManager& geometry,
    const ContactStencils& contacts,
    const FrictionMaterialTable& frictionTable,
    Real defaultCoefficient,
    Real epsilon);

[[nodiscard]] double computeContactFrictionEnergy(
    const GlobalGeometryManager& geometry,
    const FrictionContactStencils& contacts);

[[nodiscard]] ContactPotentialGradient computeContactFrictionGradientWrtGeometry(
    const GlobalGeometryManager& geometry,
    const FrictionContactStencils& contacts);

[[nodiscard]] ContactGeometryHessian computeContactFrictionHessianWrtGeometry(
    const GlobalGeometryManager& geometry,
    const FrictionContactStencils& contacts);

[[nodiscard]] ContactPotentialGradient
computeContactFrictionHessianProductWrtGeometry(
    const GlobalGeometryManager& geometry,
    const FrictionContactStencils& contacts,
    ConstGeometryView geometryDirection);

}  // namespace ksk::runtime
