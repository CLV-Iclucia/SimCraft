#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include <Runtime/types.h>

namespace ksk::runtime {

enum class ContactCase : std::uint16_t {
  PP,
  PE,
  PT,
  EE,
};

enum class ContactMollifierDistance : std::uint16_t {
  None,
  PP,
  PE,
  EE,
};

struct ContactStencil {
  ContactCase type = ContactCase::PP;
  std::array<int, 4> geometryIds{-1, -1, -1, -1};
  Real dHat = 0.0;
  Real stiffness = 0.0;
  // Total surface offset reserved from geometric distance before evaluating
  // the barrier, computed from the involved primitive radii/thicknesses.
  Real thickness = 0.0;
  ContactMollifierDistance mollifierDistance =
      ContactMollifierDistance::None;
  Real frictionCoefficient = 0.0;
  Real frictionEpsilon = 1.0e-6;
  Real frictionNormalForce = 0.0;
  glm::dvec3 frictionNormal{0.0, 1.0, 0.0};
  std::array<Real, 4> frictionWeights{0.0, 0.0, 0.0, 0.0};
  std::array<glm::dvec3, 4> frictionReferenceX{
      glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0)};
};

using ContactStencils = std::vector<ContactStencil>;

}  // namespace ksk::runtime
