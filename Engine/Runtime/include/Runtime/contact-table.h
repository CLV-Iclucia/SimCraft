#pragma once

#include <Runtime/types.h>

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

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

[[nodiscard]] constexpr int contactPointCount(ContactCase type) noexcept {
  switch (type) {
    case ContactCase::PP:
      return 2;
    case ContactCase::PE:
      return 3;
    case ContactCase::PT:
    case ContactCase::EE:
      return 4;
  }
  return 0;
}

struct ContactStencil {
  ContactCase type = ContactCase::PP;
  std::array<int, 4> geometryIds{-1, -1, -1, -1};
  Real dHat = 0.0;
  Real stiffness = 0.0;
  // Total surface offset reserved from geometric distance before evaluating
  // the barrier, computed from the involved primitive radii/thicknesses.
  Real thickness = 0.0;
  ContactMollifierDistance mollifierDistance = ContactMollifierDistance::None;
};

using ContactStencils = std::vector<ContactStencil>;

struct FrictionContactStencil {
  ContactCase type = ContactCase::PP;
  std::array<int, 4> geometryIds{-1, -1, -1, -1};
  Real coefficient = 0.0;
  Real epsilon = 1.0e-6;
  Real normalForce = 0.0;
  glm::dvec3 normal{0.0, 1.0, 0.0};
  std::array<Real, 4> weights{0.0, 0.0, 0.0, 0.0};
  std::array<glm::dvec3, 4> referenceX{glm::dvec3(0.0), glm::dvec3(0.0),
                                       glm::dvec3(0.0), glm::dvec3(0.0)};
};

using FrictionContactStencils = std::vector<FrictionContactStencil>;

}  // namespace ksk::runtime
