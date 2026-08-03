#include <Contact/contact-barrier.h>
#include <Contact/gipc/barrier.h>
#include <Contact/shared/ipc-distance.h>
#include <Core/profiler.h>
#include <Runtime/contact-friction.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/geometric.hpp>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "contact-accumulation-utils.h"
#include "contact-geometry-utils.h"

namespace ksk::runtime {
namespace {

Real effectiveDistanceSqr(const ContactStencil& contact,
                          const std::array<glm::dvec3, 4>& positions) {
  using namespace ksk::engine::contact::shared;
  switch (contact.type) {
    case ContactCase::PP:
      return shDistanceSqrPointPoint(positions[0], positions[1]);
    case ContactCase::PE:
      return shDistanceSqrPointLine(positions[0], positions[1], positions[2]);
    case ContactCase::PT:
      return shDistanceSqrPointPlane(positions[0], positions[1], positions[2],
                                     positions[3]);
    case ContactCase::EE:
      return shDistanceSqrLineLine(positions[0], positions[1], positions[2],
                                   positions[3]);
  }
  return 0.0;
}

Real estimateBarrierNormalForce(const ContactStencil& contact,
                                const std::array<glm::dvec3, 4>& positions) {
  if (contact.stiffness <= 0.0 || contact.dHat <= 0.0) {
    return 0.0;
  }

  const Real distance =
      std::sqrt(std::max(effectiveDistanceSqr(contact, positions), 0.0));
  const Real gap =
      std::max(distance - contact.thickness, contact.dHat * 1.0e-12);
  const Real i5 = (gap * gap) / (contact.dHat * contact.dHat);
  if (i5 <= 0.0 || i5 >= 1.0) {
    return 0.0;
  }

  const engine::contact::gipc::Barrier barrier(contact.dHat);
  const Real dB_dgap =
      barrier.dBdI5(i5) * (2.0 * gap) / (contact.dHat * contact.dHat);
  return std::max<Real>(0.0, -contact.stiffness * dB_dgap);
}

Real contactFrictionCoefficient(const GlobalGeometryManager& geometry,
                                const ContactStencil& contact,
                                const FrictionMaterialTable& friction_table,
                                Real default_coefficient) {
  std::vector<PhysicalMaterialId> materials;
  const int count = contactPointCount(contact.type);
  for (int i = 0; i < count; ++i) {
    const int id = contact.geometryIds[static_cast<size_t>(i)];
    if (id < 0 || id >= geometry.pointCount()) {
      continue;
    }
    const GeometryBodyIdx body = geometry.pointOwner(PointIdx{id}).body;
    if (body < 0 ||
        body >= static_cast<int>(geometry.bodyCollisionFilters.size())) {
      continue;
    }
    const PhysicalMaterialId material =
        geometry.bodyCollisionFilters[static_cast<size_t>(body)]
            .physicalMaterial;
    if (std::find(materials.begin(), materials.end(), material) ==
        materials.end()) {
      materials.push_back(material);
    }
  }

  if (materials.empty()) {
    return default_coefficient;
  }
  if (materials.size() == 1) {
    return friction_table.lookup(materials.front(), materials.front())
        .value_or(default_coefficient);
  }

  std::optional<Real> coefficient;
  for (size_t i = 0; i < materials.size(); ++i) {
    for (size_t j = i + 1; j < materials.size(); ++j) {
      const std::optional<Real> pair_coefficient =
          friction_table.lookup(materials[i], materials[j]);
      if (!pair_coefficient.has_value()) {
        continue;
      }
      coefficient = coefficient.has_value() ? std::min(coefficient.value(),
                                                       pair_coefficient.value())
                                            : pair_coefficient.value();
    }
  }
  return coefficient.value_or(default_coefficient);
}

int frictionPointCount(const FrictionContactStencil& contact) {
  return contactPointCount(contact.type);
}

bool frictionEnabled(const FrictionContactStencil& contact) {
  return contact.coefficient > 0.0 && contact.normalForce > 0.0 &&
         contact.epsilon > 0.0;
}

glm::dmat3 tangentProjector(const glm::dvec3& normal) {
  const double norm = glm::length(normal);
  const glm::dvec3 n =
      norm > 1.0e-12 ? normal / norm : glm::dvec3(0.0, 1.0, 0.0);
  glm::dmat3 projector(1.0);
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      projector[col][row] -= n[row] * n[col];
    }
  }
  return projector;
}

glm::dvec3 frictionRelativeDisplacement(
    const FrictionContactStencil& contact,
    const std::array<glm::dvec3, 4>& positions) {
  glm::dvec3 relative(0.0);
  const int count = frictionPointCount(contact);

  for (int i = 0; i < count; ++i)
    relative += contact.weights[i] * (positions[i] - contact.referenceX[i]);

  return tangentProjector(contact.normal) * relative;
}

glm::dmat3 frictionTangentHessian(const FrictionContactStencil& contact,
                                  const glm::dvec3& displacement) {
  const glm::dmat3 projector = tangentProjector(contact.normal);
  const double eps = std::max(contact.epsilon, 1.0e-12);
  const double s = std::sqrt(glm::dot(displacement, displacement) + eps * eps);
  glm::dmat3 hessian = projector * (1.0 / s);
  const double inv_s3 = 1.0 / (s * s * s);
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      hessian[col][row] -= displacement[row] * displacement[col] * inv_s3;
    }
  }
  return hessian;
}

double appendFrictionEnergyGradient(const FrictionContactStencil& contact,
                                    const std::array<glm::dvec3, 4>& positions,
                                    std::array<glm::dvec3, 4>* gradient) {
  if (!frictionEnabled(contact)) {
    return 0.0;
  }
  const glm::dvec3 displacement =
      frictionRelativeDisplacement(contact, positions);
  const double eps = std::max(contact.epsilon, 1.0e-12);
  const double s = std::sqrt(glm::dot(displacement, displacement) + eps * eps);
  const double scale = contact.coefficient * contact.normalForce;

  if (gradient != nullptr) {
    const glm::dvec3 tangent_gradient = scale * displacement / s;
    const int count = frictionPointCount(contact);
    for (int i = 0; i < count; ++i) {
      (*gradient)[i] += contact.weights[i] * tangent_gradient;
    }
  }

  return scale * (s - eps);
}

void appendFrictionHessian(const FrictionContactStencil& contact,
                           const std::array<glm::dvec3, 4>& positions,
                           engine::contact::LocalBarrierHessian& hessian) {
  if (!frictionEnabled(contact)) {
    return;
  }
  const glm::dvec3 displacement =
      frictionRelativeDisplacement(contact, positions);
  const glm::dmat3 tangent_hessian =
      frictionTangentHessian(contact, displacement) *
      (contact.coefficient * contact.normalForce);
  const int count = frictionPointCount(contact);
  for (int row = 0; row < count; ++row) {
    for (int col = 0; col < count; ++col) {
      hessian.blocks[row][col] +=
          contact.weights[row] * contact.weights[col] * tangent_hessian;
    }
  }
}

std::array<glm::dvec3, 4> frictionHessianProduct(
    const FrictionContactStencil& contact,
    const std::array<glm::dvec3, 4>& positions,
    const std::array<glm::dvec3, 4>& direction) {
  std::array result{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                    glm::dvec3(0.0)};
  if (!frictionEnabled(contact)) {
    return result;
  }
  glm::dvec3 relative_direction(0.0);
  const int count = frictionPointCount(contact);
  for (int i = 0; i < count; ++i) {
    relative_direction += contact.weights[i] * direction[i];
  }

  const glm::dvec3 displacement =
      frictionRelativeDisplacement(contact, positions);
  const glm::dmat3 tangent_hessian =
      frictionTangentHessian(contact, displacement) *
      (contact.coefficient * contact.normalForce);
  const glm::dvec3 tangent_product = tangent_hessian * relative_direction;
  for (int i = 0; i < count; ++i) {
    result[static_cast<size_t>(i)] +=
        contact.weights[static_cast<size_t>(i)] * tangent_product;
  }
  return result;
}

std::array<glm::dvec3, 4> gatherFrictionPositions(
    const GlobalGeometryManager& geometry,
    const FrictionContactStencil& contact) {
  std::array positions{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                       glm::dvec3(0.0)};
  const int count = contactPointCount(contact.type);
  for (int i = 0; i < count; ++i) {
    positions[i] = geometry.worldPosition(PointIdx{contact.geometryIds[i]});
  }
  return positions;
}

std::array<glm::dvec3, 4> gatherFrictionDirection(
    const FrictionContactStencil& contact,
    const ConstGeometryView& geometryDirection) {
  std::array direction{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                       glm::dvec3(0.0)};
  for (int i = 0; i < contactPointCount(contact.type); i++) {
    direction[static_cast<size_t>(i)] =
        geometryDirection.cpu()[contact.geometryIds[static_cast<size_t>(i)]];
  }
  return direction;
}

}  // namespace

double computeContactFrictionEnergy(const GlobalGeometryManager& geometry,
                                    const FrictionContactStencils& contacts) {
  KSK_PROFILE_SCOPE("ContactFriction/ComputeEnergy");
  KSK_PROFILE_VALUE("contact.friction_energy.stencils",
                    static_cast<double>(contacts.size()));
  double energy = 0.0;
  for (const FrictionContactStencil& contact : contacts) {
    energy += appendFrictionEnergyGradient(
        contact, gatherFrictionPositions(geometry, contact), nullptr);
  }
  return energy;
}

FrictionContactStencils buildFrictionContactStencils(
    const GlobalGeometryManager& geometry, const ContactStencils& contacts,
    const FrictionMaterialTable& frictionTable, Real defaultCoefficient,
    Real epsilon) {
  KSK_PROFILE_SCOPE("ContactFriction/BuildContactStencils");
  FrictionContactStencils friction_contacts;
  if (contacts.empty() || epsilon <= 0.0) {
    return friction_contacts;
  }

  friction_contacts.reserve(contacts.size());
  Real max_coefficient = 0.0;
  Real max_normal_force = 0.0;
  Real sum_normal_force = 0.0;
  for (const ContactStencil& contact : contacts) {
    const Real coefficient = contactFrictionCoefficient(
        geometry, contact, frictionTable, defaultCoefficient);
    if (coefficient <= 0.0) {
      continue;
    }

    FrictionContactStencil friction;
    friction.type = contact.type;
    friction.geometryIds = contact.geometryIds;
    friction.coefficient = coefficient;
    friction.epsilon = std::max<Real>(epsilon, 1.0e-12);

    const int count = contactPointCount(contact.type);
    std::array<glm::dvec3, 4> x{glm::dvec3(0.0), glm::dvec3(0.0),
                                glm::dvec3(0.0), glm::dvec3(0.0)};
    bool valid = true;
    for (int i = 0; i < count; ++i) {
      const int id = contact.geometryIds[static_cast<size_t>(i)];
      if (id < 0 || id >= geometry.pointCount()) {
        valid = false;
        break;
      }
      x[static_cast<size_t>(i)] = geometry.worldPosition(PointIdx{id});
      friction.referenceX[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
    }
    if (!valid) {
      continue;
    }

    switch (contact.type) {
      case ContactCase::PP:
        friction.weights = {1.0, -1.0, 0.0, 0.0};
        break;
      case ContactCase::PE: {
        const Real t =
            detail::clampedPointEdgeProjectionParameter(x[0], x[1], x[2]);
        friction.weights = {1.0, -(1.0 - t), -t, 0.0};
        break;
      }
      case ContactCase::PT: {
        const glm::dvec3 normal = detail::safeNormalize(
            glm::cross(x[2] - x[1], x[3] - x[1]), x[0] - x[1]);
        const glm::dvec3 projected =
            x[0] - normal * glm::dot(x[0] - x[1], normal);
        const auto bary =
            detail::triangleBarycentric(projected, x[1], x[2], x[3]);
        friction.weights = {1.0, -bary[0], -bary[1], -bary[2]};
        break;
      }
      case ContactCase::EE: {
        const auto params =
            detail::closestSegmentParameters(x[0], x[1], x[2], x[3]);
        friction.weights = {1.0 - params[0], params[0], -(1.0 - params[1]),
                            -params[1]};
        break;
      }
    }

    glm::dvec3 relative(0.0);
    for (int i = 0; i < count; ++i) {
      relative +=
          friction.weights[static_cast<size_t>(i)] * x[static_cast<size_t>(i)];
    }
    glm::dvec3 fallback = relative;
    if (contact.type == ContactCase::PT) {
      fallback = glm::cross(x[2] - x[1], x[3] - x[1]);
    } else if (contact.type == ContactCase::EE) {
      fallback = glm::cross(x[1] - x[0], x[3] - x[2]);
    }
    friction.normal = detail::safeNormalize(relative, fallback);

    friction.normalForce = estimateBarrierNormalForce(contact, x);
    if (friction.normalForce > 0.0) {
      max_coefficient = std::max(max_coefficient, friction.coefficient);
      max_normal_force = std::max(max_normal_force, friction.normalForce);
      sum_normal_force += friction.normalForce;
      friction_contacts.push_back(friction);
    }
  }

  KSK_PROFILE_VALUE("contact.friction_reference.stencils",
                    static_cast<double>(friction_contacts.size()));
  KSK_PROFILE_VALUE("contact.friction_reference.max_mu", max_coefficient);
  KSK_PROFILE_VALUE("contact.friction_reference.max_lambda", max_normal_force);
  KSK_PROFILE_VALUE("contact.friction_reference.sum_lambda", sum_normal_force);
  return friction_contacts;
}

ContactPotentialGradient computeContactFrictionGradientWrtGeometry(
    const GlobalGeometryManager& geometry,
    const FrictionContactStencils& contacts) {
  KSK_PROFILE_SCOPE("ContactFriction/ComputeGradientWrtGeometry");
  KSK_PROFILE_VALUE("contact.friction_gradient.stencils",
                    static_cast<double>(contacts.size()));
  std::vector<PointIdx> points;
  std::vector<glm::dvec3> values;
  std::unordered_map<PointIdx, int> offsets;

  for (const FrictionContactStencil& contact : contacts) {
    std::array<glm::dvec3, 4> gradient{glm::dvec3(0.0), glm::dvec3(0.0),
                                       glm::dvec3(0.0), glm::dvec3(0.0)};
    appendFrictionEnergyGradient(
        contact, gatherFrictionPositions(geometry, contact), &gradient);
    detail::scatterContactValues(contact, gradient, points, values, offsets);
  }

  KSK_PROFILE_VALUE("contact.friction_gradient.points",
                    static_cast<double>(points.size()));
  return ContactPotentialGradient{
      .points = std::move(points),
      .gradient = GeometryBuffer::FromCPU(std::move(values)),
  };
}

ContactGeometryHessian computeContactFrictionHessianWrtGeometry(
    const GlobalGeometryManager& geometry,
    const FrictionContactStencils& contacts) {
  KSK_PROFILE_SCOPE("ContactFriction/ComputeHessianWrtGeometry");
  KSK_PROFILE_VALUE("contact.friction_hessian.stencils",
                    static_cast<double>(contacts.size()));
  std::vector<ContactGeometryHessianBlock> blocks;
  std::unordered_map<std::uint64_t, int> offsets;

  for (const FrictionContactStencil& contact : contacts) {
    engine::contact::LocalBarrierHessian hessian;
    appendFrictionHessian(contact, gatherFrictionPositions(geometry, contact),
                          hessian);
    detail::scatterContactHessian(contact, hessian, blocks, offsets);
  }

  KSK_PROFILE_VALUE("contact.friction_hessian.blocks",
                    static_cast<double>(blocks.size()));
  return ContactGeometryHessian{
      .blocks = std::move(blocks),
  };
}

ContactPotentialGradient computeContactFrictionHessianProductWrtGeometry(
    const GlobalGeometryManager& geometry,
    const FrictionContactStencils& contacts,
    ConstGeometryView geometryDirection) {
  KSK_PROFILE_SCOPE("ContactFriction/ComputeHessianProductWrtGeometry");
  KSK_PROFILE_VALUE("contact.friction_hessian_product.stencils",
                    static_cast<double>(contacts.size()));
  if (!geometryDirection.isCPU()) {
    throw std::runtime_error(
        "contact friction Hessian product requires a CPU geometry direction");
  }

  std::vector<PointIdx> points;
  std::vector<glm::dvec3> values;
  std::unordered_map<PointIdx, int> offsets;

  for (const FrictionContactStencil& stencil : contacts) {
    const std::array<glm::dvec3, 4> product = frictionHessianProduct(
        stencil, gatherFrictionPositions(geometry, stencil),
        gatherFrictionDirection(stencil, geometryDirection));
    detail::scatterContactValues(stencil, product, points, values, offsets);
  }

  KSK_PROFILE_VALUE("contact.friction_hessian_product.points",
                    static_cast<double>(points.size()));
  return ContactPotentialGradient{
      .points = std::move(points),
      .gradient = GeometryBuffer::FromCPU(std::move(values)),
  };
}

double computeContactFrictionEnergyGradient(
    const FrictionContactStencil& contact,
    std::span<const glm::dvec3> positions,
    std::array<glm::dvec3, 4>* gradient) {
  std::array local_positions{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                             glm::dvec3(0.0)};
  const int count = contactPointCount(contact.type);
  if (positions.size() < static_cast<size_t>(count)) {
    throw std::invalid_argument(
        "contact friction positions do not cover stencil points");
  }
  for (int i = 0; i < count; ++i) {
    local_positions[static_cast<size_t>(i)] = positions[static_cast<size_t>(i)];
  }
  return appendFrictionEnergyGradient(contact, local_positions, gradient);
}

ContactLocalHessian computeContactFrictionHessian(
    const FrictionContactStencil& contact,
    std::span<const glm::dvec3> positions) {
  std::array local_positions{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                             glm::dvec3(0.0)};
  const int count = contactPointCount(contact.type);

  if (positions.size() < static_cast<size_t>(count)) {
    throw std::invalid_argument(
        "contact friction positions do not cover stencil points");
  }

  for (int i = 0; i < count; ++i) {
    local_positions[i] = positions[i];
  }

  engine::contact::LocalBarrierHessian local_hessian;
  appendFrictionHessian(contact, local_positions, local_hessian);
  ContactLocalHessian result;
  result.blocks = local_hessian.blocks;
  return result;
}

std::array<glm::dvec3, 4> computeContactFrictionHessianProduct(
    const FrictionContactStencil& contact,
    std::span<const glm::dvec3> positions,
    std::span<const glm::dvec3> direction) {
  std::array local_positions{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                             glm::dvec3(0.0)};
  std::array local_direction{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                             glm::dvec3(0.0)};
  const int count = contactPointCount(contact.type);
  if (positions.size() < static_cast<size_t>(count) ||
      direction.size() < static_cast<size_t>(count)) {
    throw std::invalid_argument(
        "contact friction product inputs do not cover stencil points");
  }
  for (int i = 0; i < count; ++i) {
    local_positions[static_cast<size_t>(i)] = positions[static_cast<size_t>(i)];
    local_direction[static_cast<size_t>(i)] = direction[static_cast<size_t>(i)];
  }
  return frictionHessianProduct(contact, local_positions, local_direction);
}

}  // namespace ksk::runtime
