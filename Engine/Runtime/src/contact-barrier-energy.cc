#include <Contact/contact-barrier.h>
#include <Core/profiler.h>
#include <Runtime/contact-barrier-energy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/geometric.hpp>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ksk::runtime {
namespace {
int kStencilPointCount[4] = {2, 3, 4, 4};
int stencilPointCount(ContactCase type) {
  return kStencilPointCount[static_cast<int>(type)];
}

engine::contact::EBarrierStencil barrierStencilType(ContactCase type) {
  switch (type) {
    case ContactCase::PP:
      return engine::contact::EBarrierStencil::PP;
    case ContactCase::PE:
      return engine::contact::EBarrierStencil::PE;
    case ContactCase::PT:
      return engine::contact::EBarrierStencil::PT;
    case ContactCase::EE:
      return engine::contact::EBarrierStencil::EE;
  }
  return engine::contact::EBarrierStencil::PP;
}

engine::contact::EBarrierMollifierDistance barrierMollifierDistance(
    ContactMollifierDistance type) {
  switch (type) {
    case ContactMollifierDistance::None:
      return engine::contact::EBarrierMollifierDistance::None;
    case ContactMollifierDistance::PP:
      return engine::contact::EBarrierMollifierDistance::PP;
    case ContactMollifierDistance::PE:
      return engine::contact::EBarrierMollifierDistance::PE;
    case ContactMollifierDistance::EE:
      return engine::contact::EBarrierMollifierDistance::EE;
  }
  return engine::contact::EBarrierMollifierDistance::None;
}

engine::contact::GIPCBarrierStencil constructBarrierStencil(
    const GlobalGeometryManager& geometry, const ContactStencil& contact) {
  engine::contact::GIPCBarrierStencil barrier;
  barrier.type = barrierStencilType(contact.type);
  barrier.dHat = contact.dHat;
  barrier.reservedDist = contact.thickness;
  barrier.kappa = contact.stiffness;
  barrier.mollifierDistance =
      barrierMollifierDistance(contact.mollifierDistance);

  for (int i = 0; i < stencilPointCount(contact.type); ++i) {
    const PointIdx point = contact.geometryIds[i];
    barrier.x[i] = geometry.worldPosition(point);
    barrier.restX[i] = geometry.restPosition(point);
  }
  return barrier;
}

int appendPoint(std::vector<PointIdx>& points, std::vector<glm::dvec3>& values,
                std::unordered_map<PointIdx, int>& offsets, PointIdx point) {
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

bool isNonZero(const glm::dvec3& value) {
  return value.x != 0.0 || value.y != 0.0 || value.z != 0.0;
}

bool isNonZero(const glm::dmat3& value) {
  for (int col = 0; col < 3; col++) {
    for (int row = 0; row < 3; row++) {
      if (value[col][row] != 0.0) {
        return true;
      }
    }
  }
  return false;
}

int frictionPointCount(const ContactStencil& contact) {
  return stencilPointCount(contact.type);
}

bool frictionEnabled(const ContactStencil& contact) {
  return contact.frictionCoefficient > 0.0 &&
         contact.frictionNormalForce > 0.0 && contact.frictionEpsilon > 0.0;
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
    const ContactStencil& contact, const std::array<glm::dvec3, 4>& positions) {
  glm::dvec3 relative(0.0);
  const int count = frictionPointCount(contact);

  for (int i = 0; i < count; ++i)
    relative += contact.frictionWeights[i] *
                (positions[i] - contact.frictionReferenceX[i]);

  return tangentProjector(contact.frictionNormal) * relative;
}

glm::dmat3 frictionTangentHessian(const ContactStencil& contact,
                                  const glm::dvec3& displacement) {
  const glm::dmat3 projector = tangentProjector(contact.frictionNormal);
  const double eps = std::max(contact.frictionEpsilon, 1.0e-12);
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

double appendFrictionEnergyGradient(const ContactStencil& contact,
                                    const std::array<glm::dvec3, 4>& positions,
                                    std::array<glm::dvec3, 4>* gradient) {
  if (!frictionEnabled(contact)) {
    return 0.0;
  }
  const glm::dvec3 displacement =
      frictionRelativeDisplacement(contact, positions);
  const double eps = std::max(contact.frictionEpsilon, 1.0e-12);
  const double s = std::sqrt(glm::dot(displacement, displacement) + eps * eps);
  const double scale =
      contact.frictionCoefficient * contact.frictionNormalForce;

  if (gradient != nullptr) {
    const glm::dvec3 tangent_gradient = scale * displacement / s;
    const int count = frictionPointCount(contact);
    for (int i = 0; i < count; ++i) {
      (*gradient)[i] +=
          contact.frictionWeights[i] * tangent_gradient;
    }
  }

  return scale * (s - eps);
}

void appendFrictionHessian(const ContactStencil& contact,
                           const std::array<glm::dvec3, 4>& positions,
                           engine::contact::LocalBarrierHessian& hessian) {
  if (!frictionEnabled(contact)) {
    return;
  }
  const glm::dvec3 displacement =
      frictionRelativeDisplacement(contact, positions);
  const glm::dmat3 tangent_hessian =
      frictionTangentHessian(contact, displacement) *
      (contact.frictionCoefficient * contact.frictionNormalForce);
  const int count = frictionPointCount(contact);
  for (int row = 0; row < count; ++row) {
    for (int col = 0; col < count; ++col) {
      hessian.blocks[row][col] +=
          contact.frictionWeights[row] *
          contact.frictionWeights[col] * tangent_hessian;
    }
  }
}

std::array<glm::dvec3, 4> frictionHessianProduct(
    const ContactStencil& contact, const std::array<glm::dvec3, 4>& positions,
    const std::array<glm::dvec3, 4>& direction) {
  std::array result{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                    glm::dvec3(0.0)};
  if (!frictionEnabled(contact)) {
    return result;
  }
  glm::dvec3 relative_direction(0.0);
  const int count = frictionPointCount(contact);
  for (int i = 0; i < count; ++i) {
    relative_direction += contact.frictionWeights[i] * direction[i];
  }

  const glm::dvec3 displacement =
      frictionRelativeDisplacement(contact, positions);

  const glm::dmat3 tangent_hessian =
      frictionTangentHessian(contact, displacement) *
      (contact.frictionCoefficient * contact.frictionNormalForce);
  const glm::dvec3 tangent_product = tangent_hessian * relative_direction;
  for (int i = 0; i < count; ++i) {
    result[static_cast<size_t>(i)] +=
        contact.frictionWeights[static_cast<size_t>(i)] * tangent_product;
  }
  return result;
}

std::array<glm::dvec3, 4> gatherContactPositions(
    const GlobalGeometryManager& geometry, const ContactStencil& contact) {
  std::array positions{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                       glm::dvec3(0.0)};
  const int count = stencilPointCount(contact.type);
  for (int i = 0; i < count; ++i) {
    positions[i] = geometry.worldPosition(PointIdx{contact.geometryIds[i]});
  }
  return positions;
}

std::uint64_t pointPairKey(PointIdx row, PointIdx col) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row)) << 32) |
         static_cast<std::uint32_t>(col);
}

std::array<glm::dvec3, 4> gatherBarrierDirection(
    const ContactStencil& contact, const ConstGeometryView& geometryDirection) {
  std::array direction{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                       glm::dvec3(0.0)};
  for (int i = 0; i < stencilPointCount(contact.type); i++) {
    direction[static_cast<size_t>(i)] =
        geometryDirection.cpu()[contact.geometryIds[static_cast<size_t>(i)]];
  }
  return direction;
}

void scatterBarrierValues(const ContactStencil& contact,
                          const std::array<glm::dvec3, 4>& values,
                          std::vector<PointIdx>& points,
                          std::vector<glm::dvec3>& out,
                          std::unordered_map<PointIdx, int>& offsets) {
  for (int i = 0; i < stencilPointCount(contact.type); i++) {
    if (!isNonZero(values[i])) {
      continue;
    }

    const int offset = appendPoint(points, out, offsets,
                                   contact.geometryIds[static_cast<size_t>(i)]);
    out[offset] += values[i];
  }
}

void scatterBarrierHessian(const ContactStencil& contact,
                           const engine::contact::LocalBarrierHessian& hessian,
                           std::vector<ContactGeometryHessianBlock>& blocks,
                           std::unordered_map<std::uint64_t, int>& offsets) {
  const int count = stencilPointCount(contact.type);
  for (int row = 0; row < count; row++) {
      const PointIdx row_point = contact.geometryIds[static_cast<size_t>(row)];
    for (int col = 0; col < count; col++) {
      const glm::dmat3& value = hessian.blocks[row][col];
      if (!isNonZero(value)) {
        continue;
      }

      const PointIdx col_point = contact.geometryIds[col];
      const std::uint64_t key = pointPairKey(row_point, col_point);
      const auto found = offsets.find(key);
      if (found != offsets.end()) {
        blocks[found->second].value += value;
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
}  // namespace

double computeContactEnergy(const GlobalGeometryManager& geometry,
                            const ContactStencils& contacts) {
  KSK_PROFILE_SCOPE("ContactBarrierEnergy/ComputeEnergy");
  KSK_PROFILE_VALUE("contact.energy.stencils",
                    static_cast<double>(contacts.size()));
  double energy = 0.0;
  for (const ContactStencil& contact : contacts) {
    const engine::contact::GIPCBarrierStencil barrier =
        constructBarrierStencil(geometry, contact);
    energy += engine::contact::computeGIPCBarrierEnergy(barrier);
  }
  return energy;
}

double computeContactFrictionEnergy(const GlobalGeometryManager& geometry,
                                    const ContactStencils& contacts) {
  KSK_PROFILE_SCOPE("ContactBarrierEnergy/ComputeFrictionEnergy");
  KSK_PROFILE_VALUE("contact.friction_energy.stencils",
                    static_cast<double>(contacts.size()));
  double energy = 0.0;
  for (const ContactStencil& contact : contacts) {
    energy += appendFrictionEnergyGradient(
        contact, gatherContactPositions(geometry, contact), nullptr);
  }
  return energy;
}

ContactPotentialGradient computeContactGradientWrtGeometry(
    const GlobalGeometryManager& geometry, const ContactStencils& contacts) {
  KSK_PROFILE_SCOPE("ContactBarrierEnergy/ComputeGradientWrtGeometry");
  KSK_PROFILE_VALUE("contact.gradient.stencils",
                    static_cast<double>(contacts.size()));
  std::vector<PointIdx> points;
  std::vector<glm::dvec3> values;
  std::unordered_map<PointIdx, int> offsets;

  for (const ContactStencil& contact : contacts) {
    const engine::contact::GIPCBarrierStencil barrier =
        constructBarrierStencil(geometry, contact);
    const engine::contact::LocalBarrierGradient localGradient =
        engine::contact::computeGIPCBarrierGradient(barrier);
    scatterBarrierValues(contact, localGradient.gradient, points, values,
                         offsets);
  }

  KSK_PROFILE_VALUE("contact.gradient.points",
                    static_cast<double>(points.size()));
  return ContactPotentialGradient{
      .points = std::move(points),
      .gradient = GeometryBuffer::FromCPU(std::move(values)),
  };
}

ContactPotentialGradient computeContactGradient(
    const GlobalGeometryManager& geometry, const ContactStencils& contacts) {
  return computeContactGradientWrtGeometry(geometry, contacts);
}

ContactPotentialGradient computeContactFrictionGradientWrtGeometry(
    const GlobalGeometryManager& geometry, const ContactStencils& contacts) {
  KSK_PROFILE_SCOPE("ContactBarrierEnergy/ComputeFrictionGradientWrtGeometry");
  KSK_PROFILE_VALUE("contact.friction_gradient.stencils",
                    static_cast<double>(contacts.size()));
  std::vector<PointIdx> points;
  std::vector<glm::dvec3> values;
  std::unordered_map<PointIdx, int> offsets;

  for (const ContactStencil& contact : contacts) {
    std::array<glm::dvec3, 4> gradient{glm::dvec3(0.0), glm::dvec3(0.0),
                                       glm::dvec3(0.0), glm::dvec3(0.0)};
    appendFrictionEnergyGradient(
        contact, gatherContactPositions(geometry, contact), &gradient);
    scatterBarrierValues(contact, gradient, points, values, offsets);
  }

  KSK_PROFILE_VALUE("contact.friction_gradient.points",
                    static_cast<double>(points.size()));
  return ContactPotentialGradient{
      .points = std::move(points),
      .gradient = GeometryBuffer::FromCPU(std::move(values)),
  };
}

ContactGeometryHessian computeContactHessianWrtGeometry(
    const GlobalGeometryManager& geometry, const ContactStencils& contacts) {
  KSK_PROFILE_SCOPE("ContactBarrierEnergy/ComputeHessianWrtGeometry");
  KSK_PROFILE_VALUE("contact.hessian.stencils",
                    static_cast<double>(contacts.size()));
  std::vector<ContactGeometryHessianBlock> blocks;
  std::unordered_map<std::uint64_t, int> offsets;

  for (const ContactStencil& contact : contacts) {
    const engine::contact::GIPCBarrierStencil barrier =
        constructBarrierStencil(geometry, contact);
    const engine::contact::LocalBarrierHessian hessian =
        engine::contact::computeGIPCBarrierHessian(barrier);
    scatterBarrierHessian(contact, hessian, blocks, offsets);
  }

  KSK_PROFILE_VALUE("contact.hessian.blocks",
                    static_cast<double>(blocks.size()));
  return ContactGeometryHessian{
      .blocks = std::move(blocks),
  };
}

ContactGeometryHessian computeContactFrictionHessianWrtGeometry(
    const GlobalGeometryManager& geometry, const ContactStencils& contacts) {
  KSK_PROFILE_SCOPE("ContactBarrierEnergy/ComputeFrictionHessianWrtGeometry");
  KSK_PROFILE_VALUE("contact.friction_hessian.stencils",
                    static_cast<double>(contacts.size()));
  std::vector<ContactGeometryHessianBlock> blocks;
  std::unordered_map<std::uint64_t, int> offsets;

  for (const ContactStencil& contact : contacts) {
    engine::contact::LocalBarrierHessian hessian;
    appendFrictionHessian(contact, gatherContactPositions(geometry, contact),
                          hessian);
    scatterBarrierHessian(contact, hessian, blocks, offsets);
  }

  KSK_PROFILE_VALUE("contact.friction_hessian.blocks",
                    static_cast<double>(blocks.size()));
  return ContactGeometryHessian{
      .blocks = std::move(blocks),
  };
}

ContactPotentialGradient computeContactHessianProductWrtGeometry(
    const GlobalGeometryManager& geometry, const ContactStencils& contacts,
    ConstGeometryView geometryDirection) {
  KSK_PROFILE_SCOPE("ContactBarrierEnergy/ComputeHessianProductWrtGeometry");
  KSK_PROFILE_VALUE("contact.hessian_product.stencils",
                    static_cast<double>(contacts.size()));
  if (!geometryDirection.isCPU()) {
    throw std::runtime_error(
        "GIPC contact Hessian product requires a CPU geometry direction");
  }

  std::vector<PointIdx> points;
  std::vector<glm::dvec3> values;
  std::unordered_map<PointIdx, int> offsets;

  for (const ContactStencil& stencil : contacts) {
    const engine::contact::GIPCBarrierStencil barrier =
        constructBarrierStencil(geometry, stencil);

    const std::array<glm::dvec3, 4> product =
        ksk::engine::contact::computeGIPCBarrierHessianProduct(
            barrier, gatherBarrierDirection(stencil, geometryDirection));

    scatterBarrierValues(stencil, product, points, values, offsets);
  }

  KSK_PROFILE_VALUE("contact.hessian_product.points",
                    static_cast<double>(points.size()));
  return ContactPotentialGradient{
      .points = std::move(points),
      .gradient = GeometryBuffer::FromCPU(std::move(values)),
  };
}

ContactPotentialGradient computeContactHessianProduct(
    const GlobalGeometryManager& geometry, const ContactStencils& contacts,
    ConstGeometryView geometryDirection) {
  return computeContactHessianProductWrtGeometry(geometry, contacts,
                                                 geometryDirection);
}

ContactPotentialGradient computeContactFrictionHessianProductWrtGeometry(
    const GlobalGeometryManager& geometry, const ContactStencils& contacts,
    ConstGeometryView geometryDirection) {
  KSK_PROFILE_SCOPE(
      "ContactBarrierEnergy/ComputeFrictionHessianProductWrtGeometry");
  KSK_PROFILE_VALUE("contact.friction_hessian_product.stencils",
                    static_cast<double>(contacts.size()));
  if (!geometryDirection.isCPU()) {
    throw std::runtime_error(
        "contact friction Hessian product requires a CPU geometry direction");
  }

  std::vector<PointIdx> points;
  std::vector<glm::dvec3> values;
  std::unordered_map<PointIdx, int> offsets;

  for (const ContactStencil& stencil : contacts) {
    const std::array<glm::dvec3, 4> product = frictionHessianProduct(
        stencil, gatherContactPositions(geometry, stencil),
        gatherBarrierDirection(stencil, geometryDirection));
    scatterBarrierValues(stencil, product, points, values, offsets);
  }

  KSK_PROFILE_VALUE("contact.friction_hessian_product.points",
                    static_cast<double>(points.size()));
  return ContactPotentialGradient{
      .points = std::move(points),
      .gradient = GeometryBuffer::FromCPU(std::move(values)),
  };
}

double computeContactFrictionEnergyGradient(
    const ContactStencil& contact, std::span<const glm::dvec3> positions,
    std::array<glm::dvec3, 4>* gradient) {
  std::array local_positions{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                             glm::dvec3(0.0)};
  const int count = stencilPointCount(contact.type);
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
    const ContactStencil& contact, std::span<const glm::dvec3> positions) {
  std::array local_positions{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                             glm::dvec3(0.0)};
  const int count = stencilPointCount(contact.type);

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
    const ContactStencil& contact, std::span<const glm::dvec3> positions,
    std::span<const glm::dvec3> direction) {
  std::array local_positions{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                             glm::dvec3(0.0)};
  std::array local_direction{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                             glm::dvec3(0.0)};
  const int count = stencilPointCount(contact.type);
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
