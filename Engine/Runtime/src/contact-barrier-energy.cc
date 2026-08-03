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

#include "contact-accumulation-utils.h"

namespace ksk::runtime {
namespace {
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

  for (int i = 0; i < contactPointCount(contact.type); ++i) {
    const PointIdx point = contact.geometryIds[i];
    barrier.x[i] = geometry.worldPosition(point);
    barrier.restX[i] = geometry.restPosition(point);
  }
  return barrier;
}

std::array<glm::dvec3, 4> gatherContactPositions(
    const GlobalGeometryManager& geometry, const ContactStencil& contact) {
  std::array positions{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                       glm::dvec3(0.0)};
  const int count = contactPointCount(contact.type);
  for (int i = 0; i < count; ++i) {
    positions[i] = geometry.worldPosition(PointIdx{contact.geometryIds[i]});
  }
  return positions;
}

std::array<glm::dvec3, 4> gatherBarrierDirection(
    const ContactStencil& contact, const ConstGeometryView& geometryDirection) {
  std::array direction{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                       glm::dvec3(0.0)};
  for (int i = 0; i < contactPointCount(contact.type); i++) {
    direction[static_cast<size_t>(i)] =
        geometryDirection.cpu()[contact.geometryIds[static_cast<size_t>(i)]];
  }
  return direction;
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
    detail::scatterContactValues(contact, localGradient.gradient, points,
                                 values, offsets);
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
    detail::scatterContactHessian(contact, hessian, blocks, offsets);
  }

  KSK_PROFILE_VALUE("contact.hessian.blocks",
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

    detail::scatterContactValues(stencil, product, points, values, offsets);
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

}  // namespace ksk::runtime
