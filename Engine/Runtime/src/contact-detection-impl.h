#pragma once

#include <Runtime/contact-detection.h>

namespace ksk::runtime::detail {

[[nodiscard]] GlobalContactRouter rebuildActiveContactsOnCPU(
    const GlobalGeometryManager& geometry,
    const ContactDetectionConfig& config);

[[nodiscard]] ContactCandidates detectContactCandidatesAlongDirectionOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config);

[[nodiscard]] Real computeStepSizeUpperBoundAlongDirectionOnCPU(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config);

}  // namespace ksk::runtime::detail
