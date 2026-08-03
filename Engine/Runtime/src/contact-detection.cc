#include <Runtime/contact-detection.h>

#include "contact-detection-impl.h"

#include <stdexcept>

namespace ksk::runtime {

GlobalContactRouter rebuildActiveContacts(
    const GlobalGeometryManager& geometry,
    const ContactDetectionConfig& config)
{
    return detail::rebuildActiveContactsOnCPU(geometry, config);
}

ContactCandidates detectContactCandidatesAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
    if (geometryDirection.isGPU()) {
        throw std::runtime_error(
            "detectContactCandidatesAlongDirection requested host tables but got device tables");
    }
    return detail::detectContactCandidatesAlongDirectionOnCPU(
        geometry, geometryDirection, config);
}

Real computeStepSizeUpperBoundAlongDirection(
    const GlobalGeometryManager& geometry,
    const GeometryBuffer& geometryDirection,
    const ContactDetectionConfig& config)
{
    if (geometryDirection.isGPU()) {
        throw std::runtime_error(
            "computeStepSizeUpperBoundAlongDirection requested host tables but got device tables");
    }
    return detail::computeStepSizeUpperBoundAlongDirectionOnCPU(
        geometry, geometryDirection, config);
}

}  // namespace ksk::runtime
