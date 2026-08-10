#pragma once

#include <Runtime/contact-detection.h>
#include <Runtime/newton-ipc-subsystem-api.h>
#include <Runtime/simulation-context.h>

#include <memory>
#include <vector>

namespace ksk::runtime {

// Rebuilds IPC barrier/contact constraints and computes CCD step bounds.
//
// The detector is used by the Newton + IPC global solver. Internal contacts
// are routed back to subsystems through the cached NewtonIPCSubsystemSolver
// adapters (not through the base Subsystem interface).
class ContactDetector {
 public:
  // Rebuild the active barrier constraints at the scene's current geometry.
  void rebuildActiveContacts(
      SimulationContext& simulation,
      const std::vector<std::unique_ptr<NewtonIPCSubsystemSolver>>&
          subsystemApis,
      bool updateFrictionReferenceContacts = false) const;

  // Use swept CCD only to bound the step length along a proposed direction.
  [[nodiscard]] Real computeCCDStepBound(
      SimulationContext& simulation,
      const DofBuffer& direction,
      const std::vector<std::unique_ptr<NewtonIPCSubsystemSolver>>&
          subsystemApis) const;

 private:
  [[nodiscard]] static ContactDetectionConfig createDetectionConfig(
      const RuntimeScene& scene,
      Real toi);

  [[nodiscard]] static ContactDetectionConfig createCurrentBarrierConfig(
      const RuntimeScene& scene);

  static void applyRoutedContacts(
      SimulationContext& simulation,
      const GlobalContactRouter& routedContacts,
      const std::vector<std::unique_ptr<NewtonIPCSubsystemSolver>>&
          subsystemApis,
      bool updateFrictionReferenceContacts);
};

}  // namespace ksk::runtime
