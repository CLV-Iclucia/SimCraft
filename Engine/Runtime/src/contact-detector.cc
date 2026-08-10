#include <Runtime/contact-detector.h>

#include <Core/profiler.h>

#include <utility>

namespace ksk::runtime {
namespace {
GeometryBuffer createGeometryDirectionBuffer(const DofBuffer& direction,
                                             int point_count) {
  if (direction.isCPU()) return GeometryBuffer::CPU(point_count);
  return GeometryBuffer::GPU(*direction.device(), point_count);
}
}  // namespace

ContactDetectionConfig ContactDetector::createDetectionConfig(
    const RuntimeScene& scene, Real toi) {
  const GlobalSolverConfig& solver_config = *scene.solverConfig;
  const NewtonIpcSolverConfig* newton_config =
      RF::Cast<NewtonIpcSolverConfig>(&solver_config);
  return ContactDetectionConfig{
      .storage = newton_config != nullptr
                     ? newton_config->contactDetectionStorage
                     : ContactDetectionStorage::Host,
      .dHat = newton_config != nullptr ? newton_config->contactBarrierDistance
                                       : 0.0,
      .stiffness =
          newton_config != nullptr ? newton_config->contactStiffness : 1e6,
      .toi = toi,
  };
}

ContactDetectionConfig ContactDetector::createCurrentBarrierConfig(
    const RuntimeScene& scene) {
  ContactDetectionConfig config = createDetectionConfig(scene, 0.0);
  return config;
}

void ContactDetector::applyRoutedContacts(
    SimulationContext& simulation,
    const GlobalContactRouter& routedContacts,
    const std::vector<std::unique_ptr<NewtonIPCSubsystemSolver>>& subsystemApis,
    bool updateFrictionReferenceContacts) {
  KSK_PROFILE_SCOPE("ContactDetector/ApplyRoutedContacts");
  KSK_PROFILE_VALUE("contact.routed.global",
                    static_cast<double>(routedContacts.globalContacts.size()));
  KSK_PROFILE_VALUE(
      "contact.routed.internal_subsystems",
      static_cast<double>(routedContacts.subsystemInternalContacts.size()));
  RuntimeScene& scene = simulation.scene();
  if (updateFrictionReferenceContacts) {
    scene.frictionReferenceContacts.clear();
  }
  if (updateFrictionReferenceContacts && scene.solverConfig->enableFriction) {
    size_t reference_count = routedContacts.globalContacts.size();
    for (const auto& [subsystem, data] :
         routedContacts.subsystemInternalContacts) {
      (void)subsystem;
      reference_count += data.contacts.size();
    }
    scene.frictionReferenceContacts.reserve(reference_count);
    scene.frictionReferenceContacts.insert(
        scene.frictionReferenceContacts.end(),
        routedContacts.globalContacts.begin(),
        routedContacts.globalContacts.end());
    for (const auto& [subsystem, data] :
         routedContacts.subsystemInternalContacts) {
      (void)subsystem;
      scene.frictionReferenceContacts.insert(
          scene.frictionReferenceContacts.end(), data.contacts.begin(),
          data.contacts.end());
    }
  }
  scene.contacts = std::move(routedContacts.globalContacts);

  for (const auto& api : subsystemApis) {
    ContactStencils contacts;
    const auto found =
        routedContacts.subsystemInternalContacts.find(api->subsystemId());

    if (found != routedContacts.subsystemInternalContacts.end())
      contacts = std::move(found->second.contacts);

    api->applyInternalContacts(std::move(contacts));
  }
}

void ContactDetector::rebuildActiveContacts(
    SimulationContext& simulation,
    const std::vector<std::unique_ptr<NewtonIPCSubsystemSolver>>& subsystemApis,
    bool updateFrictionReferenceContacts) const {
  KSK_PROFILE_SCOPE("ContactDetector/RebuildActiveContacts");
  GlobalContactRouter routed_contacts =
      runtime::rebuildActiveContacts(
          simulation.scene().geometry,
          createCurrentBarrierConfig(simulation.scene()));
  applyRoutedContacts(simulation, std::move(routed_contacts),
                      subsystemApis, updateFrictionReferenceContacts);
}

Real ContactDetector::computeCCDStepBound(
    SimulationContext& simulation,
    const DofBuffer& direction,
    const std::vector<std::unique_ptr<NewtonIPCSubsystemSolver>>& subsystemApis)
    const {
  KSK_PROFILE_SCOPE("ContactDetector/ComputeCCDStepBound");
  RuntimeScene& scene = simulation.scene();
  GeometryBuffer geometry_direction =
      createGeometryDirectionBuffer(direction, scene.geometry.pointCount());

  {
    KSK_PROFILE_SCOPE("ContactDetector/ComputeCCDStepBound/MapDirections");
    for (const auto& api : subsystemApis) {
      api->mapLocalDirectionToGeometry(
          direction.slice(api->dofRange()), geometry_direction.view());
    }
  }

  const ContactDetectionConfig ccd_config = createDetectionConfig(scene, 1.0);
  Real step_size_upper_bound = 1.0;
  {
    KSK_PROFILE_SCOPE(
        "ContactDetector/ComputeCCDStepBound/ComputeStepSizeUpperBound");
    step_size_upper_bound = computeStepSizeUpperBoundAlongDirection(
        scene.geometry, geometry_direction, ccd_config);
  }
  return step_size_upper_bound;
}
}  // namespace ksk::runtime
