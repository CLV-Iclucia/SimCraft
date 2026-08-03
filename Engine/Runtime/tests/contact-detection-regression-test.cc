#include <Runtime/contact-detection.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

namespace ksk::runtime {
namespace {

bool hasPointTriangleCandidate(const ContactCandidates& candidates,
                               PointIdx point) {
  return std::ranges::any_of(candidates, [&](const ContactCandidate& candidate) {
    return candidate.type == EContactCandidate::PointTriangle &&
           candidate.geometryIds[0] == point;
  });
}

TEST(ContactDetectionRegression, FindsSweptDeformableDeformablePointTriangle)
{
  GlobalGeometryManager geometry;
  const PointIdx point =
      geometry.addPoint(SubsystemIdx{0}, 0, glm::dvec3{0.25, 0.25, 1.0});
  const PointIdx a =
      geometry.addPoint(SubsystemIdx{1}, 0, glm::dvec3{0.0, 0.0, 0.0});
  const PointIdx b =
      geometry.addPoint(SubsystemIdx{1}, 1, glm::dvec3{1.0, 0.0, 0.0});
  const PointIdx c =
      geometry.addPoint(SubsystemIdx{1}, 2, glm::dvec3{0.0, 1.0, 0.0});
  geometry.addTriangle(a, b, c);

  GeometryBuffer direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(point)] = glm::dvec3{0.0, 0.0, -2.0};

  const ContactDetectionConfig config{
      .dHat = 0.01,
      .stiffness = 1.0,
      .toi = 1.0,
  };
  const ContactCandidates candidates =
      detectContactCandidatesAlongDirection(
          geometry,
          direction,
          config);
  const Real step_size_upper_bound =
      computeStepSizeUpperBoundAlongDirection(geometry, direction, config);

  EXPECT_TRUE(hasPointTriangleCandidate(candidates, point));
  EXPECT_LT(step_size_upper_bound, 1.0);
}

TEST(ContactDetectionRegression, FindsSweptDeformableRigidPointTriangle)
{
  GlobalGeometryManager geometry;
  const PointIdx point =
      geometry.addPoint(SubsystemIdx{0}, 0, glm::dvec3{0.25, 0.25, 1.0});
  const GeometryBodyIdx rigid_body = geometry.addGeometryBody(
      GeometryBodyInfo{
          .type = EGeometryBody::Rigid,
          .motion = EBodyMotionMode::Dynamic,
          .subsystem = SubsystemIdx{1},
          .localBody = 0,
      },
      GeometryBodyCollisionFilter{
          .group = kDefaultCollisionGroup,
          .mask = kAllCollisionGroups,
          .enabled = true,
      });
  const GeometryInstanceIdx rigid_instance = geometry.addInstance(
      GeometryOwner{
          .body = rigid_body,
          .subsystem = SubsystemIdx{1},
      },
      geometry.addGeometryPrototype(GeometryMeshDesc{
          .vertices = {
              glm::dvec3{0.0, 0.0, 0.0},
              glm::dvec3{1.0, 0.0, 0.0},
              glm::dvec3{0.0, 1.0, 0.0},
          },
          .triangles = {
              std::array<int, 3>{0, 1, 2},
          },
      }),
      0);
  (void)rigid_instance;

  GeometryBuffer direction = GeometryBuffer::CPU(geometry.pointCount());
  direction.cpu()[static_cast<size_t>(point)] = glm::dvec3{0.0, 0.0, -2.0};

  const ContactDetectionConfig config{
      .dHat = 0.01,
      .stiffness = 1.0,
      .toi = 1.0,
  };
  const ContactCandidates candidates =
      detectContactCandidatesAlongDirection(
          geometry,
          direction,
          config);
  const Real step_size_upper_bound =
      computeStepSizeUpperBoundAlongDirection(geometry, direction, config);

  EXPECT_TRUE(hasPointTriangleCandidate(candidates, point));
  EXPECT_LT(step_size_upper_bound, 1.0);
}

}  // namespace
}  // namespace ksk::runtime
