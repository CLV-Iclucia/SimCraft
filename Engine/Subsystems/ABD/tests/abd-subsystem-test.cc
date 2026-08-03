#include <ABD/abd-scene.h>
#include <ABD/abd-subsystem.h>

#include <Runtime/contact-detection.h>
#include <Runtime/simulation-context.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

namespace ksk::abd {
namespace {

ABDBodyDesc makeTriangleBody()
{
  ABDBodyDesc body;
  body.surface.vertices = {
      glm::dvec3{0.0, 0.0, 0.0},
      glm::dvec3{1.0, 0.0, 0.0},
      glm::dvec3{0.0, 1.0, 0.0},
  };
  body.surface.edges = {
      std::array<int, 2>{0, 1},
      std::array<int, 2>{1, 2},
      std::array<int, 2>{2, 0},
  };
  body.surface.triangles = {
      std::array<int, 3>{0, 1, 2},
  };
  return body;
}

ABDBodyDesc makePointBody(glm::dvec3 point)
{
  ABDBodyDesc body;
  body.surface.vertices = {point};
  return body;
}

ABDBodyDesc makeUnitTetraSurfaceBody()
{
  ABDBodyDesc body;
  body.surface.vertices = {
      glm::dvec3{0.0, 0.0, 0.0},
      glm::dvec3{1.0, 0.0, 0.0},
      glm::dvec3{0.0, 1.0, 0.0},
      glm::dvec3{0.0, 0.0, 1.0},
  };
  body.surface.triangles = {
      std::array<int, 3>{0, 2, 1},
      std::array<int, 3>{0, 1, 3},
      std::array<int, 3>{0, 3, 2},
      std::array<int, 3>{1, 2, 3},
  };
  body.density = 6.0;
  return body;
}

TEST(ABDSubsystem, OneBodyUsesTwelveDofs)
{
  ABDSubsystem subsystem(runtime::SubsystemIdx{3}, {makeTriangleBody()}, 7);

  const runtime::DofRange range = subsystem.dofRange();
  EXPECT_EQ(range.subsystem, runtime::SubsystemIdx{3});
  EXPECT_EQ(range.scalarOffset, 7);
  EXPECT_EQ(range.scalarCount, 12);
  EXPECT_EQ(range.blockSize, 12);
}

TEST(ABDSubsystem, SceneGroupingBuildsOneSubsystemWithBodyBlocks)
{
  runtime::RuntimeSceneDesc scene;
  const runtime::ObjectRef first =
      addABDBody(scene, makeTriangleBody(), "first");
  const runtime::ObjectRef second =
      addABDBody(scene, makeTriangleBody(), "second");

  runtime::SimulationContext context = runtime::buildSimulation(scene);

  EXPECT_TRUE(first.isValid());
  EXPECT_TRUE(second.isValid());
  ASSERT_EQ(context.subsystems().size(), 1);
  const auto* subsystem =
      static_cast<const ABDSubsystem*>(context.subsystems().front().get());
  ASSERT_NE(subsystem, nullptr);
  EXPECT_EQ(subsystem->dofRange().scalarCount, 24);
  EXPECT_EQ(subsystem->dofRange().blockSize, 12);
}

TEST(ABDSubsystem, DeclareGeometryCreatesRigidInstanceBody)
{
  ABDSubsystem subsystem(runtime::SubsystemIdx{2}, {makeTriangleBody()});
  runtime::GlobalGeometryManager geometry;

  subsystem.declareGeometry(geometry);

  ASSERT_EQ(subsystem.runtimeBodies().size(), 1);
  const ABDBodyRuntimeRef& body = subsystem.runtimeBodies().front();
  ASSERT_GE(body.geometryBody, 0);
  ASSERT_GE(body.surfacePrototype, 0);
  ASSERT_GE(body.instance, 0);

  EXPECT_EQ(geometry.geometryBody(body.geometryBody).type,
            runtime::EGeometryBody::Rigid);
  EXPECT_EQ(geometry.geometryBody(body.geometryBody).subsystem,
            runtime::SubsystemIdx{2});
  EXPECT_EQ(geometry.geometryBody(body.geometryBody).localBody, 0);
  EXPECT_EQ(geometry.instances[static_cast<size_t>(body.instance)].prototype,
            body.surfacePrototype);
  EXPECT_EQ(geometry.bodyInstanceRange(body.geometryBody).count, 1);
  EXPECT_EQ(geometry.instancePointRange(body.instance).count, 3);
  EXPECT_EQ(subsystem.geometryPointIds().size(), 3);
}

TEST(ABDSubsystem, UpdateGeometryChangesInstanceTransform)
{
  ABDSubsystem subsystem(runtime::SubsystemIdx{0}, {makeTriangleBody()});
  runtime::GlobalGeometryManager geometry;
  subsystem.declareGeometry(geometry);

  glm::dmat4x3 transform{1.0};
  transform[3] = glm::dvec3{4.0, -2.0, 3.0};
  subsystem.setBodyTransform(0, transform);
  subsystem.updateGeometry(geometry);

  const runtime::GeometryInstanceIdx instance =
      subsystem.runtimeBodies().front().instance;
  EXPECT_EQ(geometry.instances[static_cast<size_t>(instance)].transform[3],
            glm::dvec4(4.0, -2.0, 3.0, 1.0));

  const runtime::PointIdx point = geometry.instancePoint(instance, 1);
  EXPECT_DOUBLE_EQ(geometry.worldPosition(point).x, 5.0);
  EXPECT_DOUBLE_EQ(geometry.worldPosition(point).y, -2.0);
  EXPECT_DOUBLE_EQ(geometry.worldPosition(point).z, 3.0);
}

TEST(ABDSubsystem, BodyPairIAABBGatherSkipsSameBodyAndFindsCrossBodyWork)
{
  ABDSubsystem subsystem(runtime::SubsystemIdx{0},
                         {makeTriangleBody(), makeTriangleBody()});
  runtime::GlobalGeometryManager geometry;
  subsystem.declareGeometry(geometry);

  runtime::GeometryBuffer zero_direction =
      runtime::GeometryBuffer::CPU(geometry.pointCount());
  const runtime::ContactWorkList work =
      runtime::gatherCollisionWorkListAlongDirection(
          geometry,
          zero_direction,
          runtime::ContactDetectionConfig{
              .dHat = 0.25,
              .stiffness = 1.0,
              .toi = 1.0,
          });

  EXPECT_EQ(work.bodyPairs.size(), 1);
  EXPECT_EQ(work.bodyPairs.front().first,
            subsystem.runtimeBodies()[0].geometryBody);
  EXPECT_EQ(work.bodyPairs.front().second,
            subsystem.runtimeBodies()[1].geometryBody);
  EXPECT_FALSE(work.pointTriangles.empty());
  EXPECT_FALSE(work.edgeEdges.empty());
}

TEST(ABDSubsystem, InternalPointContactContributesEnergy)
{
  ABDBodyDesc first = makePointBody(glm::dvec3{0.0});
  ABDBodyDesc second = makePointBody(glm::dvec3{0.1, 0.0, 0.0});
  ABDSubsystem subsystem(runtime::SubsystemIdx{0}, {first, second});
  runtime::GlobalGeometryManager geometry;
  subsystem.declareGeometry(geometry);

  const runtime::PointIdx p0 =
      subsystem.geometryPointIds()[static_cast<size_t>(0)];
  const runtime::PointIdx p1 =
      subsystem.geometryPointIds()[static_cast<size_t>(1)];
  subsystem.applyInternalContacts({
      runtime::ContactStencil{
          .type = runtime::ContactCase::PP,
          .geometryIds = {p0, p1, -1, -1},
          .dHat = 0.5,
          .stiffness = 10.0,
          .thickness = 0.0,
      },
  });

  std::vector<double> q(static_cast<size_t>(subsystem.localScalarCount()));
  std::vector<double> qdot(static_cast<size_t>(subsystem.localScalarCount()));
  subsystem.writeState(
      runtime::DofView::CPU(q.data(), static_cast<int>(q.size())),
      runtime::DofView::CPU(qdot.data(), static_cast<int>(qdot.size())));
  subsystem.beginStep(
      runtime::ConstDofView::CPU(q.data(), static_cast<int>(q.size())),
      runtime::ConstDofView::CPU(qdot.data(), static_cast<int>(qdot.size())),
      1.0);

  EXPECT_GT(subsystem.evaluateObjective(
                runtime::ConstDofView::CPU(q.data(), static_cast<int>(q.size())),
                runtime::ConstDofView::CPU(qdot.data(),
                                           static_cast<int>(qdot.size())),
                1.0),
            0.0);
}

TEST(ABDSubsystem, ClosedSurfaceMassUsesVolumeIntegral)
{
  ABDSubsystem subsystem(runtime::SubsystemIdx{0}, {makeUnitTetraSurfaceBody()});

  std::vector<double> q(static_cast<size_t>(subsystem.localScalarCount()));
  std::vector<double> qdot(static_cast<size_t>(subsystem.localScalarCount()));
  subsystem.writeState(
      runtime::DofView::CPU(q.data(), static_cast<int>(q.size())),
      runtime::DofView::CPU(qdot.data(), static_cast<int>(qdot.size())));
  subsystem.beginStep(
      runtime::ConstDofView::CPU(q.data(), static_cast<int>(q.size())),
      runtime::ConstDofView::CPU(qdot.data(), static_cast<int>(qdot.size())),
      1.0);

  q[9] += 2.0;

  EXPECT_NEAR(subsystem.evaluateObjective(
                  runtime::ConstDofView::CPU(q.data(), static_cast<int>(q.size())),
                  runtime::ConstDofView::CPU(qdot.data(),
                                             static_cast<int>(qdot.size())),
                  1.0),
              2.0,
              1.0e-12);
}

TEST(ABDSubsystem, OrthogonalPotentialIsZeroForIdentityLinearPart)
{
  ABDBodyDesc body = makeUnitTetraSurfaceBody();
  body.orthogonalStiffness = 7.0;
  ABDSubsystem subsystem(runtime::SubsystemIdx{0}, {body});

  std::vector<double> q(static_cast<size_t>(subsystem.localScalarCount()));
  std::vector<double> qdot(static_cast<size_t>(subsystem.localScalarCount()));
  subsystem.writeState(
      runtime::DofView::CPU(q.data(), static_cast<int>(q.size())),
      runtime::DofView::CPU(qdot.data(), static_cast<int>(qdot.size())));
  subsystem.beginStep(
      runtime::ConstDofView::CPU(q.data(), static_cast<int>(q.size())),
      runtime::ConstDofView::CPU(qdot.data(), static_cast<int>(qdot.size())),
      1.0);

  EXPECT_NEAR(subsystem.evaluateObjective(
                  runtime::ConstDofView::CPU(q.data(), static_cast<int>(q.size())),
                  runtime::ConstDofView::CPU(qdot.data(),
                                             static_cast<int>(qdot.size())),
                  1.0),
              0.0,
              1.0e-12);
}

TEST(ABDSubsystem, OrthogonalPotentialGradientMatchesFiniteDifference)
{
  ABDBodyDesc body = makeUnitTetraSurfaceBody();
  body.orthogonalStiffness = 11.0;
  body.initialTransform[0] = glm::dvec3{1.15, 0.05, -0.02};
  body.initialTransform[1] = glm::dvec3{0.03, 0.92, 0.04};
  body.initialTransform[2] = glm::dvec3{-0.01, 0.07, 1.08};
  ABDSubsystem subsystem(runtime::SubsystemIdx{0}, {body});

  std::vector<double> q(static_cast<size_t>(subsystem.localScalarCount()));
  std::vector<double> qdot(static_cast<size_t>(subsystem.localScalarCount()));
  std::vector<double> gradient(static_cast<size_t>(subsystem.localScalarCount()));
  subsystem.writeState(
      runtime::DofView::CPU(q.data(), static_cast<int>(q.size())),
      runtime::DofView::CPU(qdot.data(), static_cast<int>(qdot.size())));
  subsystem.beginStep(
      runtime::ConstDofView::CPU(q.data(), static_cast<int>(q.size())),
      runtime::ConstDofView::CPU(qdot.data(), static_cast<int>(qdot.size())),
      1.0);
  subsystem.readState(
      runtime::ConstDofView::CPU(q.data(), static_cast<int>(q.size())),
      runtime::ConstDofView::CPU(qdot.data(), static_cast<int>(qdot.size())));
  subsystem.assembleLocalGradient(
      runtime::DofView::CPU(gradient.data(), static_cast<int>(gradient.size())));

  constexpr double eps = 1.0e-6;
  for (int dof = 0; dof < 9; ++dof) {
    std::vector<double> qp = q;
    std::vector<double> qm = q;
    qp[static_cast<size_t>(dof)] += eps;
    qm[static_cast<size_t>(dof)] -= eps;
    const double ep = subsystem.evaluateObjective(
        runtime::ConstDofView::CPU(qp.data(), static_cast<int>(qp.size())),
        runtime::ConstDofView::CPU(qdot.data(), static_cast<int>(qdot.size())),
        1.0);
    const double em = subsystem.evaluateObjective(
        runtime::ConstDofView::CPU(qm.data(), static_cast<int>(qm.size())),
        runtime::ConstDofView::CPU(qdot.data(), static_cast<int>(qdot.size())),
        1.0);
    EXPECT_NEAR(gradient[static_cast<size_t>(dof)],
                (ep - em) / (2.0 * eps),
                1.0e-6);
  }
  EXPECT_NEAR(gradient[9], 0.0, 1.0e-12);
  EXPECT_NEAR(gradient[10], 0.0, 1.0e-12);
  EXPECT_NEAR(gradient[11], 0.0, 1.0e-12);
}

}  // namespace
}  // namespace ksk::abd
