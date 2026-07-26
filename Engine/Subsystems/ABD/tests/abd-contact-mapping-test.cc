#include <ABD/abd-scene.h>
#include <ABD/abd-subsystem.h>

#include <Runtime/buffers.h>

#include <gtest/gtest.h>

#include <glm/geometric.hpp>

#include <array>
#include <span>
#include <vector>

namespace ksk::abd {
namespace {

ABDBodyDesc makeMappingBody()
{
  ABDBodyDesc body;
  body.surface.vertices = {
      glm::dvec3{-1.0, 0.5, 0.0},
      glm::dvec3{2.0, -0.25, 1.0},
      glm::dvec3{0.75, 1.25, -0.5},
  };
  body.surface.edges = {
      std::array<int, 2>{0, 1},
      std::array<int, 2>{1, 2},
  };
  body.surface.triangles = {
      std::array<int, 3>{0, 1, 2},
  };
  return body;
}

double dot(runtime::ConstDofView a, runtime::ConstDofView b)
{
  double result = 0.0;
  for (int i = 0; i < a.scalarCount(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

double dot(std::span<const runtime::PointIdx> points,
           runtime::ConstGeometryView a,
           runtime::ConstGeometryView b)
{
  double result = 0.0;
  for (int i = 0; i < static_cast<int>(points.size()); ++i) {
    result += glm::dot(a[points[static_cast<size_t>(i)]],
                       b[points[static_cast<size_t>(i)]]);
  }
  return result;
}

TEST(ABDContactMapping, DirectionAndGradientAreAdjoint)
{
  ABDSubsystem subsystem(runtime::SubsystemIdx{0}, {makeMappingBody()});
  runtime::GlobalGeometryManager geometry;
  subsystem.declareGeometry(geometry);

  std::vector<double> dq_values(12, 0.0);
  for (int i = 0; i < static_cast<int>(dq_values.size()); ++i) {
    dq_values[static_cast<size_t>(i)] = 0.1 * static_cast<double>(i + 1);
  }

  runtime::GeometryBuffer dx = runtime::GeometryBuffer::CPU(geometry.pointCount());
  subsystem.mapLocalDirectionToGeometry(
      runtime::ConstDofView::CPU(dq_values.data(),
                                 static_cast<int>(dq_values.size())),
      dx.view());

  const std::vector<runtime::PointIdx>& points = subsystem.geometryPointIds();
  std::vector<glm::dvec3> point_gradient_values = {
      glm::dvec3{0.25, -0.5, 1.0},
      glm::dvec3{-1.5, 0.75, 0.5},
      glm::dvec3{0.3, 2.0, -0.7},
  };
  std::vector<double> pulled_values(12, 0.0);

  subsystem.scatterContactGradient(
      points,
      runtime::ConstGeometryView::CPU(
          point_gradient_values.data(),
          static_cast<int>(point_gradient_values.size())),
      runtime::DofView::CPU(pulled_values.data(),
                            static_cast<int>(pulled_values.size())));

  runtime::GeometryBuffer gradient_full =
      runtime::GeometryBuffer::CPU(geometry.pointCount());
  for (int i = 0; i < static_cast<int>(points.size()); ++i) {
    gradient_full.cpu()[static_cast<size_t>(points[static_cast<size_t>(i)])] =
        point_gradient_values[static_cast<size_t>(i)];
  }

  const double geometry_dot = dot(points, gradient_full.view(), dx.view());
  const double dof_dot =
      dot(runtime::ConstDofView::CPU(pulled_values.data(),
                                     static_cast<int>(pulled_values.size())),
          runtime::ConstDofView::CPU(dq_values.data(),
                                     static_cast<int>(dq_values.size())));

  EXPECT_NEAR(geometry_dot, dof_dot, 1.0e-12);
}

TEST(ABDContactMapping, DirectionUsesHomogeneousLocalPoint)
{
  ABDSubsystem subsystem(runtime::SubsystemIdx{0}, {makeMappingBody()});
  runtime::GlobalGeometryManager geometry;
  subsystem.declareGeometry(geometry);

  std::vector<double> dq_values(12, 0.0);
  dq_values[0] = 2.0;
  dq_values[4] = 3.0;
  dq_values[8] = -1.0;
  dq_values[9] = 5.0;

  runtime::GeometryBuffer dx = runtime::GeometryBuffer::CPU(geometry.pointCount());
  subsystem.mapLocalDirectionToGeometry(
      runtime::ConstDofView::CPU(dq_values.data(),
                                 static_cast<int>(dq_values.size())),
      dx.view());

  const runtime::PointIdx point = subsystem.geometryPointIds()[1];
  const glm::dvec3 expected =
      glm::dvec3{2.0, 0.0, 0.0} * 2.0 +
      glm::dvec3{0.0, 3.0, 0.0} * -0.25 +
      glm::dvec3{0.0, 0.0, -1.0} * 1.0 +
      glm::dvec3{5.0, 0.0, 0.0};

  EXPECT_DOUBLE_EQ(dx.cpu()[static_cast<size_t>(point)].x, expected.x);
  EXPECT_DOUBLE_EQ(dx.cpu()[static_cast<size_t>(point)].y, expected.y);
  EXPECT_DOUBLE_EQ(dx.cpu()[static_cast<size_t>(point)].z, expected.z);
}

}  // namespace
}  // namespace ksk::abd
