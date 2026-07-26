#include <Runtime/global-geometry-manager.h>

#include <gtest/gtest.h>

#include <array>
#include <utility>
#include <vector>

namespace ksk::runtime {
namespace {

TEST(GeometryTable, PreservesPointOwnership)
{
  GlobalGeometryManager geometry;

  const auto p0 =
      geometry.addPoint(SubsystemIdx{4}, 10, glm::dvec3{1.0, 0.0, 0.0});
  const auto p1 =
      geometry.addPoint(SubsystemIdx{4}, 11, glm::dvec3{2.0, 0.0, 0.0});

  EXPECT_TRUE(geometry.contains(p0));
  EXPECT_TRUE(geometry.contains(p1));
}

TEST(GeometryTable, AppendsEdgesAndTrianglesWithStableIds)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.addPoint(SubsystemIdx{0}, 0, glm::dvec3{0.0});
  const auto p1 =
      geometry.addPoint(SubsystemIdx{0}, 1, glm::dvec3{1.0, 0.0, 0.0});
  const auto p2 =
      geometry.addPoint(SubsystemIdx{0}, 2, glm::dvec3{0.0, 1.0, 0.0});

  const int edge_id = geometry.addEdge(p0, p1, 0.05);
  const int triangle_id = geometry.addTriangle(p0, p1, p2);

  EXPECT_EQ(edge_id, 0);
  EXPECT_EQ(triangle_id, 0);
  EXPECT_EQ(geometry.edges[edge_id].p0, p0);
  EXPECT_DOUBLE_EQ(geometry.edges[edge_id].radius, 0.05);
  EXPECT_EQ(geometry.triangles[triangle_id].p2, p2);
  EXPECT_EQ(geometry.triangleRef(triangle_id).localIndex, 0);
}

TEST(GeometryTable, StoresTetsWithStableIdsAndRanges)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.addPoint(SubsystemIdx{0}, 0, glm::dvec3{0.0});
  const auto p1 =
      geometry.addPoint(SubsystemIdx{0}, 1, glm::dvec3{1.0, 0.0, 0.0});
  const auto p2 =
      geometry.addPoint(SubsystemIdx{0}, 2, glm::dvec3{0.0, 1.0, 0.0});
  const auto p3 =
      geometry.addPoint(SubsystemIdx{0}, 3, glm::dvec3{0.0, 0.0, 1.0});

  const int tet_id = geometry.addTet(p0, p1, p2, p3);

  EXPECT_EQ(tet_id, 0);
  EXPECT_EQ(geometry.tetCount(), 1);
  EXPECT_EQ(geometry.tetRange(SubsystemIdx{0}).first, 0);
  EXPECT_EQ(geometry.tetRange(SubsystemIdx{0}).count, 1);
  EXPECT_EQ(geometry.tetRef(tet_id).localIndex, 0);

  const auto tet_vertices = geometry.globalTet(tet_id);
  EXPECT_EQ(tet_vertices[0], p0);
  EXPECT_EQ(tet_vertices[3], p3);
}

TEST(GeometryTable, RejectsTetWithMixedOwners)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.addPoint(SubsystemIdx{0}, 0, glm::dvec3{0.0});
  const auto p1 =
      geometry.addPoint(SubsystemIdx{0}, 1, glm::dvec3{1.0, 0.0, 0.0});
  const auto p2 =
      geometry.addPoint(SubsystemIdx{0}, 2, glm::dvec3{0.0, 1.0, 0.0});
  const auto p3 =
      geometry.addPoint(SubsystemIdx{1}, 0, glm::dvec3{0.0, 0.0, 1.0});

  EXPECT_THROW([[maybe_unused]] const int tet = geometry.addTet(p0, p1, p2, p3),
               std::invalid_argument);
}

TEST(GeometryTable, TracksSubsystemRangesAndLocalPointMapping)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.addPoint(SubsystemIdx{0}, 7, glm::dvec3{0.0});
  const auto p1 = geometry.addPoint(SubsystemIdx{0}, 8, glm::dvec3{1.0});
  const auto p2 = geometry.addPoint(SubsystemIdx{1}, 0, glm::dvec3{2.0});

  EXPECT_EQ(geometry.pointCount(), 3);
  EXPECT_EQ(geometry.pointRange(SubsystemIdx{0}).first, 0);
  EXPECT_EQ(geometry.pointRange(SubsystemIdx{0}).count, 2);
  EXPECT_EQ(geometry.pointRange(SubsystemIdx{1}).first, 2);
  EXPECT_EQ(geometry.pointRange(SubsystemIdx{1}).count, 1);
  EXPECT_EQ(geometry.pointRef(p0).localIndex, 7);
}

TEST(GeometryTable, AnswersTopologyQueries)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.addPoint(SubsystemIdx{0}, 0, glm::dvec3{0.0});
  const auto p1 = geometry.addPoint(SubsystemIdx{0}, 1, glm::dvec3{1.0, 0.0, 0.0});
  const auto p2 = geometry.addPoint(SubsystemIdx{0}, 2, glm::dvec3{0.0, 1.0, 0.0});
  const int e0 = geometry.addEdge(p0, p1);
  const int e1 = geometry.addEdge(p1, p2);
  const int tri = geometry.addTriangle(p0, p1, p2);

  EXPECT_TRUE(geometry.edgesAdjacent(e0, e1));
  EXPECT_TRUE(geometry.triangleContainsPoint(tri, p2));
  EXPECT_FALSE(geometry.triangleContainsPoint(tri, PointIdx{99}));

  const auto edge_vertices = geometry.globalEdge(e0);
  EXPECT_EQ(edge_vertices[0], p0);
  EXPECT_EQ(edge_vertices[1], p1);
}

TEST(GeometryTable, ClassifiesConstraintOwnership)
{
  GlobalGeometryManager geometry;

  const auto a0 = geometry.addPoint(SubsystemIdx{0}, 0, glm::dvec3{0.0});
  const auto a1 = geometry.addPoint(SubsystemIdx{0}, 1, glm::dvec3{1.0});
  const auto b0 = geometry.addPoint(SubsystemIdx{1}, 0, glm::dvec3{2.0});
  const auto c0 = geometry.addColliderPoint(0, 0, glm::dvec3{3.0});

  const std::array same{a0, a1};
  const GeometryStencilInfo same_info = geometry.classify(same);
  EXPECT_TRUE(geometry.sameSubsystem(same));
  EXPECT_FALSE(same_info.crossesSubsystems);
  EXPECT_EQ(same_info.subsystemCount, 1);
  EXPECT_EQ(same_info.subsystems[0], SubsystemIdx{0});
  EXPECT_EQ(same_info.bodyCount, 1);
  EXPECT_EQ(same_info.bodies[0], geometry.pointOwner(a0).body);

  const std::array cross{a0, b0};
  const GeometryStencilInfo cross_info = geometry.classify(cross);
  EXPECT_FALSE(geometry.sameSubsystem(cross));
  EXPECT_TRUE(cross_info.crossesSubsystems);
  EXPECT_EQ(cross_info.subsystemCount, 2);
  EXPECT_EQ(cross_info.bodyCount, 2);

  const std::array collider{a0, c0};
  const GeometryStencilInfo collider_info = geometry.classify(collider);
  EXPECT_TRUE(collider_info.hasCollider);
  EXPECT_TRUE(geometry.hasCollider(collider));
  EXPECT_EQ(collider_info.bodyCount, 2);
}

TEST(GeometryTable, RejectsStencilCouplingMoreThanTwoSubsystems)
{
  GlobalGeometryManager geometry;

  const auto a0 = geometry.addPoint(SubsystemIdx{0}, 0, glm::dvec3{0.0});
  const auto b0 = geometry.addPoint(SubsystemIdx{1}, 0, glm::dvec3{1.0});
  const auto c0 = geometry.addPoint(SubsystemIdx{2}, 0, glm::dvec3{2.0});

  const std::array stencil{a0, b0, c0};
  const GeometryStencilInfo info = geometry.classify(stencil);

  EXPECT_FALSE(info.valid);
  EXPECT_TRUE(info.crossesSubsystems);
  EXPECT_EQ(info.subsystemCount, 2);
  EXPECT_EQ(info.subsystems[0], SubsystemIdx{0});
  EXPECT_EQ(info.subsystems[1], SubsystemIdx{1});
}

TEST(GeometryTable, ComputesStaticAndTrajectoryBounds)
{
  GlobalGeometryManager geometry;

  const auto p0 = geometry.addPoint(SubsystemIdx{0}, 0, glm::dvec3{0.0});
  const auto p1 =
      geometry.addPoint(SubsystemIdx{0}, 1, glm::dvec3{1.0, 2.0, 3.0});
  const int edge = geometry.addEdge(p0, p1);

  const GeometryBounds static_bounds = geometry.edgeBounds(edge);
  EXPECT_DOUBLE_EQ(static_bounds.lo.x, 0.0);
  EXPECT_DOUBLE_EQ(static_bounds.hi.z, 3.0);

  const std::vector<glm::dvec3> directions{
      glm::dvec3{0.0, -1.0, 0.0},
      glm::dvec3{2.0, 0.0, -4.0},
  };
  const GeometryBounds swept =
      geometry.trajectoryEdgeBounds(edge, directions, 0.5);
  EXPECT_DOUBLE_EQ(swept.lo.y, -0.5);
  EXPECT_DOUBLE_EQ(swept.hi.x, 2.0);
  EXPECT_DOUBLE_EQ(swept.lo.z, 0.0);
}

TEST(GeometryTable, StoresInstanceGeometryWithTransform)
{
  GlobalGeometryManager geometry;
  GeometryMeshDesc mesh;
  mesh.vertices = {
      glm::dvec3{0.0, 0.0, 0.0},
      glm::dvec3{1.0, 0.0, 0.0},
      glm::dvec3{0.0, 1.0, 0.0},
  };
  mesh.edges = {
      std::array<int, 2>{0, 1},
      std::array<int, 2>{1, 2},
  };
  mesh.triangles = {
      std::array<int, 3>{0, 1, 2},
  };

  glm::dmat4 transform{1.0};
  transform[3] = glm::dvec4{10.0, 0.0, 0.0, 1.0};
  const GeometryInstanceIdx instance =
      geometry.addInstance(SubsystemIdx{3}, 4, mesh, transform);

  EXPECT_EQ(geometry.pointRange(SubsystemIdx{3}).count, 3);
  EXPECT_EQ(geometry.edgeRange(SubsystemIdx{3}).count, 2);
  EXPECT_EQ(geometry.triangleRange(SubsystemIdx{3}).count, 1);
  EXPECT_EQ(geometry.instances[static_cast<size_t>(instance)].localInstanceId,
            4);

  glm::dmat4 moved{1.0};
  moved[3] = glm::dvec4{-2.0, 5.0, 0.0, 1.0};
  geometry.setInstanceTransform(instance, moved);
}

TEST(GeometryTable, CreatesDefaultGeometryBodiesForLegacyOwners)
{
  GlobalGeometryManager geometry;

  const auto point =
      geometry.addPoint(SubsystemIdx{2}, 5, glm::dvec3{1.0, 2.0, 3.0});
  const GeometryOwner owner = geometry.pointOwner(point);

  EXPECT_GE(owner.body, 0);
  EXPECT_TRUE(owner.isSubsystem());
  EXPECT_EQ(geometry.geometryBody(owner.body).type, EGeometryBody::Deformable);
  EXPECT_EQ(geometry.geometryBody(owner.body).subsystem, SubsystemIdx{2});
  EXPECT_EQ(geometry.bodyPointRange(owner.body).count, 1);
  EXPECT_TRUE(geometry.isDynamicBody(owner.body));

  const auto other_point =
      geometry.addPoint(SubsystemIdx{3}, 0, glm::dvec3{2.0, 0.0, 0.0});
  const GeometryOwner other_owner = geometry.pointOwner(other_point);
  EXPECT_TRUE(geometry.collisionPairEnabled(owner.body, other_owner.body));

  const auto collider_point =
      geometry.addColliderPoint(1, 0, glm::dvec3{0.0});
  const GeometryOwner collider_owner = geometry.pointOwner(collider_point);

  EXPECT_GE(collider_owner.body, 0);
  EXPECT_TRUE(collider_owner.isCollider());
  EXPECT_EQ(geometry.geometryBody(collider_owner.body).localBody, 1);
  EXPECT_TRUE(geometry.collisionPairEnabled(owner.body, collider_owner.body));
  EXPECT_FALSE(geometry.isDynamicBody(collider_owner.body));
}

}  // namespace
}  // namespace ksk::runtime
