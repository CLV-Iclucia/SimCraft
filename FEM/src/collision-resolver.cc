//
// Created by creeper on 5/25/24.
//
#include <fem/collision-resolver.h>
#include <fem/system.h>
namespace fem {
using spatify::parallel_for;

class TriangleIntersectionQuery : public spatify::SpatialQuery<TriangleIntersectionQuery> {
 public:
  explicit TriangleIntersectionQuery(int triangle_id_, const System* system_) : triangle_id(triangle_id_), system(system_) {
    const auto &t = system->surfaces;
    for (int i = 0; i < 3; i++) {
      Vector<Real, 3> current_pos = system->x.segment<3>(3 * t(i));
      Vector<Real, 3> candidate_pos = current_pos + system->xdot.segment<3>(3 * t(i));
      Real a = current_pos(0);
      Real b = current_pos(1);
      Real c = current_pos(2);
      Real d = candidate_pos(0);
      Real e = candidate_pos(1);
      Real f = candidate_pos(2);
      current_bbox.expand({a, b, c}).expand({d, e, f});
    }
  }
  bool query(const BBox<Real, 3> &bbox) const {
    return current_bbox.overlap(bbox);
  }
  bool query(int pr_id) override {
    if (pr_id <= triangle_id) return false;
    if (isAdjacentTriangle(triangle_id, pr_id)) return false;

    return true;
  }
  std::optional<CollisionInfo> collision_info{};
 private:
  int triangle_id{};
  const System* system;
  BBox<Real, 3> current_bbox;
  [[nodiscard]] bool isAdjacentTriangle(int i, int j) const {
    const auto &t = system->surfaces;
    for (int k = 0; k < 3; k++)
      if (t(k, i) == t(0, j) || t(k, i) == t(1, j) || t(k, i) == t(2, j))
        return true;
    return false;
  }
};
void CollisionDetector::detect(const System &system, Real dt) {
  const auto &triangles = system.surfaces;
  const auto &x = system.x;
  const auto &xdot = system.xdot;
  auto getBBox = [&](int id) -> BBox<Real, 3> {
    const auto &t = triangles.col(id);
    BBox<Real, 3> bbox;
    for (int i = 0; i < 3; i++) {
      Vector<Real, 3> current_pos = x.segment<3>(3 * t(i));
      Vector<Real, 3> candidate_pos = current_pos + xdot.segment<3>(3 * t(i));
      Real a = current_pos(0);
      Real b = current_pos(1);
      Real c = current_pos(2);
      Real d = candidate_pos(0);
      Real e = candidate_pos(1);
      Real f = candidate_pos(2);
      bbox.expand({a, b, c}).expand({d, e, f});
    }
    return bbox;
  };
  lbvh->update(system.numTriangles(), getBBox);
  parallel_for(0, system.numTriangles() - 1, [&](int i) {
    auto intersection_query = TriangleIntersectionQuery(i, &system);
    lbvh->runSpatialQuery(intersection_query);
    if (!intersection_query.collision_info)
      return ;
    collisions.push_back(*intersection_query.collision_info);
  });
}

std::optional<CollisionInfo> eeCCD(const EdgeEdgeCCDQuery &query, Real dt) {

}

std::optional<CollisionInfo> vtCCD(const VertexTriangleCCDQuery &query, Real dt) {

}
}