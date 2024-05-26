//
// Created by creeper on 5/25/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_SYSTEM_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_SYSTEM_H_
#include <fem/types.h>
#include <fem/tet-mesh.h>
#include <Maths/geometry.h>

namespace fem {
using maths::HalfPlane;
struct System {
  System(const std::vector<TetMesh>& tet_meshes) {
    auto num_vertices = 0, num_tets = 0, num_triangles = 0;
    for (const auto& tmesh: tet_meshes) {
      num_vertices += tmesh.num_vertices;
      num_tets += tmesh.num_tets;
      num_triangles += tmesh.surface.size();
    }
    x.resize(num_vertices * 3);
    xdot.resize(num_vertices * 3);
    X.resize(num_vertices * 3);
    for (const auto& tmesh: tet_meshes) {

    }
  }
  [[nodiscard]] int numTets() const {
    return static_cast<int>(tets.cols());
  }
  [[nodiscard]] int numTriangles() const {
    return static_cast<int>(surfaces.cols());
  }
  HalfPlane wall;
  VecXd x, xdot, X;
  Matrix<int, 4, Dynamic> tets;
  Matrix<int, 3, Dynamic> surfaces;
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_SYSTEM_H_
