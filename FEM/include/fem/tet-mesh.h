//
// Created by creeper on 5/24/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_TET_MESH_H_
#define SIMCRAFT_FEM_INCLUDE_TET_MESH_H_
#include <fem/types.h>
#include <filesystem>
#include <Core/properties.h>
namespace fem {
struct System;
using TetrahedronTopology = Vector<int, 4>;
struct TetMesh : core::NonCopyable {
  TetMesh() = default;
  int num_vertices{};
  int num_tets{};
  TetMesh(int n_vertices, int n_tets) : num_vertices(n_vertices), num_tets(n_tets) {}
  Matrix<Real, 3, Dynamic> vertices{};
  Matrix<int, 4, Dynamic> tets{};
  Matrix<int, 3, Dynamic> surfaces{};
  Matrix<int, 2, Dynamic> surfaceEdges{};
  void computeSurface();
  void computeSurfaceEdges();
};
std::unique_ptr<TetMesh> readTetMeshFromTOBJ(const std::filesystem::path &path, bool compute_surface = true);
}
#endif //SIMCRAFT_FEM_INCLUDE_TET_MESH_H_
