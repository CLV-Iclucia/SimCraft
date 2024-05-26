//
// Created by creeper on 5/24/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_TET_MESH_H_
#define SIMCRAFT_FEM_INCLUDE_TET_MESH_H_
#include <fem/types.h>
#include <filesystem>
#include <Core/properties.h>
namespace fem {
// this serves as a temporary holder of mesh info
// we will build them all into a system
struct TetMesh : core::NonCopyable {
  int num_vertices;
  int num_tets;
  TetMesh(int n_vertices, int n_tets) : num_vertices(n_vertices), num_tets(n_tets) {}
  Matrix<Real, 3, Dynamic> vertices;
  Matrix<int, 4, Dynamic> tets;
  std::vector<Vector<int, 3>> surface;
  void computeSurface();
};
std::unique_ptr<TetMesh> readTetMeshFromTobj(const std::filesystem::path &path, bool compute_surface = true);
}
#endif //SIMCRAFT_FEM_INCLUDE_TET_MESH_H_
