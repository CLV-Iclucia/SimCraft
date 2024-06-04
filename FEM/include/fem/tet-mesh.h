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
struct TetMeshTopology : core::NonCopyable {
  std::vector<TetrahedronTopology> tets{};
  std::vector<Vector<int, 3>> surface{};
  TetMeshTopology() = default;
  TetMeshTopology(TetMeshTopology &&other) noexcept: tets(std::move(other.tets)), surface(std::move(other.surface)) {}
};
struct TetMesh : core::NonCopyable {
  int num_vertices;
  int num_tets;
  TetMesh(int n_vertices, int n_tets) : num_vertices(n_vertices), num_tets(n_tets) {}
  Matrix<Real, 3, Dynamic> vertices;
  TetMeshTopology tets{};
  void computeSurface();
};
std::unique_ptr<TetMesh> readTetMeshFromTobj(const std::filesystem::path &path, bool compute_surface = true);
}
#endif //SIMCRAFT_FEM_INCLUDE_TET_MESH_H_
