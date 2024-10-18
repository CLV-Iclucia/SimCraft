//
// Created by creeper on 9/8/24.
//
#include <string>
#include <format>
#include <fem/tet-mesh.h>
#include <fem/system.h>
#include <fem/ipc/collision-detector.h>
#include <gtest/gtest.h>
using namespace fem;

int main() {
  auto mat = readTetMeshFromTOBJ(FEM_TETS_DIR "/mat100x100.tobj");
  int num_mat_vertices = mat->num_vertices;
  std::cout << std::format("mat has {} vertices, {} tets, {} surface triangles, {} surface edges\n",
                           mat->num_vertices, mat->num_tets, mat->surfaces.cols(), mat->surfaceEdges.cols());
  auto cube = readTetMeshFromTOBJ(FEM_TETS_DIR "/cube50x50.tobj");
  int num_cube_vertices = cube->num_vertices;
  std::cout << std::format("cube has {} vertices, {} tets, {} surface triangles, {} surface edges\n",
                           cube->num_vertices, cube->num_tets, cube->surfaces.cols(), cube->surfaceEdges.cols());
  for (int i = 0; i < num_cube_vertices; i++)
    cube->vertices.col(i) += Vector<Real, 3>(0.0, 0.0, 0.5);
  auto system = std::make_unique<System>();
  system->addPrimitive({
                           .mesh = std::move(cube),
                           .velocities = Matrix<Real, 3, Dynamic>::Zero(3, num_cube_vertices),
                           .energy = {},
                           .density = 0.0,
                       });
  system->addPrimitive({
                           .mesh = std::move(mat),
                           .velocities = Matrix<Real, 3, Dynamic>::Zero(3, num_mat_vertices),
                           .energy = {},
                           .density = 0.0,
                       });
  auto collisionDetector = std::make_unique<ipc::CollisionDetector>();
  VecXd p((num_mat_vertices + num_cube_vertices) * 3);
  p.setZero();
  for (int i = 0; i < num_cube_vertices; i++)
    p.segment<3>(3 * i) = Vector<Real, 3>(0, 0, -0.5);
  auto t = collisionDetector->detect(*system, p);
  if (!t) {
    std::cerr << "No contact" << std::endl;
    return 1;
  }
  std::cout << *t << std::endl;
}