//
// Created by creeper on 5/24/24.
//

#pragma once
#include <Core/deserializer.h>
#include <fem/simplex.h>
#include <fem/types.h>
#include <filesystem>
#include <optional>
#include <span>

namespace sim::fem {
struct TetMesh {
  TetMesh() = default;
  TetMesh(const std::vector<Vector<Real, 3>> &vertices,
          const std::vector<Tetrahedron> &tets,
          const std::vector<Vector<Real, 3>> &velocities = {})
      : vertices(vertices), tets(tets) {
    computeSurface();
    computeSurfaceEdges();
  }
  std::vector<Tetrahedron> tets{};
  std::vector<Triangle> surfaces{};
  std::vector<Edge> surfaceEdges{};
  static TetMesh static_deserialize(const core::JsonNode &json);
  [[nodiscard]] std::span<const Triangle> surfaceView() const {
    return std::span{surfaces};
  }
  [[nodiscard]] std::span<const Edge> surfaceEdgeView() const {
    return std::span{surfaceEdges};
  }
  void commit(SubVector<Real> x, SubVector<Real> xdot, SubVector<Real> X) {
    for (int i = 0; i < vertices.size(); i++) {
      auto &v = vertices[i];
      x.segment<3>(i * 3) = v;
      X.segment<3>(i * 3) = v;

      if (!velocities.empty()) {
        auto &vel = velocities[i];
        xdot.segment<3>(i * 3) = vel;
      } else
        xdot.segment<3>(i * 3) = Vector<Real, 3>::Zero();
    }
    vertices.clear();
    transitionToCommitted();
  }
  const std::vector<Vector<Real, 3>> &getVertices() const {
    if (committed)
      throw std::runtime_error("Mesh vertices are committed and cannot be accessed");
    return vertices;
  }

private:
  // these two will be commited and cleared so they cannot be accessed after commit
  std::vector<Vector<Real, 3>> vertices{};
  std::vector<Vector<Real, 3>> velocities{};
  bool committed{false};
  void transitionToCommitted() {
    committed = true;
  }
  void computeSurface();
  void computeSurfaceEdges();
};

std::optional<TetMesh> readTetMeshFromTOBJ(const std::filesystem::path &path);
} // namespace sim::fem