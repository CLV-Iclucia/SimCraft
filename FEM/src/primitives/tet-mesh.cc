#include <fem/primitives/tet-mesh.h>
#include <fstream>
#include <iostream>
#include <set>

namespace sim::fem {
std::optional<TetMesh> readTetMeshFromTOBJ(const std::filesystem::path &path) {
  // open file
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "[ERROR] Failed to load " << path << std::endl;
    return std::nullopt;
  }
  std::string line;
  std::vector<Vector<Real, 3>> vertices;
  std::vector<Tetrahedron> indices;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string prefix;
    stream >> prefix;
    if (prefix == "v") {
      Real x, y, z;
      stream >> x >> y >> z;
      vertices.emplace_back(x, y, z);
    } else if (prefix == "t") {
      int x, y, z, w;
      stream >> x >> y >> z >> w;
      indices.emplace_back(x - 1, y - 1, z - 1, w - 1);
    }
  }
  file.close();
  return std::make_optional<TetMesh>(vertices, indices);
}

TetMesh TetMesh::static_deserialize(const core::JsonNode &json) {
  if (!json.is<core::JsonDict>())
    throw std::runtime_error("Expected a dictionary");
  const auto &dict = json.as<core::JsonDict>();

  if (dict.contains("path")) {
    const auto &mesh =
        readTetMeshFromTOBJ(dict.at("path").as<std::string>());
    if (!mesh)
      throw std::runtime_error("Failed to load mesh");
    return *mesh;
  }

  if (dict.contains("vertices") && dict.contains("tets")) {
    if (dict.contains("velocities"))
      return TetMesh{
          core::deserialize<std::vector<Vector<Real, 3>>>(dict.at("vertices")),
          core::deserialize<std::vector<Tetrahedron>>(dict.at("tets")),
          core::deserialize<std::vector<Vector<Real, 3>>>(dict.at("velocities"))};
    return TetMesh{
        core::deserialize<std::vector<Vector<Real, 3>>>(dict.at("vertices")),
        core::deserialize<std::vector<Tetrahedron>>(dict.at("tets"))};
  }

  throw std::runtime_error("Expected either a path or vertices and tets");
}

void TetMesh::computeSurface() {
  struct TriangleCmp {
    bool operator()(const Triangle &a, const Triangle &b) const {
      if (a.x != b.x)
        return a.x < b.x;
      if (a.y != b.y)
        return a.y < b.y;
      return a.z < b.z;
    }
  };
  std::set<Triangle, TriangleCmp> face_set{};

  auto resolveInternalFace = [&](const Triangle &face) {
    auto sortedFace = [](Triangle unsortedFace) {
    std::array<int, 3> sorted = {unsortedFace[0], unsortedFace[1],
                                 unsortedFace[2]};
    std::ranges::sort(sorted);
    return Triangle{sorted[0], sorted[1], sorted[2]};
  }(face);
  if (face_set.contains(sortedFace))
    face_set.erase(sortedFace);
  else
    face_set.insert(sortedFace);
  };

  for (const auto &tet : tets) {
    resolveInternalFace({tet[0], tet[1], tet[2]});
    resolveInternalFace({tet[0], tet[1], tet[3]});
    resolveInternalFace({tet[0], tet[2], tet[3]});
    resolveInternalFace({tet[1], tet[2], tet[3]});
  }
  surfaces.resize(static_cast<int>(face_set.size()));
  int i = 0;
  for (const auto &array : face_set) {
    surfaces[i] = {array[0], array[1], array[2]};
    i++;
  }
}

void TetMesh::computeSurfaceEdges() {
  struct EdgeCmp {
    bool operator()(const Edge &a, const Edge &b) const {
      if (a.x != b.x)
        return a.x < b.x;
      return a.y < b.y;
    }
  };
  std::set<Edge, EdgeCmp> edge_set{};
  for (auto surface : surfaces) {
    if (surface[0] < surface[1] && !edge_set.contains({surface[0], surface[1]}))
      edge_set.insert({surface[0], surface[1]});
    if (surface[1] < surface[2] && !edge_set.contains({surface[1], surface[2]}))
      edge_set.insert({surface[1], surface[2]});
    if (surface[2] < surface[0] && !edge_set.contains({surface[2], surface[0]}))
      edge_set.insert({surface[2], surface[0]});
  }
  surfaceEdges.resize(static_cast<int>(edge_set.size()));
  int i = 0;
  for (const auto &edge : edge_set) {
    surfaceEdges[i] = {edge.x, edge.y};
    i++;
  }
}
} // namespace fem
