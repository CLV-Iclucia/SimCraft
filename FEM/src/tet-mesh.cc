#include <fem/tet-mesh.h>
#include <iostream>
#include <fstream>
#include <set>
namespace fem {
std::unique_ptr<TetMesh> readTetMeshFromTobj(std::filesystem::path path, bool compute_surface = true) {
  // open file
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "[ERROR] Failed to load " << path << std::endl;
    return nullptr;
  }
  std::string line;
  std::vector<Vector<Real, 3>> vertices;
  std::vector<Vector<int, 4>> indices;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string prefix;
    stream >> prefix;
    if (prefix == "v") {
      Real x, y, z;
      stream >> x >> y >> z;
      vertices.push_back(Vector<Real, 3>(x, y, z));
    } else if (prefix == "f") {
      int x, y, z, w;
      stream >> x >> y >> z >> w;
      indices.push_back(Vector<int, 4>(x, y, z, w));
    }
  }
  file.close();
  auto nv = static_cast<int>(vertices.size());
  auto nt = static_cast<int>(indices.size());
  auto tmesh = std::make_unique<TetMesh>(nv, nt);
  if (compute_surface)
    tmesh->computeSurface();
  return tmesh;
}

static void resolveInternalFace(std::set<std::tuple<int, int, int>> &face_set, const std::tuple<int, int, int> &face) {
  if (face_set.contains(face))
    face_set.erase(face);
  else face_set.insert(face);
}

void TetMesh::computeSurface() {
  std::set<std::tuple<int, int, int>> face_set{};
  for (int i = 0; i < num_tets; i++) {
    auto tet = tets.col(i);
    // sort the indices in tet
    std::sort(&tet(0), &tet(0) + 4);
    resolveInternalFace(face_set, {tet(0), tet(1), tet(2)});
    resolveInternalFace(face_set, {tet(0), tet(1), tet(3)});
    resolveInternalFace(face_set, {tet(0), tet(2), tet(3)});
    resolveInternalFace(face_set, {tet(1), tet(2), tet(3)});
  }
  surface.reserve(face_set.size());
  for (const auto& [a, b, c] : face_set)
    surface.emplace_back(a, b, c);
}
}