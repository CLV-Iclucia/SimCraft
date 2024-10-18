#include <fem/tet-mesh.h>
#include <iostream>
#include <fstream>
#include <set>
namespace fem {
std::unique_ptr<TetMesh> readTetMeshFromTOBJ(const std::filesystem::path &path, bool compute_surface) {
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
      vertices.emplace_back(x, y, z);
    } else if (prefix == "t") {
      int x, y, z, w;
      stream >> x >> y >> z >> w;
      indices.emplace_back(x - 1, y - 1, z - 1, w - 1);
    }
  }
  file.close();
  auto nv = static_cast<int>(vertices.size());
  auto nt = static_cast<int>(indices.size());
  auto tmesh = std::make_unique<TetMesh>(nv, nt);
  tmesh->vertices.resize(3, nv);
  tmesh->tets.resize(4, nt);
  for (int i = 0; i < nv; i++)
    tmesh->vertices.col(i) = vertices[i];
  for (int i = 0; i < nt; i++)
    tmesh->tets.col(i) = indices[i];
  if (compute_surface) {
    tmesh->computeSurface();
    tmesh->computeSurfaceEdges();
  }
  return tmesh;
}

static void resolveInternalFace(std::set<std::array<int, 3>> &face_set, const std::array<int, 3> &face) {
  auto sortedFace = face;
  std::sort(sortedFace.begin(), sortedFace.end());
  if (face_set.contains(sortedFace))
    face_set.erase(sortedFace);
  else
    face_set.insert(sortedFace);
}

void TetMesh::computeSurface() {
  std::set<std::array<int, 3>> face_set{};
  for (int i = 0; i < num_tets; i++) {
    auto tet = tets.col(i);
    resolveInternalFace(face_set, {tet(0), tet(1), tet(2)});
    resolveInternalFace(face_set, {tet(0), tet(1), tet(3)});
    resolveInternalFace(face_set, {tet(0), tet(2), tet(3)});
    resolveInternalFace(face_set, {tet(1), tet(2), tet(3)});
  }
  surfaces.resize(Eigen::NoChange, static_cast<int>(face_set.size()));
  int i = 0;
  for (const auto &array : face_set) {
    surfaces.col(i) = Vector<int, 3>(array[0], array[1], array[2]);
    i++;
  }
}

void TetMesh::computeSurfaceEdges() {
  std::set<std::array<int, 2>> edge_set{};
  for (int i = 0; i < surfaces.cols(); i++) {
    auto surface = surfaces.col(i);
    if (surface(0) < surface(1) && !edge_set.contains({surface(0), surface(1)}))
      edge_set.insert({surface(0), surface(1)});
    if (surface(1) < surface(2) && !edge_set.contains({surface(1), surface(2)}))
      edge_set.insert({surface(1), surface(2)});
    if (surface(2) < surface(0) && !edge_set.contains({surface(2), surface(0)}))
      edge_set.insert({surface(2), surface(0)});
  }
  surfaceEdges.resize(Eigen::NoChange, static_cast<int>(edge_set.size()));
  int i = 0;
  for (const auto &[a, b] : edge_set) {
    surfaceEdges.col(i) = Vector<int, 2>(a, b);
    i++;
  }
}
}