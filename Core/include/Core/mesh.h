//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_SURFACE_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_SURFACE_H_
#include <Core/core.h>
#include <string>
namespace sim::core {
struct Mesh {
  int triangleCount;
  std::vector<Vec3d> vertices;
  std::vector<Vec3d> normals;
  std::vector<uint> indices;
};

bool myLoadObj(const std::string& path, Mesh* mesh);
bool exportObj(const std::string& path, const Mesh& mesh);
}// namespace core
#endif // SIMCRAFT_CORE_INCLUDE_CORE_SURFACE_H_