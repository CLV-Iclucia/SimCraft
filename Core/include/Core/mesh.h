//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_SURFACE_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_SURFACE_H_
#include <Core/core.h>
#include <string>
namespace core {
struct Mesh {
  int triangleCount;
  std::vector<Vec3d> vertices;
  std::vector<Vec3d> normals;
  std::vector<uint> indices;
};

bool myLoadObj(const std::string& path, Mesh* mesh);
}
#endif // SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_SDF_H_