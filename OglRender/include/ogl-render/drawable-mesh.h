//
// Created by creeper on 5/27/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_DRAWABLE_MESH_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_DRAWABLE_MESH_H_
#include <cstdint>
namespace opengl {
struct DrawableMesh {
  uint32_t num_triangles;
  uint32_t num_edges;
  uint32_t num_vertices;
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_DRAWABLE_MESH_H_
