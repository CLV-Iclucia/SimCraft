//
// Created by creeper on 5/27/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_DRAWABLE_MESH_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_DRAWABLE_MESH_H_
#include <cstdint>
#include <ogl-render/va-manager.h>
namespace opengl {
enum class MeshRenderOption : uint8_t {
  Wireframe,
  Fill
};
struct DrawableMesh {
  VertexArrayManager vertex_array_manager;
  std::optional<BufferObject<GL_ELEMENT_ARRAY_BUFFER>> ebo;
  void render(MeshRenderOption option) const {
    if (option == MeshRenderOption::Wireframe)
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    vertex_array_manager.bind();
    if (ebo.has_value()) {
      ebo.value().bind();
      glCheckError(glDrawElements(GL_TRIANGLES, ebo.value().size / sizeof(GLuint), GL_UNSIGNED_INT, nullptr));
    } else
      glCheckError(glDrawArrays(GL_TRIANGLES, 0, vertex_array_manager.vertex_count));
  }
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_DRAWABLE_MESH_H_
