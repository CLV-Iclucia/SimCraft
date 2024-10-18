//
// Created by creeper on 5/27/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_DRAWABLE_MESH_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_DRAWABLE_MESH_H_

#include <cstdint>
#include <ogl-render/vertex-array.h>

namespace opengl {
enum class MeshRenderOption : uint8_t {
  Wireframe,
  Fill
};

struct DrawableMesh {
  explicit DrawableMesh(const AttributeTable &attributeLayout)
      : vertexArray(attributeLayout) {}
  void render(MeshRenderOption option = MeshRenderOption::Fill) const {
    vertexArray.bind();
    if (option == MeshRenderOption::Wireframe)
      glCheckError(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
    else
      glCheckError(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
    if (vertexArray.useElementData())
      glCheckError(glDrawElements(GL_TRIANGLES, vertexArray.elementCount(), GL_UNSIGNED_INT, nullptr));
    else
      glCheckError(glDrawArrays(GL_TRIANGLES, 0, vertexArray.vertexCount()));
  }
  VertexArray vertexArray;
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_DRAWABLE_MESH_H_
