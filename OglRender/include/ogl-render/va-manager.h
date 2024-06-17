//
// Created by creeper on 6/17/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_VA_MANAGER_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_VA_MANAGER_H_
#include <ogl-render/resource-handles.h>
#include <ogl-render/properties.h>
#include <ogl-render/shader-prog.h>
#include <ogl-render/gl-utils.h>
namespace opengl {
struct VertexArrayManager : Resource {
  const VertexArrayObject vao{};
  void bind() const {
    vao.bind();
  }
  int vertex_count{};
  void registerAttributeLayout(const AttributeLayout& layout) {
    attribute_layout = layout;
  }
  template <GLenum Usage, typename T>
  void initVertexAttribute(const std::string& attribute_name, std::span<const T> data) const {
    auto loc = attribute_layout.location(attribute_name);
    if (!loc.has_value()) {
      // TODO: ERROR
    }
    auto loc_value = loc.value();
    vertex_buffers[loc_value].bind();
    glCheckError(vertex_buffers[loc_value].bufferData<Usage>(data));
    VertexArrayObject::enableVertexAttribArray(loc_value);
  }
 private:
  AttributeLayout attribute_layout{};
  std::vector<BufferObject<GL_ARRAY_BUFFER>> vertex_buffers{};
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_VA_MANAGER_H_
