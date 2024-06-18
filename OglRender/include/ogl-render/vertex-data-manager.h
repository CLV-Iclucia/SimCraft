//
// Created by creeper on 6/17/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_VERTEX_DATA_MANAGER_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_VERTEX_DATA_MANAGER_H_
#include <ogl-render/resource-handles.h>
#include <ogl-render/properties.h>
#include <ogl-render/shader-prog.h>
#include <ogl-render/gl-utils.h>
namespace opengl {
struct VertexDataManager : Resource {
  const VertexArrayObject vao{};
  explicit VertexDataManager(const AttributeLayout &attribute_layout_)
      : attribute_layout(attribute_layout_),
        vertex_buffers(attribute_layout_.size()),
        location_index_map(attribute_layout_.activeLocations()) {
  }
  void bind() const {
    vao.bind();
  }
  int vertex_count{};

  template<GLenum Usage, int NumComponentPerAttribute, typename ComponentType>
  void initAttributeData(const std::string &attribute_name, std::span<const ComponentType> data) {
    auto loc = attribute_layout.location(attribute_name);
    if (!loc.has_value()) {
      // TODO: ERROR
      std::cerr << "Attribute not found\n";
      return;
    }
    auto loc_value = loc.value();
    int index = static_cast<int>(location_index_map[loc_value]);
    vao.bind();
    vertex_buffers[index].bind();
    vertex_buffers[index].bufferData<Usage, ComponentType>(data);
    VertexAttribPointerOption option{.location = loc_value,
        .components_per_attribute = NumComponentPerAttribute,
        .component_type = glType<ComponentType>(),
        .stride_in_bytes = 0};
    VertexArrayObject::vertexAttribPointer(option);
    VertexArrayObject::enableVertexAttribArray(loc_value);
  }

  template<GLenum Usage, typename Attribute>
  requires std::is_pod_v<typename Attribute::value_type>
  void initAttributeData(const std::string &attribute_name, std::span<const Attribute> data) {
    using ComponentType = typename Attribute::value_type;
    initAttributeData<Usage, Attribute, ComponentType>(attribute_name, data);
  }

 private:
  const AttributeLayout &attribute_layout;
  std::array<uint8_t, 32> location_index_map;
  std::vector<BufferObject<GL_ARRAY_BUFFER>> vertex_buffers{};
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_VERTEX_DATA_MANAGER_H_
