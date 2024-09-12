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

struct AttributeConfig {
  const std::string &attributeName;
  int componentsPerAttribute;
  std::span<const std::byte> data;
};

struct VertexArray : Resource {
  const VertexArrayObject vao{};

  explicit VertexArray(const AttributeTable &attribute_layout_, std::span<const GLuint> indices = {})
      : attributeLayout(attribute_layout_),
        vertexBuffers(attribute_layout_.size()),
        location_index_map(attribute_layout_.activeLocations()) {
    if (!indices.empty())
      configElementData(std::as_bytes(indices));
  }

  void bind() const {
    vao.bind();
  }

  template<GLenum Usage, typename ComponentType> requires std::is_pod_v<ComponentType>
  void configAttributeData(const AttributeConfig &option) {
    const auto &[attribute_name, components_per_attribute, data] = option;
    auto loc = attributeLayout.location(attribute_name);
    if (!loc.has_value()) {
      std::cerr << "Attribute " << attribute_name << " not found in the shader program" << std::endl;
      return;
    }
    auto loc_value = loc.value();
    int index = static_cast<int>(location_index_map[loc_value]);
    vao.bind();
    vertexBuffers[index].bind();
    vertexBuffers[index].bufferData<Usage>(data);
    VertexArrayObject::vertexAttribPointer(
        {.index = loc_value,
            .components_per_attribute = components_per_attribute,
            .component_type = glType<ComponentType>(),
            .stride_in_bytes = 0});
    VertexArrayObject::enableVertexAttribArray(loc_value);
  }

  template<GLenum Usage, typename AttributeType, typename ComponentType>
  requires std::is_pod_v<ComponentType> && (sizeof(AttributeType) % sizeof(ComponentType) == 0)
      && std::is_same_v<typename AttributeType::value_type, ComponentType>
  void configAttributeData(const std::string &attribute_name, std::span<const std::byte> data) {
    constexpr int NumComponentPerAttribute = sizeof(AttributeType) / sizeof(ComponentType);
    configAttributeData<Usage, ComponentType>({attribute_name, NumComponentPerAttribute, data});
  }

  void configElementData(std::span<const std::byte> data) {
    vao.bind();
    if (!ebo.has_value())
      ebo.emplace();
    ebo->bind();
    ebo->bufferData<GL_STATIC_DRAW>(data);
    m_elementCount = data.size_bytes() / sizeof(GLuint);
  }
  [[nodiscard]] bool useElementData() const {
    return ebo.has_value();
  }
  [[nodiscard]] GLuint vertexCount() const {
    return m_vertexCount;
  }
  [[nodiscard]] GLuint elementCount() const {
    return m_elementCount;
  }
private:
  const AttributeTable &attributeLayout;
  std::array<uint8_t, 32> location_index_map;
  std::vector<VertexBufferObj> vertexBuffers{};
  std::optional<ElementBufferObj> ebo{};
  GLuint m_elementCount{};
  GLuint m_vertexCount{};
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_VERTEX_DATA_MANAGER_H_
