//
// Created by creeper on 6/17/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_ATTRIBUTE_UTILS_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_ATTRIBUTE_UTILS_H_
#include <ogl-render/ogl-types.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <array>
namespace opengl {

struct ShaderVariable {
  OglTypeInfo typeInfo;
  std::string name;
};

struct AttributeTable {
  AttributeTable() = default;
  explicit AttributeTable(const std::unordered_map<GLuint, ShaderVariable> &index_attribute_map_) : index_attribute_map(
      index_attribute_map_) {
    for (const auto &[index, attrib] : index_attribute_map)
      name_index_map[attrib.name] = index;
  }
  std::array<uint8_t, 32> activeLocations() const {
    std::array<uint8_t, 32> active_locations{};
    int i = 0;
    for (const auto &[name, loc] : name_index_map)
      active_locations[loc] = i++;
    return active_locations;
  }
  std::optional<GLuint> location(const std::string &name) const {
    if (name_index_map.contains(name))
      return name_index_map.at(name);
    return std::nullopt;
  }
  size_t size() const {
    return name_index_map.size();
  }
 private:
  std::unordered_map<std::string, GLuint> name_index_map{};
  std::unordered_map<GLuint, ShaderVariable> index_attribute_map{};
};

struct AttributeDataSOA {

};

}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_ATTRIBUTE_UTILS_H_
