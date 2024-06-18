//
// Created by creeper on 6/17/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_ATTRIBUTE_UTILS_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_ATTRIBUTE_UTILS_H_
#include <vector>
#include <string>
#include <unordered_map>
namespace opengl {
struct AttributeLayout {
  AttributeLayout() = default;
  explicit AttributeLayout(const std::unordered_map<GLuint, std::string> &index_attribute_map_) : location_attribute_map(
      index_attribute_map_) {
    for (const auto &[index, name] : location_attribute_map)
      attribute_location_map[name] = index;
  }
  std::array<uint8_t, 32> activeLocations() const {
    std::array<uint8_t, 32> active_locations{};
    int i = 0;
    for (const auto &[name, loc] : attribute_location_map)
      active_locations[i++] = loc;
    return active_locations;
  }
  std::optional<GLuint> location(const std::string &name) const {
    if (attribute_location_map.contains(name))
      return attribute_location_map.at(name);
    return std::nullopt;
  }
  std::optional<std::string_view> attribute(int loc) const {
    if (location_attribute_map.contains(loc))
      return location_attribute_map.at(loc);
    return std::nullopt;
  }
  size_t size() const {
    return attribute_location_map.size();
  }
 private:
  std::unordered_map<std::string, GLuint> attribute_location_map{};
  std::unordered_map<GLuint, std::string> location_attribute_map{};
};

struct AttributeDataSOA {

};

}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_ATTRIBUTE_UTILS_H_
