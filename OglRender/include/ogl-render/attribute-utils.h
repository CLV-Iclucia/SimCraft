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
  std::optional<int> location(const std::string& name) const {
    if (attribute_index_map.contains(name))
      return attribute_index_map.at(name);
    return std::nullopt;
  }
  std::optional<std::string_view> attribute(int loc) const {
    return attribute_names[loc];
  }
  size_t size() const {
    return attribute_names.size();
  }
 private:
  std::vector<std::string> attribute_names{};
  std::unordered_map<std::string, int> attribute_index_map{};
};

struct AttributeDataSOA {

};

}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_ATTRIBUTE_UTILS_H_
