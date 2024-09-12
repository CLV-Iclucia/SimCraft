#ifndef OGL_RENDER_INCLUDE_OGL_RENDER_SHADER_PROG_H_
#define OGL_RENDER_INCLUDE_OGL_RENDER_SHADER_PROG_H_

#include <glad/glad.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <unordered_map>
#include <glm/glm.hpp>
#include <vector>
#include <filesystem>
#include <format>
#include <ogl-render/resource-handles.h>
#include <ogl-render/attribute-utils.h>
#include <ogl-render/gl-utils.h>

namespace opengl {

struct ShaderProgramConfig {
  std::filesystem::path vertexShaderPath{};
  std::filesystem::path fragmentShaderPath{};
  std::filesystem::path geometry_shader_path{};
};

// RAII shader program
struct ShaderProgram : GLHandleBase<ShaderProgram> {
  using GLHandleBase<ShaderProgram>::id;

  // create shader program from vertex and fragment shader source
  explicit ShaderProgram(const ShaderProgramConfig &config);

  void create() {
    glCheckError(id = glCreateProgram());
  }

  void destroy() {
    glCheckError(glDeleteProgram(id));
  }

  void bind() const override {
    glCheckError(glUseProgram(id));
  }

  static void unbind() {
    glCheckError(glUseProgram(0));
  }

  ShaderProgram(ShaderProgram &&other) noexcept
      : uniform_count(other.uniform_count), attribute_count(other.attribute_count) {
    uniform_handles = std::move(other.uniform_handles);
    attribute_table = other.attribute_table;
    other.uniform_count = 0;
    other.attribute_count = 0;
  }

  ShaderProgram &operator=(ShaderProgram &&other) noexcept {
    if (this != &other) {
      uniform_handles = std::move(other.uniform_handles);
      attribute_table = other.attribute_table;
      uniform_count = other.uniform_count;
      attribute_count = other.attribute_count;
      other.uniform_count = 0;
      other.attribute_count = 0;
    }
    return *this;
  }

  std::unordered_map<std::string, GLint> uniform_handles;
  std::unordered_map<std::string, OglTypeInfo> uniform_type_info;
  AttributeTable attribute_table;

  const AttributeTable &attributeTable() const {
    return attribute_table;
  }

  void use() const { bind(); }

  void setUniform(const std::string &name, int value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Int, .component_count = 1, .array_size = 1}));
    glCheckError(glUniform1i(uniform_handles[name], value));
  }

  void setUniform(const std::string &name, int x, int y) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Int, .component_count = 2, .array_size = 1}));
    glCheckError(glUniform2i(uniform_handles[name], x, y));
  }

  void setUniform(const std::string &name, int x, int y, int z) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Int, .component_count = 3, .array_size = 1}));
    glCheckError(glUniform3i(uniform_handles[name], x, y, z));
  }

  void setUniform(const std::string &name, int x, int y, int z, int w) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Int, .component_count = 4, .array_size = 1}));
    glCheckError(glUniform4i(uniform_handles[name], x, y, z, w));
  }

  void setUniform(const std::string &name, float value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 1, .array_size = 1}));
    glCheckError(glUniform1f(uniform_handles[name], value));
  }

  void setUniform(const std::string &name, float x, float y) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 2, .array_size = 1}));
    glCheckError(glUniform2f(uniform_handles[name], x, y));
  }

  void setUniform(const std::string &name, const glm::vec2 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 2, .array_size = 1}));
    glCheckError(glUniform2f(uniform_handles[name], value.x, value.y));
  }

  void setUniform(const std::string &name, float x, float y, float z) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 3, .array_size = 1}));
    glCheckError(glUniform3f(uniform_handles[name], x, y, z));
  }

  void setUniform(const std::string &name, const glm::vec3 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 3, .array_size = 1}));
    glCheckError(glUniform3f(uniform_handles[name], value.x, value.y, value.z));
  }

  void setUniform(const std::string &name, float x, float y, float z, float w) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 4, .array_size = 1}));
    glCheckError(glUniform4f(uniform_handles[name], x, y, z, w));
  }

  void setUniform(const std::string &name, const glm::vec4 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 4, .array_size = 1}));
    glCheckError(glUniform4f(uniform_handles[name], value.x, value.y, value.z, value.w));
  }

  void setUniform(const std::string &name, const glm::mat2 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 4, .array_size = 1}));
    glCheckError(glUniformMatrix2fv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::mat3 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 9, .array_size = 1}));
    glCheckError(glUniformMatrix3fv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::mat4 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 16, .array_size = 1}));
    glCheckError(glUniformMatrix4fv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::mat2x3 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 6, .array_size = 1}));
    glCheckError(glUniformMatrix2x3fv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::mat3x2 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 6, .array_size = 1}));
    glCheckError(glUniformMatrix3x2fv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::mat2x4 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 8, .array_size = 1}));
    glCheckError(glUniformMatrix2x4fv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::mat4x2 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 8, .array_size = 1}));
    glCheckError(glUniformMatrix4x2fv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::mat3x4 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 12, .array_size = 1}));
    glCheckError(glUniformMatrix3x4fv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::mat4x3 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Float, .component_count = 12, .array_size = 1}));
    glCheckError(glUniformMatrix4x3fv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  // double matrix
  void setUniform(const std::string &name, const glm::dmat2 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 4, .array_size = 1}));
    glCheckError(glUniformMatrix2dv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::dmat3 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 9, .array_size = 1}));
    glCheckError(glUniformMatrix3dv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::dmat4 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 16, .array_size = 1}));
    glCheckError(glUniformMatrix4dv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::dmat2x3 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 6, .array_size = 1}));
    glCheckError(glUniformMatrix2x3dv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::dmat3x2 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 6, .array_size = 1}));
    glCheckError(glUniformMatrix3x2dv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::dmat2x4 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 8, .array_size = 1}));
    glCheckError(glUniformMatrix2x4dv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::dmat4x2 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 8, .array_size = 1}));
    glCheckError(glUniformMatrix4x2dv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::dmat3x4 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 12, .array_size = 1}));
    glCheckError(glUniformMatrix3x4dv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::dmat4x3 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 12, .array_size = 1}));
    glCheckError(glUniformMatrix4x3dv(uniform_handles[name], 1, GL_FALSE, &value[0][0]));
  }

  void setUniform(const std::string &name, const glm::dvec2 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 2, .array_size = 1}));
    glCheckError(glUniform2dv(uniform_handles[name], 1, &value[0]));
  }

  void setUniform(const std::string &name, const glm::dvec3 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 3, .array_size = 1}));
    glCheckError(glUniform3dv(uniform_handles[name], 1, &value[0]));
  }

  void setUniform(const std::string &name, const glm::dvec4 &value) {
    auto type_info = uniform_type_info[name];
    assert((type_info == OglTypeInfo{.type = OglBasicType::Double, .component_count = 4, .array_size = 1}));
    glCheckError(glUniform4dv(uniform_handles[name], 1, &value[0]));
  }

 private:
  void initActiveAttributeTable();

  int uniform_count{};
  int attribute_count{};

  void initUniformVariables();

  static void checkCompileErrors(GLuint shader, const std::string &type);
};
}  // namespace core

#endif