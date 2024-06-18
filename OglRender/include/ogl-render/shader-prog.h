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
  std::filesystem::path vertex_shader_path{};
  std::filesystem::path fragment_shader_path{};
  std::filesystem::path geometry_shader_path{};
};

// RAII shader program
struct ShaderProgram : GLHandleBase<ShaderProgram> {
  using GLHandleBase<ShaderProgram>::id;
  // create shader program from vertex and fragment shader source
  explicit ShaderProgram(const ShaderProgramConfig& config);
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
    attribute_layout = other.attribute_layout;
    other.uniform_count = 0;
    other.attribute_count = 0;
  }
  ShaderProgram &operator=(ShaderProgram &&other) noexcept {
    if (this != &other) {
      uniform_handles = std::move(other.uniform_handles);
      attribute_layout = other.attribute_layout;
      uniform_count = other.uniform_count;
      attribute_count = other.attribute_count;
      other.uniform_count = 0;
      other.attribute_count = 0;
    }
    return *this;
  }
  std::unordered_map<std::string, GLint> uniform_handles;
  AttributeLayout attribute_layout;
  const AttributeLayout &attributeLayout() const {
    return attribute_layout;
  }


  void use() const { bind(); }
  void setInt(const std::string &name, int value) {
    glCheckError(glUniform1i(uniform_handles[name], value));
  }
  void setFloat(const std::string &name, float value) {
    glCheckError(glUniform1f(uniform_handles[name], value));
  }
  void setVec2f(const std::string &name, float x, float y) {
    glCheckError(glUniform2f(uniform_handles[name], x, y));
  }
  void setVec2f(const std::string &name, const glm::vec2 &value) {
    glCheckError(glUniform2f(uniform_handles[name], value.x, value.y));
  }
  void setVec3f(const std::string &name, float x, float y, float z) {
    glCheckError(glUniform3f(uniform_handles[name], x, y, z));
  }
  void setVec3f(const std::string &name, const glm::vec3 &value) {
    glCheckError(glUniform3f(uniform_handles[name], value.x, value.y, value.z));
  }
  void setVec4f(const std::string &name, float x, float y, float z, float w) {
    glCheckError(glUniform4f(uniform_handles[name], x, y, z, w));
  }
  void setVec4f(const std::string &name, const glm::vec4 &value) {
    glCheckError(glUniform4f(uniform_handles[name], value.x, value.y, value.z, value.w));
  }
  void setMat2f(const std::string &name, const glm::mat2 &mat) {
    glCheckError(glUniformMatrix2fv(uniform_handles[name], 1, GL_FALSE, &mat[0][0]));
  }
  void setMat3f(const std::string &name, const glm::mat3 &mat) {
    glCheckError(glUniformMatrix3fv(uniform_handles[name], 1, GL_FALSE, &mat[0][0]));
  }
  void setMat4f(const std::string &name, const glm::mat4 &mat) {
    glCheckError(glUniformMatrix4fv(uniform_handles[name], 1, GL_FALSE, &mat[0][0]));
  }

  ~ShaderProgram() = default;
 private:
  void initActiveAttributeLayout();
  int uniform_count{};
  int attribute_count{};
  void initUniformHandles();
  static void checkCompileErrors(GLuint shader, const std::string &type);
};
}  // namespace core

#endif