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
#include <ogl-render/ogl-ctx.h>
namespace opengl {

struct ShaderProgramConfig {
  std::filesystem::path vertex_shader_path;
  std::filesystem::path fragment_shader_path;
  std::filesystem::path geometry_shader_path;
};

// RAII shader program
struct ShaderProg : Resource {
  GLuint id;
  // create shader program from vertex and fragment shader source
  explicit ShaderProg(ShaderProgramConfig config);
  std::unordered_map<std::string, GLuint> uniform_handles;
  std::unordered_map<std::string, GLuint> attribute_handles;

  void activeAttributes(std::vector<std::string> &attribute_names) const {
    GLint num_attribs;
    glGetProgramiv(id, GL_ACTIVE_ATTRIBUTES, &num_attribs);
    char name[256];
    GLint written, size, location;
    GLenum type;
    for (int i = 0; i < num_attribs; ++i) {
      glGetActiveAttrib(id, i, 256, &written, &size, &type, name);
      location = glGetAttribLocation(id, name);
      attribute_names.emplace_back(name);
    }
  }

  void use() const { glUseProgram(id); }
  void setInt(const std::string &name, int value) {
    glUniform1i(uniform_handles[name], value);
  }
  void setFloat(const std::string &name, float value) {
    glUniform1f(uniform_handles[name], value);
  }
  void setVec2f(const std::string &name, float x, float y) {
    glUniform2f(uniform_handles[name], x, y);
  }
  void setVec2f(const std::string &name, const glm::vec2 &value) {
    glUniform2f(uniform_handles[name], value.x, value.y);
  }
  void setVec3f(const std::string &name, float x, float y, float z) {
    glUniform3f(uniform_handles[name], x, y, z);
  }
  void setVec3f(const std::string &name, const glm::vec3 &value) {
    glUniform3f(uniform_handles[name], value.x, value.y, value.z);
  }
  void setVec4f(const std::string &name, float x, float y, float z, float w) {
    glUniform4f(uniform_handles[name], x, y, z, w);
  }
  void setVec4f(const std::string &name, const glm::vec4 &value) {
    glUniform4f(uniform_handles[name], value.x, value.y, value.z, value.w);
  }
  void setMat2f(const std::string &name, const glm::mat2 &mat) {
    glUniformMatrix2fv(uniform_handles[name], 1, GL_FALSE, &mat[0][0]);
  }
  void setMat3f(const std::string &name, const glm::mat3 &mat) {
    glUniformMatrix3fv(uniform_handles[name], 1, GL_FALSE, &mat[0][0]);
  }
  void setMat4f(const std::string &name, const glm::mat4 &mat) {
    glUniformMatrix4fv(uniform_handles[name], 1, GL_FALSE, &mat[0][0]);
  }

  ~ShaderProg() { glDeleteProgram(id); }
 private:
  int uniform_count;
  int attribute_count;
  void initUniformHandles();
  void initAttributeHandles();
  static void checkCompileErrors(GLuint shader, const std::string &type);
};
}  // namespace core

#endif