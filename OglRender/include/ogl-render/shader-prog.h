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
#include <glm/core/type.hpp>
#include <vector>
namespace opengl {

// RAII shader program
struct ShaderProg {
  GLuint id;
  // create shader program from vertex and fragment shader source
  ShaderProg(const char *vs_path, const char *fs_path, const char *gs_path = nullptr) {
    // 1. retrieve the vertex/fragment source code from filePath
    std::string vertexCode;
    std::string fragmentCode;
    std::string geometryCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;
    std::ifstream gShaderFile;
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      vShaderFile.open(vs_path);
      fShaderFile.open(fs_path);
      std::stringstream vShaderStream, fShaderStream;
      vShaderStream << vShaderFile.rdbuf();
      fShaderStream << fShaderFile.rdbuf();
      vShaderFile.close();
      fShaderFile.close();
      vertexCode = vShaderStream.str();
      fragmentCode = fShaderStream.str();
      if (gs_path != nullptr) {
        gShaderFile.open(gs_path);
        std::stringstream gShaderStream;
        gShaderStream << gShaderFile.rdbuf();
        gShaderFile.close();
        geometryCode = gShaderStream.str();
      }
    }
    catch (std::ifstream::failure &e) {
      std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl;
    }
    const char *vShaderCode = vertexCode.c_str();
    const char *fShaderCode = fragmentCode.c_str();
    unsigned int vertex, fragment;
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, nullptr);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");
    // fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, nullptr);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");
    // if geometry shader is given, compile geometry shader
    unsigned int geometry;
    if (gs_path != nullptr) {
      const char *gShaderCode = geometryCode.c_str();
      geometry = glCreateShader(GL_GEOMETRY_SHADER);
      glShaderSource(geometry, 1, &gShaderCode, nullptr);
      glCompileShader(geometry);
      checkCompileErrors(geometry, "GEOMETRY");
    }
    // shader Program
    id = glCreateProgram();
    glAttachShader(id, vertex);
    glAttachShader(id, fragment);
    if (gs_path != nullptr)
      glAttachShader(id, geometry);
    glLinkProgram(id);
    checkCompileErrors(id, "PROGRAM");
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (gs_path != nullptr)
      glDeleteShader(geometry);
    glGetProgramiv(id, GL_ACTIVE_UNIFORMS, &uniform_count);
    glGetProgramiv(id, GL_ACTIVE_ATTRIBUTES, &attribute_count);
    initAttributeHandles();
    initUniformHandles();
  }
  std::unordered_map<std::string, GLuint> uniform_handles;
  std::unordered_map<std::string, GLuint> attribute_handles;
  void initUniformHandles();
  void initAttributeHandles();

  void getActiveAttributes(std::vector<std::string>& attribute_names) const {
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
  static void unuse() {
    glUseProgram(0);
  }
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
  static void checkCompileErrors(GLuint shader, std::string type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
      glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
      if (!success) {
        glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
        std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog
                  << "\n -- --------------------------------------------------- -- " << std::endl;
      }
    } else {
      glGetProgramiv(shader, GL_LINK_STATUS, &success);
      if (!success) {
        glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
        std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog
                  << "\n -- --------------------------------------------------- -- " << std::endl;
      }
    }
  }
};
}  // namespace core

#endif