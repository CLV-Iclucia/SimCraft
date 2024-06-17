//
// Created by creeper on 10/25/23.
//
#include <ogl-render/shader-prog.h>

namespace opengl {
void ShaderProgram::initUniformHandles() {
  char name[256];
  for (int i = 0; i < uniform_count; i++) {
    int length;
    GLenum type;
    int size;
    glCheckError(glGetActiveUniform(id, i, 256, &length, &size, &type, name));
    uniform_handles[name] = glGetUniformLocation(id, name);
  }
}
void ShaderProgram::initAttributeHandles() {
  char name[256];
  for (int i = 0; i < attribute_count; i++) {
    int length;
    GLenum type;
    int size;
    glCheckError(glGetActiveAttrib(id, i, 256, &length, &size, &type, name));
//    attribute_handles[name] = glGetAttribLocation(id, name);
  }
}
ShaderProgram::ShaderProgram(ShaderProgramConfig config) {
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
    vShaderFile.open(config.vertex_shader_path);
    fShaderFile.open(config.fragment_shader_path);
    std::stringstream vShaderStream, fShaderStream;
    vShaderStream << vShaderFile.rdbuf();
    fShaderStream << fShaderFile.rdbuf();
    vShaderFile.close();
    fShaderFile.close();
    vertexCode = vShaderStream.str();
    fragmentCode = fShaderStream.str();
    if (!config.geometry_shader_path.empty()) {
      gShaderFile.open(config.geometry_shader_path);
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
  if (!geometryCode.empty()) {
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
  if (!geometryCode.empty())
    glAttachShader(id, geometry);
  glLinkProgram(id);
  checkCompileErrors(id, "PROGRAM");
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  if (!geometryCode.empty())
    glDeleteShader(geometry);
  glGetProgramiv(id, GL_ACTIVE_UNIFORMS, &uniform_count);
  glGetProgramiv(id, GL_ACTIVE_ATTRIBUTES, &attribute_count);
  initAttributeHandles();
  initUniformHandles();
}
void ShaderProgram::checkCompileErrors(GLuint shader, const std::string &type) {
  GLint success;
  GLchar infoLog[1024];
  if (type != "PROGRAM") {
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
      std::cout << std::format(
          "ERROR::SHADER_COMPILATION_ERROR of type: {}\n{}\n -------------------------------------------------------\n",
          type,
          infoLog) << std::endl;
    }
  } else {
    glGetProgramiv(shader, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
      std::cout << std::format(
          "ERROR::PROGRAM_LINKING_ERROR of type: {}\n{}\n -------------------------------------------------------\n",
          type,
          infoLog) << std::endl;
    }
  }
}
}