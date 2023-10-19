#ifndef CORE_SHADER_PROG_H_
#define CORE_SHADER_PROG_H_

#include <glad/glad.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cerrno>
#include <cstring>
namespace core {

// RAII shader program
struct ShaderProg {
  static char error_log[512];
  uint id;
  // create shader program from vertex and fragment shader source
  ShaderProg(const char* vs_path, const char* fs_path) {
    id = glCreateProgram();
    uint vert_id = glCreateShader(GL_VERTEX_SHADER);
    uint frag_id = glCreateShader(GL_FRAGMENT_SHADER);
    std::ifstream vert_file(vs_path);
    std::ifstream frag_file(fs_path);
    // if failed, check the reason and output
    if (!vert_file.is_open()) {
      std::cerr << "Failed to open vertex shader file: " << vs_path << std::endl;
      std::cerr << "Reason: " << std::strerror(errno) << std::endl;
      exit(-1);
    }
    if (!frag_file.is_open()) {
      std::cerr << "Failed to open fragment shader file: " << fs_path << std::endl;
      std::cerr << "Reason: " << std::strerror(errno) << std::endl;
      exit(-1);
    }
    std::stringstream vert_stream, frag_stream;
    vert_stream << vert_file.rdbuf();
    frag_stream << frag_file.rdbuf();
    std::string vert = vert_stream.str();
    std::string frag = frag_stream.str();
    const char* vert_src = vert.c_str();
    const char* frag_src = frag.c_str();
    glShaderSource(vert_id, 1, &vert_src, NULL);
    glShaderSource(frag_id, 1, &frag_src, NULL);
    glCompileShader(vert_id);
    glCompileShader(frag_id);
    glAttachShader(id, vert_id);
    glAttachShader(id, frag_id);
    glLinkProgram(id);
    glDeleteShader(vert_id);
    glDeleteShader(frag_id);
  }
  void use() { glUseProgram(id); }
  void setInt(const std::string& name, int value) {
    glUniform1i(glGetUniformLocation(id, name.c_str()), value);
  }
  void setFloat(const std::string& name, float value) {
    glUniform1f(glGetUniformLocation(id, name.c_str()), value);
  }
  void setVec2(const std::string& name, float x, float y) {
    glUniform2f(glGetUniformLocation(id, name.c_str()), x, y);
  }
  ~ShaderProg() { glDeleteProgram(id); }
};
}  // namespace core

#endif