//
// Created by creeper on 10/25/23.
//
#include <ogl-render/shader-prog.h>

namespace opengl {
void ShaderProg::initUniformHandles() {
  char name[256];
  for (int i = 0; i < uniform_count; i++) {
    int length;
    GLenum type;
    int size;
    glGetActiveUniform(id, i, 256, &length, &size, &type, name);
    uniform_handles[name] = glGetUniformLocation(id, name);
  }
}
void ShaderProg::initAttributeHandles() {
  char name[256];
  for (int i = 0; i < attribute_count; i++) {
    int length;
    GLenum type;
    int size;
    glGetActiveAttrib(id, i, 256, &length, &size, &type, name);
    attribute_handles[name] = glGetAttribLocation(id, name);
    std::cout << "Attribute " << name << " has location " << attribute_handles[name] << std::endl;
  }
}
}