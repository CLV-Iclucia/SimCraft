//
// Created by creeper on 5/27/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_GL_UTILS_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_GL_UTILS_H_

#include <GL/gl.h>
#include <iostream>
#define glCheckError(func) \
  do { \
    func;                   \
    details::gl_check_error(#func, __FILE__, __LINE__); \
    } while (0)

#define FOR_EACH_GL_ERROR(replace) \
    replace(INVALID_ENUM)          \
    replace(INVALID_VALUE)         \
    replace(INVALID_OPERATION)     \
    replace(STACK_OVERFLOW)        \
    replace(STACK_UNDERFLOW)       \
    replace(OUT_OF_MEMORY)         \
    replace(TABLE_TOO_LARGE)

namespace opengl::details {
inline void gl_check_error(const char *function_name, const char *file, int line) {
    GLenum errorCode;
    if ((errorCode = glGetError()) != GL_NO_ERROR) {
      std::string error;
      switch (errorCode) {
#define REPLACE_ENUM(name) case GL_##name:error = #name; break;
        FOR_EACH_GL_ERROR(REPLACE_ENUM)
#undef REPLACE_ENUM
        default:error = "UNKNOWN_ERROR";
          break;
      }
      std::cerr << "OpenGL Error: " << error << " in " << function_name << " " << file << ":" << line << std::endl;
      exit(1);
    }
}
}
#undef FOR_EACH_GL_ERROR
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_GL_UTILS_H_
