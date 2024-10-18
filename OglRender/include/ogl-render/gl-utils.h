//
// Created by creeper on 5/27/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_GL_UTILS_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_GL_UTILS_H_
#include <GL/gl.h>
#include <iostream>
#include <format>
#ifndef NDEBUG
#define glCheckError(statement) \
  do { \
    statement;                   \
    details::gl_check_error(#statement, __FILE__, __LINE__); \
    } while (0)
#else
#define glCheckError(func) func
#endif
#define FOR_EACH_GL_ERROR(replace) \
    replace(INVALID_ENUM)          \
    replace(INVALID_VALUE)         \
    replace(INVALID_OPERATION)     \
    replace(STACK_OVERFLOW)        \
    replace(STACK_UNDERFLOW)       \
    replace(OUT_OF_MEMORY)

namespace opengl {
namespace details {
inline void gl_check_error(const char *statement, const char *file, int line) {
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
    std::cerr << std::format("OpenGL error in statement {} at {}:{}: {}\n", statement, file, line, error);
    exit(1);
  }
}
}
template <typename T>
requires std::is_pod_v<T>
constexpr GLenum glType() {
  if constexpr (std::is_same_v<T, float>) {
    return GL_FLOAT;
  } else if constexpr (std::is_same_v<T, int>) {
    return GL_INT;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return GL_UNSIGNED_INT;
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    return GL_UNSIGNED_BYTE;
  } else if constexpr (std::is_same_v<T, double>) {
    return GL_DOUBLE;
  } else {
    static_assert(false, "Unsupported type");
  }
}

}

#undef FOR_EACH_GL_ERROR
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_GL_UTILS_H_
