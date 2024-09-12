//
// Created by creeper on 7/30/24.
//

#ifndef MESHVIEW_OGLRENDER_INCLUDE_OGL_RENDER_OGL_TYPES_H_
#define MESHVIEW_OGLRENDER_INCLUDE_OGL_RENDER_OGL_TYPES_H_

#include <cstdint>
#include <glad/glad.h>
#include <iostream>

namespace opengl {
enum class OglBasicType : uint8_t {
  Unsigned,
  Int,
  Float,
  Bool,
  Double,
};

struct OglTypeInfo {
  OglBasicType type;
  int component_count;
  int array_size;

  bool operator==(const OglTypeInfo &other) const {
    return type == other.type && component_count == other.component_count && array_size == other.array_size;
  }
};

inline OglBasicType glBasicType(GLenum type_code) {
  switch (type_code) {
    case GL_UNSIGNED_BYTE:
    case GL_UNSIGNED_SHORT:
    case GL_UNSIGNED_INT:return OglBasicType::Unsigned;
    case GL_BYTE:
    case GL_SHORT:
    case GL_INT:return OglBasicType::Int;
    case GL_FLOAT:
    case GL_FLOAT_VEC2:
    case GL_FLOAT_VEC3:
    case GL_FLOAT_VEC4:
    case GL_FLOAT_MAT2:
    case GL_FLOAT_MAT3:
    case GL_FLOAT_MAT4:
    case GL_FLOAT_MAT2x3:
    case GL_FLOAT_MAT2x4:
    case GL_FLOAT_MAT3x2:
    case GL_FLOAT_MAT3x4:
    case GL_FLOAT_MAT4x2:
    case GL_FLOAT_MAT4x3:return OglBasicType::Float;
    case GL_BOOL:
    case GL_BOOL_VEC2:
    case GL_BOOL_VEC3:
    case GL_BOOL_VEC4:return OglBasicType::Bool;
    case GL_DOUBLE:
    case GL_DOUBLE_VEC2:
    case GL_DOUBLE_VEC3:
    case GL_DOUBLE_VEC4:
    case GL_DOUBLE_MAT2:
    case GL_DOUBLE_MAT3:
    case GL_DOUBLE_MAT4:
    case GL_DOUBLE_MAT2x3:
    case GL_DOUBLE_MAT2x4:
    case GL_DOUBLE_MAT3x2:
    case GL_DOUBLE_MAT3x4:
    case GL_DOUBLE_MAT4x2:
    case GL_DOUBLE_MAT4x3:return OglBasicType::Double;
    default: {
      std::cerr << "Unknown type code: " << type_code << std::endl;
      exit(1);
    }
  }
}

inline int componentCount(GLenum type_code) {
  switch (type_code) {
    case GL_FLOAT:
    case GL_INT:
    case GL_UNSIGNED_INT:
    case GL_BYTE:
    case GL_SHORT:
    case GL_UNSIGNED_BYTE:
    case GL_UNSIGNED_SHORT:
    case GL_BOOL:
    case GL_DOUBLE:return 1;

    case GL_FLOAT_VEC2:
    case GL_INT_VEC2:
    case GL_UNSIGNED_INT_VEC2:
    case GL_BOOL_VEC2:
    case GL_DOUBLE_VEC2:return 2;
    case GL_FLOAT_VEC3:
    case GL_INT_VEC3:
    case GL_UNSIGNED_INT_VEC3:
    case GL_BOOL_VEC3:
    case GL_DOUBLE_VEC3:return 3;
    case GL_FLOAT_VEC4:
    case GL_INT_VEC4:
    case GL_UNSIGNED_INT_VEC4:
    case GL_BOOL_VEC4:
    case GL_DOUBLE_VEC4:return 4;

    case GL_FLOAT_MAT2:
    case GL_DOUBLE_MAT2:return 4; // 2x2 matrix has 4 components
    case GL_FLOAT_MAT3:
    case GL_DOUBLE_MAT3:return 9; // 3x3 matrix has 9 components
    case GL_FLOAT_MAT4:
    case GL_DOUBLE_MAT4:return 16; // 4x4 matrix has 16 components
    case GL_FLOAT_MAT2x3:
    case GL_DOUBLE_MAT2x3:return 6; // 2x3 matrix has 6 components
    case GL_FLOAT_MAT2x4:
    case GL_DOUBLE_MAT2x4:return 8; // 2x4 matrix has 8 components
    case GL_FLOAT_MAT3x2:
    case GL_DOUBLE_MAT3x2:return 6; // 3x2 matrix has 6 components
    case GL_FLOAT_MAT3x4:
    case GL_DOUBLE_MAT3x4:return 12; // 3x4 matrix has 12 components
    case GL_FLOAT_MAT4x2:
    case GL_DOUBLE_MAT4x2:return 8; // 4x2 matrix has 8 components
    case GL_FLOAT_MAT4x3:
    case GL_DOUBLE_MAT4x3:return 12; // 4x3 matrix has 12 components

    default: {
      std::cerr << "Unknown type code: " << type_code << std::endl;
      exit(1);
    }
  }
}

inline bool isMatrix(GLenum type_code) {
  switch (type_code) {
    case GL_FLOAT_MAT2:
    case GL_FLOAT_MAT3:
    case GL_FLOAT_MAT4:
    case GL_FLOAT_MAT2x3:
    case GL_FLOAT_MAT2x4:
    case GL_FLOAT_MAT3x2:
    case GL_FLOAT_MAT3x4:
    case GL_FLOAT_MAT4x2:
    case GL_FLOAT_MAT4x3:
    case GL_DOUBLE_MAT2:
    case GL_DOUBLE_MAT3:
    case GL_DOUBLE_MAT4:
    case GL_DOUBLE_MAT2x3:
    case GL_DOUBLE_MAT2x4:
    case GL_DOUBLE_MAT3x2:
    case GL_DOUBLE_MAT3x4:
    case GL_DOUBLE_MAT4x2:
    case GL_DOUBLE_MAT4x3:return true;
    default:return false;
  }
}

}
#endif //MESHVIEW_OGLRENDER_INCLUDE_OGL_RENDER_OGL_TYPES_H_
