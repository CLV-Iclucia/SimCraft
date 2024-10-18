//
// Created by creeper on 6/17/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_PROPERTIES_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_PROPERTIES_H_
namespace opengl {
struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};

struct Resource {
  Resource() = default;
  Resource(Resource &&) = delete;
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_PROPERTIES_H_
