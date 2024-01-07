#ifndef SPATIFY_INCLUDE_SPATIFY_TYPES_H_
#define SPATIFY_INCLUDE_SPATIFY_TYPES_H_
#include <glm/glm.hpp>
#include <vector>
namespace spatify {
using Real = double;
using Index = int;
using Vec2i = glm::ivec2;
using Vec3i = glm::ivec3;
using Vec4i = glm::ivec4;
using Vec2u = glm::uvec2;
using Vec3u = glm::uvec3;
using Vec4u = glm::uvec4;
using Vec2f = glm::vec2;
using Vec3f = glm::vec3;
using Vec4f = glm::vec4;
using Vec2d = glm::highp_vec2;
using Vec3d = glm::highp_vec3;
using Vec4d = glm::highp_vec4;
using Mat2f = glm::mat2;
using Mat3f = glm::mat3;
using Mat4f = glm::mat4;
using Mat2d = glm::highp_mat2;
using Mat3d = glm::highp_mat3;
using Mat4d = glm::highp_mat4;
using std::vector;
template <typename T, int Dim> struct TVector {
  static_assert(Dim == 2 || Dim == 3 || Dim == 4, "Dim must be 2, 3 or 4");
};
template <typename T> struct TVector<T, 2> {
  using type = glm::detail::tvec2<T>;
};
template <typename T> struct TVector<T, 3> {
  using type = glm::detail::tvec3<T>;
};
template <typename T> struct TVector<T, 4> {
  using type = glm::detail::tvec4<T>;
};
template <typename T, int Dim> using Vector = typename TVector<T, Dim>::type;
template <typename T, int Dim> struct TMatrix {
  static_assert(Dim == 2 || Dim == 3 || Dim == 4, "Dim must be 2, 3 or 4");
};
template <typename T> struct TMatrix<T, 2> {
  using type = glm::detail::tmat2x2<T>;
};
template <typename T> struct TMatrix<T, 3> {
  using type = glm::detail::tmat3x3<T>;
};
template <typename T> struct TMatrix<T, 4> {
  using type = glm::detail::tmat4x4<T>;
};

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};
}

#endif