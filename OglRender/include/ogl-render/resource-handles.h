#ifndef OGL_RENDER_INCLUDE_OGL_RENDER_OGL_CTX_H_
#define OGL_RENDER_INCLUDE_OGL_RENDER_OGL_CTX_H_

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <span>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <ogl-render/gl-utils.h>
#include <ogl-render/properties.h>
namespace opengl {

template<typename Derived>
struct GLHandleBase : NonCopyable {
  GLuint id{};
  GLHandleBase() {
    static_cast<Derived *>(this)->create();
  }
  GLHandleBase(GLHandleBase &&other) noexcept {
    id = other.id;
    other.id = 0;
  }
  GLHandleBase &operator=(GLHandleBase &&other) noexcept {
    if (this != &other) {
      id = other.id;
      other.id = 0;
    }
    return *this;
  }
  virtual void bind() const {
    static_cast<const Derived *>(this)->bind();
  }
  ~GLHandleBase() {
    static_cast<Derived *>(this)->destroy();
  }
};

struct BufferSubDataOption {
  std::span<const std::byte> data;
  size_t offset_in_bytes;
};

template<GLenum Target>
struct BufferObject : GLHandleBase<BufferObject<Target>> {
  using GLHandleBase<BufferObject<Target>>::id;
  size_t size{};
  void create() {
    glCheckError(glGenBuffers(1, &id));
  }
  void bind() const override {
    glCheckError(glBindBuffer(Target, id));
  }
  static void unbind() {
    glCheckError(glBindBuffer(Target, 0));
  }
  void destroy() {
    glCheckError(glDeleteBuffers(1, &id));
  }
  template<GLenum Usage>
  void bufferData(std::span<const std::byte> data) {
    size = data.size_bytes();
    glCheckError(glBufferData(Target, data.size_bytes(), data.data(), Usage));
  }
  void bufferSubData(const BufferSubDataOption &option) {
    assert(option.offset_in_bytes + option.data.size_bytes() <= size);
    glCheckError(glBufferSubData(Target, option.offset_in_bytes, option.data.size_bytes(), option.data.data()));
  }
};

using VertexBufferObj = BufferObject<GL_ARRAY_BUFFER>;
using ElementBufferObj = BufferObject<GL_ELEMENT_ARRAY_BUFFER>;

struct VertexAttribPointerOption {
  GLuint index{};
  GLint size{};
  GLenum type{};
  GLboolean normalized{GL_FALSE};
  GLsizei stride{};
  const void *pointer{};
};

struct VertexArrayObject : GLHandleBase<VertexArrayObject> {
  using GLHandleBase<VertexArrayObject>::id;
  void create() {
    glCheckError(glGenVertexArrays(1, &id));
  }
  void bind() const override {
    glCheckError(glBindVertexArray(id));
  }
  void destroy() {
    glCheckError(glDeleteVertexArrays(1, &id));
  }
  static void unbind() {
    glCheckError(glBindVertexArray(0));
  }
  static void vertexAttribPointer(const VertexAttribPointerOption &option) {
    const auto& [index, size, type, normalized, stride, pointer] = option;
    glCheckError(glVertexAttribPointer(index, size, type, normalized, stride, pointer));
  }
  static void enableVertexAttribArray(GLuint index) {
    glCheckError(glEnableVertexAttribArray(index));
  }
  static void disableVertexAttribArray(GLuint index) {
    glCheckError(glEnableVertexAttribArray(index));
  }
};

template<GLenum Target>
struct TextureObject : GLHandleBase<TextureObject<Target>> {
  using GLHandleBase<TextureObject>::id;
  void create() {
    glCheckError(glGenTextures(1, &id));
  }
  void bind() const override {
    glCheckError(glBindTexture(Target, id));
  }
  void destroy() {
    glCheckError(glDeleteTextures(1, &id));
  }
  static void unbind() {
    glCheckError(glBindTexture(Target, 0));
  }
};

}

#endif