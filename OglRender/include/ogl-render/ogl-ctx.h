#ifndef OGL_RENDER_INCLUDE_OGL_RENDER_OGL_CTX_H_
#define OGL_RENDER_INCLUDE_OGL_RENDER_OGL_CTX_H_

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <ogl-render/shader-prog.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
namespace opengl {
#ifndef offsetof
#define offsetof(s, m) ((size_t)&(((s*)0)->m))
#endif
struct ShaderProg;

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};

template<typename Derived>
struct OpenGLHandle : NonCopyable {
  GLuint id;
  OpenGLHandle() = default;
  OpenGLHandle(OpenGLHandle&& other) {
    id = other.id;
    other.id = 0;
  }
};

struct VertexBufferObj : NonCopyable {
  GLuint id;
  VertexBufferObj() {
    glGenBuffers(1, &id);
  }
  VertexBufferObj(VertexBufferObj &&other) {
    id = other.id;
    other.id = 0;
  }
  void bind() const {
    glBindBuffer(GL_ARRAY_BUFFER, id);
  }
  template<typename T>
  void allocData(const std::vector<T> &data) {
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), (void *) data.data(), GL_STATIC_DRAW);
  }
  template<typename T>
  void allocData(const T *data, int size) {
    glBufferData(GL_ARRAY_BUFFER, size * sizeof(T), (void *) data, GL_STATIC_DRAW);
  }
  template<typename T>
  void passData(const std::vector<T> &data) {
    glBufferSubData(GL_ARRAY_BUFFER, 0, data.size() * sizeof(T), (void *) data.data());
  }
  template<typename T>
  void updateData(const T *data, int offset, int size) {
    glBufferSubData(GL_ARRAY_BUFFER, offset * sizeof(T), size * sizeof(T), data);
  }
  void bind() {
    glBindBuffer(GL_ARRAY_BUFFER, id);
  }
  ~VertexBufferObj() {
    glDeleteBuffers(1, &id);
  }
};

struct VertexArrayObj : NonCopyable {
  GLuint id;
  VertexArrayObj() {
    glGenVertexArrays(1, &id);
  }
  void bind() const {
    glBindVertexArray(id);
  }
  static void unbind() {
    glBindVertexArray(0);
  }
  ~VertexArrayObj() {
    glDeleteVertexArrays(1, &id);
    glBindVertexArray(0);
  }
};

struct ElementBufferObj : NonCopyable {
  GLuint id;
  ElementBufferObj() {
    glGenBuffers(1, &id);
  }
  ElementBufferObj(const std::vector<GLuint> &data) {
    glGenBuffers(1, &id);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(GLuint), data.data(), GL_STATIC_DRAW);
  }
  void passData(const std::vector<GLuint> &data) {
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(GLuint), data.data(), GL_STATIC_DRAW);
  }
  void bind() {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id);
  }
  ~ElementBufferObj() {
    glDeleteBuffers(1, &id);
  }
};

struct OpenGLContext : NonCopyable {
  VertexArrayObj vao;
  std::vector<VertexBufferObj> vbo;
  ElementBufferObj ebo;
  glm::vec3 clear_color;
  int attribute_count = 0;
  OpenGLContext() = default;
  std::unordered_map<std::string, GLuint> attributes;

  void registerAttribute(const std::string &name,
                         GLsizei size,
                         GLenum type,
                         GLboolean normalized,
                         GLsizei stride,
                         const void *pointer) {
    attributes[name] = attribute_count++;
    glVertexAttribPointer(attributes[name], size, type, normalized, stride, pointer);
    glEnableVertexAttribArray(attributes[name]);
  }
  VertexBufferObj &VBO(const std::string &name) {
    return vbo[attributes[name]];
  }
  int attribute(const std::string &name) {
    return static_cast<int>(attributes[name]);
  }
  void loadAttributesFromShader(const ShaderProg &shader) {
    for (const auto &attr : shader.attribute_handles)
      attributes[attr.first] = attr.second;
  }

  template<typename T>
  void newAttribute(const std::string &name, const std::vector<T> &data, int size, int stride, int type) {
    vbo.emplace_back();
    vbo.back().bind();
    vbo.back().allocData(data);
    registerAttribute(name, size, type, false, stride, 0);
  }

  template<typename T>
  void designateAttributeData(const std::string &name, const std::vector<T> &data, int size, int stride, int type) {
    if (attributes.find(name) == attributes.end()) {
      std::cerr << "[Warning] Attribute " << name << " not found" << std::endl;
      return;
    }
    glVertexAttribPointer(attributes[name], size, type, false, stride, 0);
    glEnableVertexAttribArray(attributes[name]);
  }

  static void draw(GLuint mode, int count) {
    glDrawElements(mode, count, GL_UNSIGNED_INT, 0);
  }

  static void unbind() {
    VertexArrayObj::unbind();
  }
  virtual ~OpenGLContext() = default;
};
}

#endif