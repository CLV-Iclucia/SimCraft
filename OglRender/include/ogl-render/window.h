//
// Created by creeper on 6/17/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_WINDOW_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_WINDOW_H_
#include <ogl-render/glad-glfw.h>
#include <ogl-render/properties.h>
#include <ogl-render/ext-gui-wrapper.h>
#include <iostream>
#include <memory>
namespace opengl {
struct Window : Resource {
  Window(int width_, int height_, std::string_view title) : m_width(width_), m_height(height_) {
    if (!glfwInit()) {
      // TODO: ERROR
      std::cerr << "Failed to initialize GLFW\n";
      exit(-1);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    m_window = glfwCreateWindow(m_width, m_height, title.data(), nullptr, nullptr);
    if (!m_window) {
      std::cerr << "Failed to create window\n";
      glfwTerminate();
      exit(-1);
    }
    glfwMakeContextCurrent(m_window);
  }
  void preRender() {
    glfwPollEvents();
    glfwGetFramebufferSize(m_window, &m_width, &m_height);
  }
  void postRender() {
    glfwSwapBuffers(m_window);
  }
  [[nodiscard]] int width() const {
    return m_width;
  }
  [[nodiscard]] int height() const {
    return m_height;
  }
  bool shouldClose() {
    return glfwWindowShouldClose(m_window);
  }
  [[nodiscard]] GLFWwindow *window() const {
    return m_window;
  }
  ~Window() {
    glfwDestroyWindow(m_window);
    glfwTerminate();
  }
 private:
  int m_width{}, m_height{};
  GLFWwindow *m_window{};
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_WINDOW_H_
