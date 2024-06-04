//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_GLFW_WINDOW_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_GLFW_WINDOW_H_
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
namespace opengl {
// should be used after glfwInit
struct GLFWGuard {
  GLFWGuard(int version_major, int version_minor) {
    if (!glfwInit()) {
      std::cerr << "Failed to initialize GLFW\n";
      exit(1);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, version_major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, version_minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  }
  ~GLFWGuard() {
    glfwTerminate();
  }
};

class Window {
 public:
  Window(int width, int height, const std::string &title) {
    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window)
      glfwTerminate();
  }
  GLFWwindow *get() {
    return window;
  }
  ~Window() {
    glfwDestroyWindow(window);
  }
 private:
  GLFWwindow *window{};
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_GLFW_WINDOW_H_
