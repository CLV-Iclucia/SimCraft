//
// Created by creeper on 6/18/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_CONTROLLER_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_CONTROLLER_H_
#include <ogl-render/camera.h>
#include <ogl-render/window.h>
namespace opengl {
struct CursorMoveEvent {
  double x;
  double y;
};
struct MouseScrollEvent {
  double yoffset;
};
struct KeyboardEvent {
  int key;
  int action;
  int mods;
};
struct Window;
template <typename Derived>
struct CameraController {
  explicit CameraController(Camera &camera_, Window& window_) : camera(camera_), window(window_) {
    glfwSetWindowUserPointer(window.window(), static_cast<void*>(this));
  }
  virtual void registerInputListeners() {};
  virtual void processMouseMove(const CursorMoveEvent &event) {};
  virtual void processMouseScroll(const MouseScrollEvent& event) {};
  virtual void processKeyboard(const KeyboardEvent& event) {};
 protected:
  static void setWindowUserPointer(GLFWwindow *window, CameraController *controller) {
    glfwSetWindowUserPointer(window, controller);
  }
  static Derived* windowUserPointer(GLFWwindow *window) {
    return static_cast<Derived *>(glfwGetWindowUserPointer(window));
  }
  Camera &camera;
  Window &window;
};

struct FpsCameraController : CameraController<FpsCameraController> {
  explicit FpsCameraController(Camera &camera_, Window& window_) : CameraController(camera_, window_) {}
  void registerInputListeners() override;
  void processMouseMove(const CursorMoveEvent &event) override;
 private:
  float last_x{};
  float last_y{};
  bool firstMouse{true};
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_CONTROLLER_H_
