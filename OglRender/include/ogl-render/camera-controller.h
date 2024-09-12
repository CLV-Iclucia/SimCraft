//
// Created by creeper on 6/18/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_CONTROLLER_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_CONTROLLER_H_
#include <optional>
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
struct Window;
template<typename Derived>
struct CameraController : NonCopyable {
  explicit CameraController(Camera &camera_, Window &window_)
      : camera(camera_), window(window_) {
    glfwSetWindowUserPointer(window.window(), static_cast<void *>(this));
    registerInputListeners();
  }
  void registerInputListeners() const {
    glfwSetInputMode(window.window(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if constexpr (Derived::registerCustomCursorMoveListener) {
      glfwSetCursorPosCallback(window.window(), [](GLFWwindow *window, double xpos, double ypos) {
        auto *controller = windowUserPointer(window);
        controller->processCursorMove({xpos, ypos});
      });
    }
    if constexpr (Derived::registersCustomMouseScrollListener) {
      glfwSetScrollCallback(window.window(), [](GLFWwindow *window, double xoffset, double yoffset) {
        auto *controller = windowUserPointer(window);
        controller->processMouseScroll({yoffset});
      });
    }
    if constexpr (Derived::registerCustomKeyboardProcessor) {
      window.setPreRenderCallback([](GLFWwindow *window) {
        auto *controller = windowUserPointer(glfwGetCurrentContext());
        controller->processKeyboard();
      });
    }
  };
  void removeInputListeners() const {
    glfwSetInputMode(window.window(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    if constexpr (Derived::registerCustomCursorMoveListener) {
      glfwSetCursorPosCallback(window.window(), nullptr);
    }
    if constexpr (Derived::registersCustomMouseScrollListener) {
      glfwSetScrollCallback(window.window(), nullptr);
    }
  };
  virtual void processCursorMove(const CursorMoveEvent &event) {};
  virtual void processMouseScroll(const MouseScrollEvent &event) {};
  virtual void processKeyboard() {};
  ~CameraController() {
    removeInputListeners();
    glfwSetWindowUserPointer(window.window(), nullptr);
  }
protected:
  static constexpr bool registerCustomCursorMoveListener = false;
  static constexpr bool registersCustomMouseScrollListener = false;
  static constexpr bool registerCustomKeyboardProcessor = false;
  [[nodiscard]] CameraParameters &cameraParams() {
    return camera.m_params;
  }
  [[nodiscard]] const CameraParameters &cameraParams() const {
    return camera.m_params;
  }
  static void setWindowUserPointer(GLFWwindow *window, CameraController *controller) {
    glfwSetWindowUserPointer(window, controller);
  }
  static Derived *windowUserPointer(GLFWwindow *window) {
    return static_cast<Derived *>(glfwGetWindowUserPointer(window));
  }
  Camera &camera;
  Window &window;
};

struct FpsCameraControllerParams {
  Camera &camera;
  Window &window;
  float movingSpeed{0.01f};
  float movingSensitivity{0.05f};
};

struct FpsCameraController : CameraController<FpsCameraController> {
  explicit FpsCameraController(const FpsCameraControllerParams &params) :
      CameraController(params.camera, params.window),
      movingSpeed(params.movingSpeed),
      mouseSensitivity(params.movingSensitivity) {}
  void processCursorMove(const CursorMoveEvent &event) override;
  void processMouseScroll(const MouseScrollEvent &event) override;
  void processKeyboard() override;
  float movingSpeed{};
  float mouseSensitivity{};
  static constexpr bool registerCustomCursorMoveListener = true;
  static constexpr bool registerCustomMouseScrollListener = false;
  static constexpr bool registerCustomKeyboardProcessor = true;
private:
  float last_x{};
  float last_y{};
  std::optional<double> last_mouse_move_time{};
  std::optional<double> last_mouse_scroll_time{};
  std::optional<double> last_keyboard_time{};
};

}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_CONTROLLER_H_
