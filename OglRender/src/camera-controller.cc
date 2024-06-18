//
// Created by creeper on 6/18/24.
//
#include <ogl-render/camera-controller.h>
namespace opengl {

void FpsCameraController::processMouseMove(const CursorMoveEvent &event) {
  if (firstMouse) {
    last_x = event.x;
    last_y = event.y;
    firstMouse = false;
  }
  float xoffset = event.x - last_x;
  float yoffset = last_y - event.y;
  last_x = event.x;
  last_y = event.y;

}
void FpsCameraController::registerInputListeners() {
  glfwSetInputMode(window.window(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetCursorPosCallback(window.window(), [](GLFWwindow *window, double xpos, double ypos) {
    auto *controller = windowUserPointer(window);
    controller->processMouseMove({xpos, ypos});
  });
}
}