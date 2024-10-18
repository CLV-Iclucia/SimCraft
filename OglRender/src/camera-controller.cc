//
// Created by creeper on 6/18/24.
//
#include <ogl-render/camera-controller.h>
namespace opengl {

void FpsCameraController::processCursorMove(const CursorMoveEvent &event) {
  auto [x, y] = event;
  if (!last_mouse_move_time) {
    last_x = x;
    last_y = y;
    last_mouse_move_time = glfwGetTime();
    return;
  }
  float dt = glfwGetTime() - last_mouse_move_time.value();
  double xoffset = x - last_x;
  double yoffset = last_y - y;
  last_x = x;
  last_y = y;
  double delta_yaw_in_deg = xoffset * mouseSensitivity * dt;
  double delta_pitch_in_deg = yoffset * mouseSensitivity * dt;
  cameraParams().yaw_in_deg += static_cast<float>(delta_yaw_in_deg);
  cameraParams().pitch_in_deg += static_cast<float>(delta_pitch_in_deg);
}
void FpsCameraController::processMouseScroll(const MouseScrollEvent &event) {

}
void FpsCameraController::processKeyboard() {
  if (!last_keyboard_time) {
    last_keyboard_time = glfwGetTime();
    return;
  }
  double dt = glfwGetTime() - last_keyboard_time.value();
  if (glfwGetKey(window.window(), GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window.window(), GLFW_TRUE);
  if (glfwGetKey(window.window(), GLFW_KEY_D) == GLFW_PRESS)
    cameraParams().position -= camera.right() * movingSpeed * static_cast<float>(dt);
  if (glfwGetKey(window.window(), GLFW_KEY_S) == GLFW_PRESS)
    cameraParams().position += camera.right() * movingSpeed * static_cast<float>(dt);
  if (glfwGetKey(window.window(), GLFW_KEY_A) == GLFW_PRESS)
    cameraParams().position += camera.front() * movingSpeed * static_cast<float>(dt);
  if (glfwGetKey(window.window(), GLFW_KEY_W) == GLFW_PRESS)
    cameraParams().position -= camera.front() * movingSpeed * static_cast<float>(dt);
}
}