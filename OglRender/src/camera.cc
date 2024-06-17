//
// Created by creeper on 10/25/23.
//
#include <ogl-render/window.h>
#include <ogl-render/camera.h>

namespace opengl {

void FpsCamera::updateCameraVectors() {
  glm::vec3 newFront;
  newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
  newFront.y = sin(glm::radians(pitch));
  newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
  front = glm::normalize(newFront);
}
void FpsCamera::processKeyBoard(const Window& window, float deltaTime) {
  float cameraSpeed = movementSpeed * deltaTime;
  if (glfwGetKey(window.window(), GLFW_KEY_W) == GLFW_PRESS)
    position += front * cameraSpeed;
  if (glfwGetKey(window.window(), GLFW_KEY_S) == GLFW_PRESS)
    position -= front * cameraSpeed;
  if (glfwGetKey(window.window(), GLFW_KEY_A) == GLFW_PRESS)
    position -= glm::normalize(glm::cross(front, up)) * cameraSpeed;
  if (glfwGetKey(window.window(), GLFW_KEY_D) == GLFW_PRESS)
    position += glm::normalize(glm::cross(front, up)) * cameraSpeed;
}
void TargetCamera::processKeyBoard(const Window& window, float deltaTime) {
  float cameraSpeed = 2.f * std::numbers::pi * movementSpeed * deltaTime;

  if (glfwGetKey(window.window(), GLFW_KEY_A) == GLFW_PRESS)
    yaw -= cameraSpeed;
  if (glfwGetKey(window.window(), GLFW_KEY_D) == GLFW_PRESS)
    yaw += cameraSpeed;
  if (glfwGetKey(window.window(), GLFW_KEY_W) == GLFW_PRESS)
    pitch += cameraSpeed;
  if (glfwGetKey(window.window(), GLFW_KEY_S) == GLFW_PRESS)
    pitch -= cameraSpeed;

  if (pitch > 89.0f)
    pitch = 89.0f;
  if (pitch < -89.0f)
    pitch = -89.0f;

  updateCameraVectors();
}
void TargetCamera::updateCameraVectors() {
  position.x = targetPosition.x + distance * cos(glm::radians(yaw)) * cos(
      glm::radians(pitch));
  position.y = targetPosition.y + distance * sin(glm::radians(pitch));
  position.z = targetPosition.z + distance * sin(glm::radians(yaw)) * cos(
      glm::radians(pitch));
  glm::vec3 front = glm::normalize(targetPosition - position);

  glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
  glm::vec3 right = glm::normalize(glm::cross(front, worldUp));

  up = glm::normalize(glm::cross(right, front));
}
}
