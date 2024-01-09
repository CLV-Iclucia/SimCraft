//
// Created by creeper on 10/25/23.
//

#ifndef JEOCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_H_
#define JEOCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_H_
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>

namespace opengl {
// camera movement for fps style camera
enum class CameraMovement {
  Forward,
  Backward,
  Left,
  Right
};
// Default camera values
static const float YAW = -90.0f;
static const float PITCH = 0.0f;
static const float SPEED = 2.5f;
static const float SENSITIVITY = 0.1f;
static const float ZOOM = 45.0f;
static const float PI = 3.1415926;
class PerspectiveCamera {
  public:
    PerspectiveCamera(float fov = ZOOM, float sensitivity = SENSITIVITY)
      : fov(fov), sensitivity(sensitivity) {
    }

    virtual void processKeyBoard(GLFWwindow* window, float deltaTime) {
    };
    virtual void processMouseMovement(float xoffset, float yoffset) {
    };
    virtual void processMouseScroll(float yoffset) {
    }
    virtual glm::mat4 getViewMatrix() const = 0;
    virtual glm::mat4 getProjectionMatrix(float width, float height) const = 0;

    void setFOV(float fov_) { fov = fov_; }
    float getFOV() const { return fov; }

    void setSensitivity(float sensitivity_) { sensitivity = sensitivity_; }
    float getSensitivity() const { return sensitivity; }
    virtual ~PerspectiveCamera() = default;

  protected:
    float fov;
    float sensitivity;
};

// An camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
// it is fps style
class FpsCamera : public PerspectiveCamera {
  public:
    FpsCamera(float Fov = ZOOM, float Sensitivity = SENSITIVITY,
              float speed = SPEED)
      : PerspectiveCamera(Fov, Sensitivity),
        position(glm::vec3(0.0f, 0.0f, 3.0f)),
        front(glm::vec3(0.0f, 0.0f, -1.0f)), up(glm::vec3(0.0f, 1.0f, 0.0f)),
        yaw(YAW), pitch(PITCH), movementSpeed(speed) {
    }

    void processKeyBoard(GLFWwindow* window, float deltaTime) override {
      float cameraSpeed = movementSpeed * deltaTime;
      if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        position += front * cameraSpeed;
      if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        position -= front * cameraSpeed;
      if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        position -= glm::normalize(glm::cross(front, up)) * cameraSpeed;
      if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        position += glm::normalize(glm::cross(front, up)) * cameraSpeed;
    }

    void processMouseMovement(float xoffset, float yoffset) override {
      xoffset *= sensitivity;
      yoffset *= sensitivity;

      yaw += xoffset;
      pitch += yoffset;

      if (pitch > 89.0f)
        pitch = 89.0f;
      if (pitch < -89.0f)
        pitch = -89.0f;

      updateCameraVectors();
    }

    glm::mat4 getViewMatrix() const override {
      return glm::lookAt(position, position + front, up);
    }

    glm::mat4
    getProjectionMatrix(float screenWidth, float screenHeight) const override {
      return glm::perspective(glm::radians(fov), screenWidth / screenHeight,
                              0.1f, 100.0f);
    }

  private:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    float yaw;
    float pitch;
    float movementSpeed;
    float aspectRatio;

    void updateCameraVectors() {
      glm::vec3 newFront;
      newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
      newFront.y = sin(glm::radians(pitch));
      newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
      front = glm::normalize(newFront);
    }
};
class TargetCamera : public PerspectiveCamera {
  public:
    TargetCamera(glm::vec3 target = glm::vec3(0.0f),
                 float distance = 3.0f,
                 float fov = 45.0f,
                 float sensitivity = 0.1f,
                 float movementSpeed = 2.5f)
      : PerspectiveCamera(fov, sensitivity),
        targetPosition(target),
        distance(distance), yaw(-90.0f), pitch(0.0f),
        movementSpeed(movementSpeed) {
    }

    void processKeyBoard(GLFWwindow* window, float deltaTime) override {
      float cameraSpeed = 2 * PI * movementSpeed * deltaTime;

      if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        yaw -= cameraSpeed;
      }
      if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        yaw += cameraSpeed;
      }
      if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        pitch += cameraSpeed;
      }
      if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        pitch -= cameraSpeed;
      }

      if (pitch > 89.0f)
        pitch = 89.0f;
      if (pitch < -89.0f)
        pitch = -89.0f;

      updateCameraVectors();
    }

    void processMouseMovement(float xoffset, float yoffset) override {
    }

    void processMouseScroll(float yoffset) override {
      distance -= yoffset * sensitivity;
    }

    glm::mat4 getViewMatrix() const override {
      return glm::lookAt(position, targetPosition, up);
    }

    glm::mat4 getProjectionMatrix(float width, float height) const override {
      return glm::perspective(glm::radians(fov), width / height, 0.1f, 100.0f);
    }
    glm::vec3 getPosition() const {
      return position;
    }

  private:
    void updateCameraVectors() {
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
    glm::vec3 targetPosition;
    float distance;
    float yaw;
    float pitch;
    float movementSpeed;
    glm::vec3 position;
    // Position could be calculated based on other parameters
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    // Up vector, could be customizable
};
}
#endif //JEOCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_H_