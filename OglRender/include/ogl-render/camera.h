//
// Created by creeper on 10/25/23.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_H_
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>

namespace opengl {
struct CameraParameters {
  glm::vec3 position;
  glm::vec3 up;
  glm::vec3 right;
  float yaw_in_deg;
  float pitch_in_deg;
  float fov_in_deg;
  float aspectRatio;
  float nearPlane;
  float farPlane;
  void selfCheck() const {
    assert(nearPlane < farPlane);
    assert(fov_in_deg > 0.0f && fov_in_deg < 180.0f);
    assert(aspectRatio > 0.0f);
  }
};

inline CameraParameters defaultCameraParams() {
  return CameraParameters{
      .position = glm::vec3(0.0f, 0.0f, 3.0f),
      .up = glm::vec3(0.0f, 1.0f, 0.0f),
      .right = glm::vec3(1.0f, 0.0f, 0.0f),
      .yaw_in_deg = -90.0f,
      .pitch_in_deg = 0.0f,
      .fov_in_deg = 45.0f,
      .aspectRatio = 16.0f / 9.0f,
      .nearPlane = 0.1f,
      .farPlane = 100.0f
  };
}

struct Camera {
  Camera() : m_params(defaultCameraParams()) {}
  explicit Camera(const CameraParameters &params) : m_params(params) {
    params.selfCheck();
  }
  [[nodiscard]] glm::mat4 viewMatrix() const {
    return glm::lookAt(m_params.position, m_params.position + front(), m_params.up);
  }
  [[nodiscard]] glm::mat4 perspectiveProjectionMatrix() const {
    return glm::perspective(fovInRad(), m_params.aspectRatio, m_params.nearPlane, m_params.farPlane);
  }
  [[nodiscard]] const glm::vec3 &position() const {
    return m_params.position;
  }
  [[nodiscard]] glm::vec3 front() const {
    glm::vec3 front;
    front.x = cos(glm::radians(m_params.yaw_in_deg)) * cos(glm::radians(m_params.pitch_in_deg));
    front.y = sin(glm::radians(m_params.pitch_in_deg));
    front.z = sin(glm::radians(m_params.yaw_in_deg)) * cos(glm::radians(m_params.pitch_in_deg));
    return glm::normalize(front);
  }
  [[nodiscard]] const glm::vec3 &up() const {
    return m_params.up;
  }
  [[nodiscard]] const glm::vec3 &right() const {
    return m_params.right;
  }
  [[nodiscard]] float fovInDeg() const {
    return m_params.fov_in_deg;
  }
  [[nodiscard]] float fovInRad() const {
    return glm::radians(m_params.fov_in_deg);
  }
  [[nodiscard]] float aspectRatio() const {
    return m_params.aspectRatio;
  }
  [[nodiscard]] float nearPlane() const {
    return m_params.nearPlane;
  }
  [[nodiscard]] float yawInDeg() const {
    return m_params.yaw_in_deg;
  }
  [[nodiscard]] float pitchInDeg() const {
    return m_params.pitch_in_deg;
  }
 protected:
  template<typename Derived>
  friend
  struct CameraController;
  CameraParameters m_params;
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_H_