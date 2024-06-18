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
  glm::vec3 front;
  glm::vec3 up;
  glm::vec3 right;
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

struct Camera {
  explicit Camera(const CameraParameters &params) : m_position(params.position),
                                                    m_front(params.front),
                                                    m_up(params.up),
                                                    m_right(params.right),
                                                    m_fov_in_deg(params.fov_in_deg),
                                                    m_aspect_ratio(params.aspectRatio),
                                                    m_near_plane(params.nearPlane),
                                                    m_far_plane(params.farPlane) {
    params.selfCheck();
  }
  [[nodiscard]] glm::mat4 viewMatrix() const {
    return glm::lookAt(m_position, m_position + m_front, m_up);
  }
  [[nodiscard]] glm::mat4 projectionMatrix() const {
    return glm::perspective(glm::radians(m_fov_in_deg), m_aspect_ratio, m_near_plane, m_far_plane);
  }
  [[nodiscard]] const glm::vec3 &position() const {
    return m_position;
  }
  [[nodiscard]] const glm::vec3 &front() const {
    return m_front;
  }
  [[nodiscard]] const glm::vec3 &up() const {
    return m_up;
  }
  [[nodiscard]] const glm::vec3 &right() const {
    return m_right;
  }
  [[nodiscard]] float fovInDeg() const {
    return m_fov_in_deg;
  }
  [[nodiscard]] float fovInRad() const {
    return glm::radians(m_fov_in_deg);
  }
  [[nodiscard]] float aspectRatio() const {
    return m_aspect_ratio;
  }
  [[nodiscard]] float nearPlane() const {
    return m_near_plane;
  }
 private:
  template <typename Derived>
  friend struct CameraController;
  glm::vec3 m_position;
  glm::vec3 m_front;
  glm::vec3 m_up;
  glm::vec3 m_right;
  float m_fov_in_deg;
  float m_aspect_ratio;
  float m_near_plane;
  float m_far_plane;
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_CAMERA_H_