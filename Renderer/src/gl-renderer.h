#pragma once
#include <Renderer/renderer.h>
#include <unordered_map>
#include <string>

struct GLFWwindow;

namespace sim::renderer {

class GLRenderer final : public Renderer {
protected:
  void initialize(const RendererConfig& config) override;
  void drawFrame(const SceneProxy& scene) override;
  void cleanup() override;
  bool pollAndSwap() override;

private:
  struct GLMeshState {
    unsigned int vao = 0, vbo = 0, ebo = 0, nbo = 0;
    int indexCount = 0;
    size_t vertexCount = 0; // 用于检测是否需要重新分配 buffer
  };

  // 按 name 缓存 GPU 资源，SceneProxy 更新时同步
  std::unordered_map<std::string, GLMeshState> m_meshCache;

  // GLFW window handle
  ::GLFWwindow* m_window = nullptr;

  // Shader programs
  unsigned int m_meshShader = 0;
  unsigned int m_wireShader = 0;
  unsigned int m_particleShader = 0;

  // 相机交互状态
  bool m_mousePressed = false;
  double m_lastMouseX = 0, m_lastMouseY = 0;
  float m_yaw = -90.0f;  // 水平角度
  float m_pitch = 0.0f;   // 垂直角度
  float m_distance = 5.0f; // 距离目标的距离

  void uploadMesh(const MeshProxy& mesh);
  void drawMesh(const MeshProxy& mesh);
  void drawWireframe(const WireframeProxy& wf);
  void drawParticles(const ParticleProxy& particles);
  glm::mat4 computeViewMatrix() const;
  glm::mat4 computeProjectionMatrix() const;
  
  // 相机交互
  void setupInputCallbacks();
  void updateCameraFromInput();
};

} // namespace sim::renderer
