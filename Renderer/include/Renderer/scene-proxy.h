#pragma once
#include <Core/core.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>

namespace sim::renderer {

/// 单个可渲染网格的数据快照 (拥有数据所有权)
struct MeshProxy {
  std::string name;
  std::vector<core::Vec3f> positions;    // 顶点位置 (float, 渲染精度足够)
  std::vector<core::Vec3u> triangles;    // 三角形索引
  std::vector<core::Vec3f> normals;      // 逐顶点法线 (可为空, 渲染时自动计算面法线)
  std::vector<core::Vec3f> colors;       // 逐顶点颜色 (可为空)
  core::Vec3f objectColor{-1.0f};        // Per-object 颜色; 负值 = 使用全局默认色
};

/// 单个可渲染线框/边集
struct WireframeProxy {
  std::string name;
  std::vector<core::Vec3f> positions;
  std::vector<core::Vec2u> edges;
  core::Vec3f color{1.0f, 1.0f, 1.0f};
};

/// 粒子系统快照
struct ParticleProxy {
  std::string name;
  std::vector<core::Vec3f> positions;
  float radius = 0.01f;
  core::Vec3f color{0.2f, 0.5f, 1.0f};
};

/// 相机参数
struct CameraState {
  glm::vec3 position{0, 2, 5};
  glm::vec3 target{0, 0, 0};
  glm::vec3 up{0, 1, 0};
  float fov = 45.0f;        // degrees
  float nearPlane = 0.01f;
  float farPlane = 100.0f;
};

/// 一帧的完整场景快照
struct SceneProxy {
  std::vector<MeshProxy> meshes;
  std::vector<WireframeProxy> wireframes;
  std::vector<ParticleProxy> particles;
  CameraState camera;
  float simulationTime = 0.0f;
  int frameIndex = 0;
};

} // namespace sim::renderer
