#include <Renderer/offline-renderer.h>

#include <Renderer/frame-sink.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <iostream>

namespace ksk::renderer {

OfflineRenderer::~OfflineRenderer() = default;

void OfflineRenderer::initialize(const RendererConfig& config) {
  m_config = config;
  const auto& oc = config.offline;

  m_orbitYawDeg = oc.orbitYawDeg;

  if (oc.sinkName.empty()) {
    std::cerr << "[OfflineRenderer] no sink selected\n";
    return;
  }

  m_sink = createFrameSink(oc.sinkName);
  if (!m_sink) {
    std::cerr << "[OfflineRenderer] unknown sink '" << oc.sinkName << "'.\n"
              << "  registered sinks:";
    for (const auto& name : registeredFrameSinkNames())
      std::cerr << " " << name;
    std::cerr << "\n  register your own via renderer::registerFrameSink().\n";
    return;
  }

  FrameSinkConfig sinkConfig{
      .outputDir = oc.outputDir,
      .width = oc.outputWidth,
      .height = oc.outputHeight,
      .imageFormat = oc.imageFormat,
      .startFrame = oc.frameStart,
  };

  if (!m_sink->initialize(sinkConfig)) {
    std::cerr << "[OfflineRenderer] failed to initialize sink\n";
    m_sink.reset();
    return;
  }

  std::cout << "[OfflineRenderer] initialized sink '" << oc.sinkName
            << "' -> " << oc.outputDir << " (" << oc.outputWidth << "x"
            << oc.outputHeight << ")\n";
}

void OfflineRenderer::renderLoop() {
  initialize(m_config);
  if (!m_sink) {
    cleanup();
    return;
  }

  const auto& oc = m_config.offline;
  int target = 0;  // index of the frame currently being popped from the queue

  while (true) {
    auto frame = m_queue.wait_pop();
    if (!frame) break;  // producer signalled shutdown and queue is drained

    if (target < oc.frameStart) {
      ++target;  // consume but skip writing (before frameStart)
      continue;
    }
    if (oc.frameEnd >= 0 && target > oc.frameEnd) {
      // Reached the requested end: stop the producer (unblocks any pending
      // push on the bounded queue) and finish the bake early.
      shutdown();
      break;
    }

    // frame is std::optional<std::unique_ptr<SceneProxy>>; dereference twice
    // to reach the SceneProxy itself (first the optional, then the pointer).
    applyCamera(**frame);
    drawFrame(**frame);
    ++m_renderedFrames;
    ++target;
  }

  cleanup();
  std::cout << "[OfflineRenderer] baked " << m_renderedFrames << " frames\n";
}

void OfflineRenderer::drawFrame(const SceneProxy& scene) {
  if (!m_sink) return;
  m_sink->renderFrame(scene, makeFrameContext(scene));
}

RenderFrameContext OfflineRenderer::makeFrameContext(const SceneProxy& scene) const {
  const auto& cam = scene.camera;
  const auto& oc = m_config.offline;

  const float aspect =
      (oc.outputHeight > 0) ? static_cast<float>(oc.outputWidth) /
                                  static_cast<float>(oc.outputHeight)
                            : 1.0f;

  return RenderFrameContext{
      .view = glm::lookAt(cam.position, cam.target, cam.up),
      .projection = glm::perspective(glm::radians(cam.fov), aspect,
                                     cam.nearPlane, cam.farPlane),
      .cameraPosition = cam.position,
      .cameraTarget = cam.target,
      .cameraDistance = glm::distance(cam.position, cam.target),
      .framebufferWidth = oc.outputWidth,
      .framebufferHeight = oc.outputHeight,
  };
}

void OfflineRenderer::applyCamera(SceneProxy& scene) {
  const auto& oc = m_config.offline;

  if (oc.useSceneCamera) return;  // keep the per-frame camera from the proxy

  if (oc.fixedCamera) {
    scene.camera = *oc.fixedCamera;
    return;
  }

  // Orbit camera.
  if (oc.orbitAutoRotateDegPerFrame != 0.0f)
    m_orbitYawDeg += oc.orbitAutoRotateDegPerFrame;

  const float yaw = glm::radians(m_orbitYawDeg);
  const float pitch = glm::radians(oc.orbitPitchDeg);
  scene.camera.target = oc.orbitTarget;
  scene.camera.position = oc.orbitTarget + glm::vec3(
      oc.orbitRadius * std::cos(pitch) * std::cos(yaw),
      oc.orbitRadius * std::sin(pitch),
      oc.orbitRadius * std::cos(pitch) * std::sin(yaw));
}

void OfflineRenderer::cleanup() {
  if (m_sink) {
    m_sink->finish();
    m_sink.reset();
  }
}

} // namespace ksk::renderer
