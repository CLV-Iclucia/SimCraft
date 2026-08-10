#pragma once
#include <Core/mutex-queue.h>
#include <Renderer/scene-proxy.h>
#include <memory>
#include <thread>
#include <functional>
#include <atomic>
#include <optional>

namespace ksk::renderer {

enum class RenderBackendKind {
  Default,
  OpenGL,
  RHI,
};

/// How a Renderer consumes the SceneProxy frame queue.
enum class RenderMode {
  Realtime,  // live window + interactive orbit camera
  Offline,   // deterministic offline baking, headless
};

/// Configuration for offline baking mode (used when RenderMode::Offline).
struct OfflineRendererConfig {
  std::string outputDir = "bake";          // where baked frames are written
  int outputWidth = 1920;
  int outputHeight = 1080;
  std::string imageFormat = "bmp";         // passed through to the FrameSink
  std::string sinkName = "null";           // registered FrameSink name
  int frameStart = 0;                      // first frame index to write
  int frameEnd = -1;                       // last frame to write; -1 = all

  /// Camera: by default each SceneProxy's embedded camera is respected,
  /// which is exactly what the realtime renderer shows.
  bool useSceneCamera = true;

  /// Fixed camera for the whole bake (overrides useSceneCamera).
  std::optional<CameraState> fixedCamera;

  /// Orbit camera (used when useSceneCamera == false and no fixedCamera).
  glm::vec3 orbitTarget{0.0f, 0.0f, 0.0f};
  float orbitRadius = 5.0f;
  float orbitYawDeg = -60.0f;
  float orbitPitchDeg = 25.0f;
  float orbitAutoRotateDegPerFrame = 0.0f;  // 0 = static
};

struct RendererConfig {
  int windowWidth = 1280;
  int windowHeight = 720;
  std::string windowTitle = "Kiseki";
  bool vsync = true;
  bool headless = false;
  RenderBackendKind backend = RenderBackendKind::Default;
  RenderMode mode = RenderMode::Realtime;
  OfflineRendererConfig offline;  // used when mode == RenderMode::Offline
};

class Renderer {
public:
  virtual ~Renderer() = default;

  using FrameQueue = core::mutex_queue<std::unique_ptr<SceneProxy>>;

  [[nodiscard]] FrameQueue& queue() { return m_queue; }

  void setConfig(const RendererConfig& config) { m_config = config; }

  void runOnCurrentThread() {
    m_running = true;
    renderLoop();
  }

  void shutdown() {
    m_running = false;
    m_queue.shutdown();
  }

  [[nodiscard]] bool isRunning() const { return m_running; }

  void setInputCallback(std::function<void(CameraState&)> cb) { m_inputCallback = std::move(cb); }

protected:
  virtual void initialize(const RendererConfig& config) = 0;

  /// Frame consumption loop. The base implementation is the realtime loop
  /// (interactive camera + present). Offline renderers override this with a
  /// deterministic drain loop that terminates when the queue is exhausted.
  virtual void renderLoop() {
    initialize(m_config);

    std::unique_ptr<SceneProxy> currentFrame;

    while (m_running) {
      if (auto newFrame = m_queue.try_pop())
        currentFrame = std::move(*newFrame);

      if (!currentFrame) {
        auto frame = m_queue.wait_pop();
        if (frame)
          currentFrame = std::move(*frame);
        if (!currentFrame) break; // shutdown
      }

      if (m_inputCallback)
        m_inputCallback(m_camera);

      currentFrame->camera = m_camera;
      drawFrame(*currentFrame);

      if (!pollAndSwap()) {
        m_running = false;
        m_queue.shutdown();
      }
    }

    cleanup();
  }

  virtual void drawFrame(const SceneProxy& scene) = 0;
  virtual void cleanup() = 0;
  virtual bool pollAndSwap() = 0;

  RendererConfig m_config;
  CameraState m_camera;
  std::function<void(CameraState&)> m_inputCallback;
  FrameQueue m_queue;

private:
  std::atomic<bool> m_running{false};
};

std::unique_ptr<Renderer> createRenderer(const RendererConfig& config = {});

} // namespace ksk::renderer
