#pragma once

#include <Renderer/frame-sink.h>
#include <Renderer/renderer.h>

#include <memory>

namespace ksk::renderer {

/// Headless renderer for offline baking.
///
/// Consumes the exact same SceneProxy frame stream as the realtime
/// InteractiveRenderer and feeds every frame to a pluggable FrameSink
/// (selected via RendererConfig::offline::sinkName). No window, no input,
/// no vsync — just deterministic, ordered frame consumption terminated
/// either by the producer signalling shutdown or by `frameEnd`.
class OfflineRenderer final : public Renderer {
public:
  ~OfflineRenderer() override;

protected:
  void initialize(const RendererConfig& config) override;
  void renderLoop() override;
  void drawFrame(const SceneProxy& scene) override;
  void cleanup() override;
  bool pollAndSwap() override { return true; }  // unused in offline mode

private:
  RenderFrameContext makeFrameContext(const SceneProxy& scene) const;
  void applyCamera(SceneProxy& scene);

  std::unique_ptr<FrameSink> m_sink;
  int m_renderedFrames = 0;
  float m_orbitYawDeg = -60.0f;
};

} // namespace ksk::renderer
