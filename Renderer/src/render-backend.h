#pragma once

#include <Renderer/render-context.h>
#include <Renderer/scene-proxy.h>

namespace ksk::renderer {

struct RenderBackendInit {
  void* nativeWindow = nullptr;
  int width = 1;
  int height = 1;
  bool vsync = true;
  bool headless = false;
};

class RenderBackend {
public:
  virtual ~RenderBackend() = default;

  virtual void configureWindow() = 0;
  virtual bool initialize(const RenderBackendInit& init) = 0;
  virtual void renderFrame(const SceneProxy& scene,
                           const RenderFrameContext& context) = 0;
  virtual bool present() = 0;
  virtual void cleanup() = 0;
};

} // namespace ksk::renderer
