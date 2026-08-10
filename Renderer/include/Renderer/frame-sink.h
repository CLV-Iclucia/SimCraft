#pragma once

#include <Renderer/render-context.h>
#include <Renderer/scene-proxy.h>

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ksk::renderer {

/// Configuration shared by all offline frame sinks.
struct FrameSinkConfig {
  std::string outputDir = "bake";
  int width = 1920;
  int height = 1080;
  std::string imageFormat = "bmp";  // passed through to the sink
  int startFrame = 0;               // first frame index written to disk
};

/// Offline render target.
///
/// Consumes a stream of SceneProxy frames (the same data the realtime
/// renderer shows) and turns them into durable output — an image sequence,
/// a video, a disk cache for a path tracer, or anything else.
///
/// This is the plug-in point for your own offline renderer:
///   1. subclass FrameSink and implement the three virtuals;
///   2. register it:   renderer::registerFrameSink("mytracer", [] {
///                        return std::make_unique<MyTracer>();
///                      });
///   3. select it:     config.offline.sinkName = "mytracer";
class FrameSink {
public:
  virtual ~FrameSink() = default;

  /// Called once before the first frame. Returns false to abort the bake.
  virtual bool initialize(const FrameSinkConfig& config) = 0;

  /// Called once per baked frame, in order, after the camera is resolved.
  virtual bool renderFrame(const SceneProxy& scene,
                           const RenderFrameContext& context) = 0;

  /// Called once after the last frame to finalize output.
  virtual bool finish() = 0;
};

// ─── Factory / plug-in registry ────────────────────────────────────────────

using FrameSinkCreator = std::function<std::unique_ptr<FrameSink>()>;

/// Register a custom FrameSink under a name so it can be selected through
/// OfflineRendererConfig::sinkName. Call once before running the bake.
void registerFrameSink(std::string_view name, FrameSinkCreator creator);

/// Create a sink by registered name. Returns nullptr for unknown names.
std::unique_ptr<FrameSink> createFrameSink(std::string_view name);

/// Names of all currently registered sinks (for diagnostics).
std::vector<std::string> registeredFrameSinkNames();

} // namespace ksk::renderer
