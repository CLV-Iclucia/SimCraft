#include <Renderer/frame-sink.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ksk::renderer {

namespace {

/// Built-in dry-run sink: consumes every frame and reports the count.
///
/// It validates the offline pipeline end-to-end without doing any actual
/// rendering. The real offline renderer (software rasterizer, path tracer,
/// external tool, ...) is intentionally left out — plug yours in via
/// renderer::registerFrameSink() and select it with
/// RendererConfig::offline::sinkName.
class NullFrameSink final : public FrameSink {
public:
  bool initialize(const FrameSinkConfig& config) override {
    m_config = config;
    m_frames = 0;
    return true;
  }

  bool renderFrame(const SceneProxy&, const RenderFrameContext&) override {
    ++m_frames;
    return true;
  }

  bool finish() override {
    std::cout << "[NullFrameSink] dry-run consumed " << m_frames
              << " frames\n";
    return true;
  }

private:
  FrameSinkConfig m_config;
  int m_frames = 0;
};

using Registry = std::unordered_map<std::string, FrameSinkCreator>;

Registry& registry() {
  static Registry r;
  return r;
}

struct RegisterBuiltins {
  RegisterBuiltins() {
    registerFrameSink("null", [] { return std::make_unique<NullFrameSink>(); });
  }
};
RegisterBuiltins g_registerBuiltins;

} // namespace

void registerFrameSink(std::string_view name, FrameSinkCreator creator) {
  registry()[std::string(name)] = std::move(creator);
}

std::unique_ptr<FrameSink> createFrameSink(std::string_view name) {
  auto it = registry().find(std::string(name));
  if (it == registry().end()) return nullptr;
  return it->second();
}

std::vector<std::string> registeredFrameSinkNames() {
  std::vector<std::string> names;
  names.reserve(registry().size());
  for (const auto& [name, _] : registry()) names.push_back(name);
  std::sort(names.begin(), names.end());
  return names;
}

} // namespace ksk::renderer
