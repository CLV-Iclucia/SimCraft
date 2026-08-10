#pragma once

#include <glm/glm.hpp>

namespace ksk::renderer {

/// View/projection state shared by realtime backends and offline sinks.
///
/// Both the interactive GL backend and the offline baking pipeline derive
/// their matrices from the same CameraState carried by each SceneProxy, so
/// an offline renderer sees exactly what the realtime viewer would show.
struct RenderFrameContext {
  glm::mat4 view{1.0f};
  glm::mat4 projection{1.0f};
  glm::vec3 cameraPosition{0.0f};
  glm::vec3 cameraTarget{0.0f};
  float cameraDistance = 1.0f;
  int framebufferWidth = 1;
  int framebufferHeight = 1;
};

} // namespace ksk::renderer
