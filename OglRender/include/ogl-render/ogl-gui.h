//
// Created by creeper on 6/17/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_OGL_GUI_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_OGL_GUI_H_
#include <ogl-render/window.h>
#include <memory>
#include <concepts>
#include <functional>
#include <optional>
namespace opengl {

struct GuiOption {
  int width{}, height{};
  std::string_view title{};
};

struct OpenglGui : Resource {
  using RenderLoop = std::function<void()>;
  using LoopCondition = std::function<bool()>;
  explicit OpenglGui(const GuiOption& option) {
    auto& [width, height, title] = option;
    window = std::make_unique<Window>(width, height, title);
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
      throw std::runtime_error("Failed to initialize GLAD");
  }
  void render(const RenderLoop &loop_body,
              const LoopCondition &condition = {[]() -> bool { return true; }}) const {
    while (!window->shouldClose() && condition()) {
      window->preRender();
      loop_body();
      window->postRender();
    }
  }
  std::unique_ptr<Window> window{};
};

}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_OGL_GUI_H_
