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
  std::unique_ptr<ExtGuiWrapper> optional_ext_gui{};
};

struct OpenglGui : Resource {
  using RenderLoop = std::function<void()>;
  using LoopCondition = std::function<bool()>;
  explicit OpenglGui(GuiOption& option) {
    auto& [width, height, title, optional_ext_gui] = option;
    window = std::make_unique<Window>(width, height, title);
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
      // TODO: ERROR
      std::cerr << "Failed to initialize GLAD\n";
      exit(-1);
    }
    ext_gui = std::move(optional_ext_gui);
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
  std::unique_ptr<ExtGuiWrapper> ext_gui{};
  ~OpenglGui() = default;
};

}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_OGL_GUI_H_
