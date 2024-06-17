//
// Created by creeper on 6/17/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_OGL_GUI_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_OGL_GUI_H_
#include <ogl-render/window.h>
#include <memory>
#include <concepts>
#include <functional>
namespace opengl {

struct OpenglGui : Resource {
  using RenderLoop = std::function<void()>;
  using LoopCondition = std::function<bool()>;
  OpenglGui(int width, int height, std::string_view title) : window(std::make_unique<Window>(width, height, title)) {
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
      // TODO: ERROR
      std::cerr << "Failed to initialize GLAD\n";
      exit(-1);
    }
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
  ~OpenglGui() = default;
};

}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_OGL_GUI_H_
