//
// Created by creeper on 6/17/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_EXT_GUI_WRAPPER_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_EXT_GUI_WRAPPER_H_
namespace opengl {
struct ExtGuiWrapper {
  virtual void preRender() = 0;
  virtual void render() = 0;
  virtual void postRender() = 0;
  ~ExtGuiWrapper() = default;
};

struct ImGUIWrapper : ExtGuiWrapper {

};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_OGL_RENDER_EXT_GUI_WRAPPER_H_
