//
// Created by creeper on 6/18/24.
//

#ifndef SIMCRAFT_OGLRENDER_INCLUDE_EXT_GUI_IMGUI_IMGUI_WRAPPER_H_
#define SIMCRAFT_OGLRENDER_INCLUDE_EXT_GUI_IMGUI_IMGUI_WRAPPER_H_
#include <ogl-render/ext-gui-wrapper.h>
namespace opengl {
struct ImGUIWrapper final : ExtGuiWrapper {
  void preRender() override;
  void postRender() override;
};
}
#endif //SIMCRAFT_OGLRENDER_INCLUDE_EXT_GUI_IMGUI_IMGUI_WRAPPER_H_
