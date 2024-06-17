//
// Created by creeper on 6/17/24.
//
#include <ogl-render/resource-handles.h>
#include <ogl-render/ogl-gui.h>

using namespace opengl;

int main() {
  std::unique_ptr<OpenglGui> gui = std::make_unique<OpenglGui>(1024, 1024, "Mesh View");
  gui->render([&]() {
    glCheckError(glClearColor(0.1f, 1.0f, 1.0f, 1.0f));
    glCheckError(glClear(GL_COLOR_BUFFER_BIT));
  });
}