//
// Created by creeper on 9/8/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_VISUALIZER_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_VISUALIZER_H_
#include <ogl-render/drawable-mesh.h>

namespace fem {
struct System;
struct Visualizer {
  const System& system;
  opengl::DrawableMesh mesh;
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_VISUALIZER_H_
