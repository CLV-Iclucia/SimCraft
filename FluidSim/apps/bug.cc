//
// Created by creeper on 4/25/24.
//
#include <Core/core.h>
#include <Core/mesh.h>
#include <FluidSim/cpu/rebuild-surface.h>
#include <FluidSim/cpu/sdf.h>

int main() {
  auto sdf = fluid::loadSDF("fluid.sdf");
  core::Mesh mesh;
  fluid::cpu::rebuildSurface(mesh, *sdf);
  if (!core::exportObj("fluid.obj", mesh)) {
    std::cerr << "Failed to save mesh" << std::endl;
    return 1;
  }
}