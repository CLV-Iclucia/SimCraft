//
// Created by creeper on 3/15/24.
//

#ifndef REBUILD_SURFACE_H
#define REBUILD_SURFACE_H
#include <Core/mesh.h>
#include <FluidSim/cpu/sdf.h>
namespace fluid {
void rebuildSurface(core::Mesh& mesh, const SDF<3>& sdf);
}
#endif //MARCHING_CUBES_H
