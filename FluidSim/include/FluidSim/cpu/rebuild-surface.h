//
// Created by creeper on 3/15/24.
//

#ifndef CPU_REBUILD_SURFACE_H
#define CPU_REBUILD_SURFACE_H
#include <Core/mesh.h>
#include <FluidSim/cpu/sdf.h>
namespace fluid::cpu {
void rebuildSurface(core::Mesh& mesh, const SDF<3>& sdf);
}
#endif //CPU_REBUILD_SURFACE_H
