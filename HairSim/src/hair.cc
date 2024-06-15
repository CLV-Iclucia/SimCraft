//
// Created by creeper on 23-8-9.
//
#include <HairSim/hair.h>
#include <HairSim/system.h>

namespace hairsim {
Hair::Hair(System* system_, int idx, RefConfig ref_config) : system(system_),
  ref(std::move(ref_config)),
  m_q(system->q.data() + idx * (4 * system->numVerticesPerHair() - 1),
      4 * system->numVerticesPerHair() - 1),
  m_qdot(system->qdot.data() + idx * (4 * system->numVerticesPerHair() - 1),
         4 * system->numVerticesPerHair() - 1) {
  for (int i = 0; i < system->numVerticesPerHair(); i++) {
    pos(i) = ref.ref_pos[i];
    vel(i) = Vec3d::Zero();
  }
  // next compute the reference configuration

}
} // namespace hairsim