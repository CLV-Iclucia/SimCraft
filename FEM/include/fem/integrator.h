//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_INTEGRATOR_H_
#define SIMCRAFT_FEM_INCLUDE_INTEGRATOR_H_
#include <fem/system.h>
namespace fem {
struct Integrator {
  virtual void step(Real dt) = 0;
  explicit Integrator(System &system_) : system_to_integrate(system_) {}
  System& system_to_integrate;
  [[nodiscard]] System& system() const { return system_to_integrate; }
  virtual ~Integrator() = default;
};

}
#endif //SIMCRAFT_FEM_INCLUDE_INTEGRATOR_H_
