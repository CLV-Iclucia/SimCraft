//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_INTEGRATOR_H_
#define SIMCRAFT_FEM_INCLUDE_INTEGRATOR_H_
#include <fem/system.h>
namespace fem {
struct Integrator {
  virtual void step(System& system, Real dt) = 0;
};

}
#endif //SIMCRAFT_FEM_INCLUDE_INTEGRATOR_H_
