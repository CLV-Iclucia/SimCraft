//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_INTEGRATOR_H_
#define SIMCRAFT_FEM_INCLUDE_INTEGRATOR_H_
namespace fem {
struct Integrator {
  virtual void step(Real dt) = 0;
};

struct ImplicitEuler : public Integrator {
  void step(Real dt) override;
};
}
#endif //SIMCRAFT_FEM_INCLUDE_INTEGRATOR_H_
