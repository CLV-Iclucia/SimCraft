//
// Created by creeper on 2/14/24.
//

#ifndef TIME_STEPPERS_H
#define TIME_STEPPERS_H
#include <HairSim/hair-sim.h>
#include <HairSim/system.h>
#include <Core/properties.h>

namespace hairsim {
class TimeStepper : core::NonCopyable {
  public:
  virtual void step(Real dt, System* system) = 0;
  virtual ~TimeStepper() = default;
};

class StableConstrainedSolver : public TimeStepper {
};

class CodimIPC : public TimeStepper {

};
}

#endif //TIME_STEPPERS_H
