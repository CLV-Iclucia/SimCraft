//
// Created by creeper on 23-8-13.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_ANIMATION_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_ANIMATION_H_
#include <Core/core.h>
namespace core {
class Frame final {
public:
  Index idx = 0;
  Real dt = 1.0 / 100.0;
  Real currentTime() const { return idx * dt; }
  void onAdvance() { idx++; }
};

class Animation {
  public:
  virtual ~Animation() = default;
  virtual void step(Frame &frame) = 0;
};
}
#endif // SIMCRAFT_CORE_INCLUDE_CORE_ANIMATION_H_
