//
// sync.h
// Sync primitives: Fence (host-visible) and Semaphore (queue-to-queue).
// Single-ownership, NOT ref-counted (see docs/rhi-plan.md §3.7 / R17).
//

#pragma once

#include <Core/properties.h>

namespace sim::rhi {

class Fence : public sim::core::NonCopyable {
 public:
  virtual ~Fence() = default;

 protected:
  Fence() = default;
};

class Semaphore : public sim::core::NonCopyable {
 public:
  virtual ~Semaphore() = default;

 protected:
  Semaphore() = default;
};

}  // namespace sim::rhi
