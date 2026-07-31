#pragma once

#include <Runtime/contact-detection.h>

namespace ksk::runtime {

enum class GlobalSolveMode {
  Decoupled,
  Coupled,
};

struct GlobalSolverConfig {
  int maxNewtonIterations = 25;
  int maxPcgIterations = 200;
  int maxLineSearchIterations = 12;
  double newtonGradientTolerance = 1.0e-8;
  double newtonStepTolerance = 1.0e-2;
  double lineSearchArmijo = 1.0e-4;
  double ccdStepSizeScale = 0.9;
  double pcgTolerance = 1.0e-8;
  bool enablePCGWarmStart = true;
  bool enableContact = true;
  GlobalSolveMode solveMode = GlobalSolveMode::Decoupled;
  double coupledDiagonalRegularization = 1.0e-12;
  ContactDetectionStorage contactDetectionStorage =
      ContactDetectionStorage::Host;
  double contactBarrierDistance = 0.0;
  double contactStiffness = 1e6;
};

}  // namespace ksk::runtime
