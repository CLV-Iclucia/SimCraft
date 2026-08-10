#pragma once

#include <Runtime/contact-detection.h>
#include <Runtime/types.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "solver-config.generated.h"

namespace ksk::runtime {

// Which time-step formulation the global solver should use.
//
// NewtonIPC: nonlinear objective, gradients, Hessian/operator, CCD, line
//            search, IPC barrier contact.
// ProjectiveDynamics: local/global iterations, local projections, quadratic
//            global solve, optional post-processed collision handling.
enum class GlobalSolverType {
  NewtonIPC,
  ProjectiveDynamics,
};

// Which matrix ownership model the linear solve should use.
//
// Coupled: the global solver assembles/owns the complete matrix (or global
//          operator) and solves the coupled system directly.
// Decoupled: subsystems own local block solves/operators, usually used by the
//          global solver as preconditioners.
enum class LinearSolveMode {
  Decoupled,
  Coupled,
};

struct FrictionMaterialPair {
  PhysicalMaterialId first = kDefaultPhysicalMaterial;
  PhysicalMaterialId second = kDefaultPhysicalMaterial;
  double coefficient = 0.0;
};

struct FrictionMaterialTable {
  std::vector<FrictionMaterialPair> entries;

  void clear() noexcept
  {
    entries.clear();
  }

  [[nodiscard]] bool empty() const noexcept
  {
    return entries.empty();
  }

  void set(PhysicalMaterialId first,
           PhysicalMaterialId second,
           double coefficient)
  {
    const auto found = std::ranges::find_if(
        entries,
        [first, second](const FrictionMaterialPair& entry) {
          return matches(entry, first, second);
        });
    if (found != entries.end()) {
      found->coefficient = coefficient;
      return;
    }
    entries.push_back({
        .first = first,
        .second = second,
        .coefficient = coefficient,
    });
  }

  [[nodiscard]] std::optional<double> lookup(
      PhysicalMaterialId first,
      PhysicalMaterialId second) const
  {
    const auto found = std::ranges::find_if(
        entries,
        [first, second](const FrictionMaterialPair& entry) {
          return matches(entry, first, second);
        });
    if (found == entries.end()) {
      return std::nullopt;
    }
    return found->coefficient;
  }

 private:
  [[nodiscard]] static bool matches(const FrictionMaterialPair& entry,
                                    PhysicalMaterialId first,
                                    PhysicalMaterialId second) noexcept
  {
    return (entry.first == first && entry.second == second) ||
           (entry.first == second && entry.second == first);
  }
};

// --- Base config: parameters common to every solver family. ---
//
// Runtime code branches on the dynamic type (via RF::Cast) to reach
// solver-specific parameters (NewtonIpcSolverConfig / ProjectiveDynamicsSolverConfig).
struct KLASS() GlobalSolverConfig {
  KLASS_GENERATED_BODY()

  GlobalSolverConfig() = default;
  virtual ~GlobalSolverConfig() = default;

  int maxNewtonIterations = 25;  // shared: Newton uses it; PD has its own loop
  double newtonGradientTolerance = 1.0e-8;
  double newtonStepTolerance = 1.0e-2;
  bool enableContact = true;
  bool enableFriction = false;
  GlobalSolverType solverType = GlobalSolverType::NewtonIPC;
  LinearSolveMode solveMode = LinearSolveMode::Decoupled;
};

// --- Newton + IPC specific parameters. ---
struct KLASS(parent = GlobalSolverConfig) NewtonIpcSolverConfig
    : GlobalSolverConfig {
  KLASS_GENERATED_BODY()

  NewtonIpcSolverConfig() = default;

  int maxPcgIterations = 200;
  int maxLineSearchIterations = 12;
  double lineSearchArmijo = 1.0e-4;
  double ccdStepSizeScale = 0.95;
  double pcgTolerance = 1.0e-8;
  bool enablePCGWarmStart = true;
  double coupledDiagonalRegularization = 1.0e-12;
  ContactDetectionStorage contactDetectionStorage =
      ContactDetectionStorage::Host;
  double contactBarrierDistance = 0.0;
  double contactStiffness = 1e6;
  double frictionCoefficient = 0.0;
  double frictionEpsilon = 1.0e-6;
  FrictionMaterialTable frictionTable;
};

// --- Projective Dynamics specific parameters. ---
struct KLASS(parent = GlobalSolverConfig) ProjectiveDynamicsSolverConfig
    : GlobalSolverConfig {
  KLASS_GENERATED_BODY()

  ProjectiveDynamicsSolverConfig() = default;

  int maxPdIterations = 50;
  double pdResidualTolerance = 1.0e-6;

  // Chebyshev acceleration (semi-implicit PD).
  bool enableChebyshev = true;
  int chebyshevWindowSize = 8;

  // Collision resolution (post-processing).
  bool enablePdCollision = false;
  int maxCollisionResolveIterations = 5;
};

// Create a default config of the requested solver family.
[[nodiscard]] inline std::unique_ptr<GlobalSolverConfig> makeSolverConfig(
    GlobalSolverType type)
{
  switch (type) {
    case GlobalSolverType::NewtonIPC:
      return std::make_unique<NewtonIpcSolverConfig>();
    case GlobalSolverType::ProjectiveDynamics:
      return std::make_unique<ProjectiveDynamicsSolverConfig>();
  }
  return std::make_unique<NewtonIpcSolverConfig>();
}

// Default scene configuration: Newton + IPC preserves the legacy behavior.
[[nodiscard]] inline std::unique_ptr<GlobalSolverConfig>
makeDefaultSolverConfig()
{
  return std::make_unique<NewtonIpcSolverConfig>();
}

// Deep clone preserving the dynamic solver config type.
[[nodiscard]] inline std::unique_ptr<GlobalSolverConfig> cloneSolverConfig(
    const GlobalSolverConfig& config)
{
  if (const auto* newton = RF::Cast<NewtonIpcSolverConfig>(&config)) {
    return std::make_unique<NewtonIpcSolverConfig>(*newton);
  }
  if (const auto* pd = RF::Cast<ProjectiveDynamicsSolverConfig>(&config)) {
    return std::make_unique<ProjectiveDynamicsSolverConfig>(*pd);
  }
  return std::make_unique<GlobalSolverConfig>(config);
}

}  // namespace ksk::runtime
