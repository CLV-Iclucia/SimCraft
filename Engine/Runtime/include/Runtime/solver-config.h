#pragma once

#include <Runtime/contact-detection.h>
#include <Runtime/types.h>

#include <algorithm>
#include <optional>
#include <vector>

namespace ksk::runtime {

enum class GlobalSolveMode {
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

struct GlobalSolverConfig {
  int maxNewtonIterations = 25;
  int maxPcgIterations = 200;
  int maxLineSearchIterations = 12;
  double newtonGradientTolerance = 1.0e-8;
  double newtonStepTolerance = 1.0e-2;
  double lineSearchArmijo = 1.0e-4;
  double ccdStepSizeScale = 0.95;
  double pcgTolerance = 1.0e-8;
  bool enablePCGWarmStart = true;
  bool enableContact = true;
  bool enableFriction = false;
  GlobalSolveMode solveMode = GlobalSolveMode::Decoupled;
  double coupledDiagonalRegularization = 1.0e-12;
  ContactDetectionStorage contactDetectionStorage =
      ContactDetectionStorage::Host;
  double contactBarrierDistance = 0.0;
  double contactStiffness = 1e6;
  double frictionCoefficient = 0.0;
  FrictionMaterialTable frictionTable;
  double frictionEpsilon = 1.0e-6;
};

}  // namespace ksk::runtime
