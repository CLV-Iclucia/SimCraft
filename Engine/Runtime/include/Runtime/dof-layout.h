#pragma once

#include <optional>
#include <vector>

#include <Runtime/types.h>

namespace ksk::runtime {

struct DofRange {
  SubsystemIdx subsystem = -1;
  int scalarOffset = 0;
  int scalarCount = 0;
  int blockSize = 0;
};

struct DofLayout {
  int totalScalars = 0;
  std::vector<DofRange> ranges;

  [[nodiscard]] DofRange appendRange(SubsystemIdx subsystem,
                                     int scalarCount,
                                     int blockSize = 0);
  [[nodiscard]] std::optional<DofRange> findRange(SubsystemIdx subsystem) const;
};

}  // namespace ksk::runtime
