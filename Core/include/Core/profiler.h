//
// Created by CreeperIclucia-Vader on 25-5-26.
//

#pragma once

#include <chrono>

namespace sim::core {
struct Profiler {
  Profiler() = default;
  void tick() { m_start = std::chrono::steady_clock::now(); }

  void tock() {
    m_end = std::chrono::steady_clock::now();
  }

  [[nodiscard]] std::chrono::duration<float> duration() const {
    return std::chrono::duration<float>(m_duration);
  }

private:
  std::chrono::steady_clock::time_point m_start{};
  std::chrono::steady_clock::time_point m_end{};
  float m_duration{};
};
} // namespace core