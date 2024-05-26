//
// Created by creeper on 23-8-13.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_TIMER_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_TIMER_H_
#include <Core/core.h>
#include <chrono>
#include <iostream>
#include <concepts>
#include <utility>
namespace core {
template <typename T>
concept Timer = requires(T& timer, const std::string& event) {
  timer.start();
  timer.stop();
  { timer.elapsedTime } -> std::convertible_to<float>;
  timer.logElapsedTime(event);
};

// this timer is from CIS 565
class CpuTimer {
public:
  void start() {
    if (cpu_timer_started) {
      std::cerr << "Error: CPU timer already started." << std::endl;
      exit(-1);
    }
    cpu_timer_started = true;
    cpu_start_time = std::chrono::high_resolution_clock::now();
  }
  void stop() {
    cpu_end_time = std::chrono::high_resolution_clock::now();
    if (!cpu_timer_started) {
      std::cerr << "Error: CPU timer not started." << std::endl;
      exit(-1);
    }
    std::chrono::duration<double, std::milli> duro = cpu_end_time - cpu_start_time;
    cpu_elapsed_time = static_cast<float>(duro.count());
    cpu_timer_started = false;
  }
  float elapsedTime() const { return cpu_elapsed_time; }
  void logElapsedTime(const std::string& event) const {
    std::cout << std::format("{}: {} ms\n", event, elapsedTime());
  }

private:
  std::chrono::high_resolution_clock::time_point cpu_start_time;
  std::chrono::high_resolution_clock::time_point cpu_end_time;
  bool cpu_timer_started = false;
  float cpu_elapsed_time = 0.0;
};

template <Timer T>
struct TimerGuard {
  explicit TimerGuard(std::string event_name) : event(std::move(event_name)) { timer.start(); }
  TimerGuard() {
    timer.start();
  }
  ~TimerGuard() {
    timer.stop();
    timer.logElapsedTime(event);
  }
  T timer;
  std::string event;
};
} // namespace core
#endif // SIMCRAFT_CORE_INCLUDE_CORE_TIMER_H_
