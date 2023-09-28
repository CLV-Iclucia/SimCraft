//
// Created by creeper on 23-8-13.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_TIMER_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_TIMER_H_
#include <Core/core.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
namespace core {
// this timer is from CIS 565
class Timer {
public:
  Timer() {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
  }
  ~Timer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }
  void startCpuTimer() {
    if (cpu_timer_started) {
      std::cerr << "Error: CPU timer already started." << std::endl;
      exit(-1);
    }
    cpu_timer_started = true;
    cpu_start_time = std::chrono::high_resolution_clock::now();
  }
  void stopCpuTimer() {
    cpu_end_time = std::chrono::high_resolution_clock::now();
    if (!cpu_timer_started) {
      std::cerr << "Error: CPU timer not started." << std::endl;
      exit(-1);
    }
    std::chrono::duration<double, std::milli> duro =
        cpu_end_time - cpu_start_time;
    cpu_elapsed_time = static_cast<float>(duro.count());
    cpu_timer_started = false;
  }
  void startGpuTimer() {
    if (gpu_timer_started) {
      std::cerr << "Error: GPU timer already started." << std::endl;
      exit(-1);
    }
    gpu_timer_started = true;
    cudaEventRecord(start_event);
  }
  void stopGpuTimer() {
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    if (!gpu_timer_started) {
      std::cerr << "Error: GPU timer not started." << std::endl;
      exit(-1);
    }
    gpu_timer_started = false;
    cudaEventElapsedTime(&gpu_elapsed_time, start_event, stop_event);
  }
  float CpuElapsedTime() const { return cpu_elapsed_time; }
  float GpuElapsedTime() const { return gpu_elapsed_time; }
  void logCpuElapsedTime(const char *event) const {
    std::printf("%s: %lf ms\n", event, CpuElapsedTime());
  }
  void logGpuElapsedTime(const char *event) const {
    std::printf("%s: %lf ms\n", event, GpuElapsedTime());
  }

private:
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  std::chrono::high_resolution_clock::time_point cpu_start_time;
  std::chrono::high_resolution_clock::time_point cpu_end_time;
  bool cpu_timer_started = false;
  bool gpu_timer_started = false;
  float cpu_elapsed_time = 0.0;
  float gpu_elapsed_time = 0.0;
};
} // namespace core
#endif // SIMCRAFT_CORE_INCLUDE_CORE_TIMER_H_
