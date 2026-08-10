#pragma once

// Common result / statistics types shared by every global solver family
// (Newton + IPC, Projective Dynamics). Kept in a dedicated header so the
// per-family solver headers can include it without forming cycles.

#include <map>
#include <string>
#include <vector>

namespace ksk::runtime {

struct RuntimeStepResult {
  int iterations = 0;
  double finalGradientNorm = 0.0;
  double finalStepNorm = 0.0;
  bool converged = false;
  std::string terminationReason;
};

class GlobalSolverStatsCollector {
 public:
  using StatMap = std::map<std::string, double>;

  void beginFrame();
  void clear() noexcept;

  void frameSet(std::string key, double value);
  void frameAdd(std::string key, double value);
  void frameMax(std::string key, double value);
  void frameMin(std::string key, double value);
  void frameAverage(std::string key, double value);

  void globalSet(std::string key, double value);
  void globalAdd(std::string key, double value);
  void globalMax(std::string key, double value);
  void globalMin(std::string key, double value);
  void globalAverage(std::string key, double value);

  [[nodiscard]] const StatMap& global() const noexcept { return global_; }
  [[nodiscard]] const std::vector<StatMap>& frames() const noexcept {
    return frames_;
  }

 private:
  static void add(StatMap& stats, std::string key, double value);
  static void max(StatMap& stats, std::string key, double value);
  static void min(StatMap& stats, std::string key, double value);
  static void average(StatMap& stats, std::map<std::string, int>& counts,
                      std::string key, double value);

  [[nodiscard]] StatMap& currentFrame();

  StatMap global_;
  std::map<std::string, int> global_average_counts_;
  std::vector<StatMap> frames_;
  std::vector<std::map<std::string, int>> frame_average_counts_;
};

}  // namespace ksk::runtime
