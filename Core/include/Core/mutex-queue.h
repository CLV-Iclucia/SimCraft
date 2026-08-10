#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

namespace ksk::core {

template <typename T>
class mutex_queue {
 public:
  explicit mutex_queue(std::size_t max_size = 16) : max_size_(max_size) {}

  void push(T value)
  {
    std::unique_lock lock(mutex_);
    producer_cv_.wait(lock, [&] {
      return queue_.size() < max_size_ || shutdown_;
    });
    if (shutdown_) {
      return;
    }
    queue_.push(std::move(value));
    consumer_cv_.notify_one();
  }

  std::optional<T> wait_pop()
  {
    std::unique_lock lock(mutex_);
    consumer_cv_.wait(lock, [&] {
      return !queue_.empty() || shutdown_;
    });
    if (shutdown_ && queue_.empty()) {
      return std::nullopt;
    }
    T value = std::move(queue_.front());
    queue_.pop();
    producer_cv_.notify_one();
    return value;
  }

  std::optional<T> try_pop()
  {
    std::lock_guard lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    T value = std::move(queue_.front());
    queue_.pop();
    producer_cv_.notify_one();
    return value;
  }

  void shutdown()
  {
    std::lock_guard lock(mutex_);
    shutdown_ = true;
    consumer_cv_.notify_all();
    producer_cv_.notify_all();
  }

  [[nodiscard]] std::size_t size() const
  {
    std::lock_guard lock(mutex_);
    return queue_.size();
  }

  [[nodiscard]] bool empty() const
  {
    std::lock_guard lock(mutex_);
    return queue_.empty();
  }

  [[nodiscard]] bool is_shutdown() const
  {
    std::lock_guard lock(mutex_);
    return shutdown_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable consumer_cv_;
  std::condition_variable producer_cv_;
  std::queue<T> queue_;
  std::size_t max_size_;
  bool shutdown_ = false;
};

}  // namespace ksk::core
