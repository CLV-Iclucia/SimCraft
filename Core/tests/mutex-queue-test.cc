#include <Core/mutex-queue.h>

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

using ksk::core::mutex_queue;

TEST(MutexQueueTest, TryPopReturnsEmptyWhenQueueIsEmpty)
{
  mutex_queue<int> queue;
  EXPECT_TRUE(queue.empty());
  EXPECT_FALSE(queue.try_pop().has_value());
}

TEST(MutexQueueTest, PushAndTryPopMoveOnlyValue)
{
  mutex_queue<std::unique_ptr<int>> queue;
  queue.push(std::make_unique<int>(42));

  auto value = queue.try_pop();
  ASSERT_TRUE(value.has_value());
  ASSERT_TRUE(*value);
  EXPECT_EQ(**value, 42);
  EXPECT_TRUE(queue.empty());
}

TEST(MutexQueueTest, WaitPopBlocksUntilValueArrives)
{
  mutex_queue<int> queue;
  std::atomic<int> result{0};

  std::thread consumer([&] {
    auto value = queue.wait_pop();
    ASSERT_TRUE(value.has_value());
    result = *value;
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  queue.push(7);
  consumer.join();

  EXPECT_EQ(result.load(), 7);
}

TEST(MutexQueueTest, ShutdownWakesWaitingConsumer)
{
  mutex_queue<int> queue;
  std::atomic<bool> woke{false};

  std::thread consumer([&] {
    auto value = queue.wait_pop();
    EXPECT_FALSE(value.has_value());
    woke = true;
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  queue.shutdown();
  consumer.join();

  EXPECT_TRUE(woke.load());
  EXPECT_TRUE(queue.is_shutdown());
}

TEST(MutexQueueTest, PushWaitsForCapacity)
{
  mutex_queue<int> queue(1);
  queue.push(1);
  std::atomic<bool> pushed_second{false};

  std::thread producer([&] {
    queue.push(2);
    pushed_second = true;
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(pushed_second.load());

  auto first = queue.wait_pop();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(*first, 1);

  producer.join();
  EXPECT_TRUE(pushed_second.load());

  auto second = queue.wait_pop();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*second, 2);
}
