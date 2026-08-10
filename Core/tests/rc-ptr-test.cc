#include <Core/rc-ptr.h>

#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using ksk::core::rc_ptr;

namespace {

struct MockRefCounted {
  std::atomic<uint32_t> rc{0};
  std::atomic<int>* destroy_counter = nullptr;

  void addRef() noexcept { rc.fetch_add(1, std::memory_order_relaxed); }

  void release() noexcept
  {
    if (rc.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (destroy_counter) {
        destroy_counter->fetch_add(1, std::memory_order_relaxed);
      }
      delete this;
    }
  }
};

}  // namespace

TEST(RcPtrTest, DefaultIsNull)
{
  rc_ptr<MockRefCounted> p;
  EXPECT_FALSE(p.valid());
  EXPECT_FALSE(static_cast<bool>(p));
  EXPECT_EQ(p.get(), nullptr);
}

TEST(RcPtrTest, ConstructIncrementsRefcount)
{
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroy_counter = &destroyed;
  {
    rc_ptr<MockRefCounted> p(raw);
    EXPECT_TRUE(p.valid());
    EXPECT_EQ(p->rc.load(), 1u);
    EXPECT_EQ(destroyed.load(), 0);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RcPtrTest, AdoptDoesNotIncrement)
{
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroy_counter = &destroyed;
  raw->addRef();
  {
    auto p = rc_ptr<MockRefCounted>::adopt(raw);
    EXPECT_EQ(p->rc.load(), 1u);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RcPtrTest, CopyShares)
{
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroy_counter = &destroyed;
  {
    rc_ptr<MockRefCounted> p(raw);
    {
      rc_ptr<MockRefCounted> q = p;
      EXPECT_EQ(p->rc.load(), 2u);
      EXPECT_EQ(p.get(), q.get());
    }
    EXPECT_EQ(p->rc.load(), 1u);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RcPtrTest, MoveStealsOwnership)
{
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroy_counter = &destroyed;
  {
    rc_ptr<MockRefCounted> p(raw);
    rc_ptr<MockRefCounted> q = std::move(p);
    EXPECT_FALSE(p.valid());
    EXPECT_TRUE(q.valid());
    EXPECT_EQ(q->rc.load(), 1u);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RcPtrTest, AssignReleasesOldThenIncrementsNew)
{
  std::atomic<int> destroyed_a{0};
  std::atomic<int> destroyed_b{0};
  auto* a = new MockRefCounted{};
  a->destroy_counter = &destroyed_a;
  auto* b = new MockRefCounted{};
  b->destroy_counter = &destroyed_b;
  {
    rc_ptr<MockRefCounted> p(a);
    rc_ptr<MockRefCounted> q(b);
    p = q;
    EXPECT_EQ(destroyed_a.load(), 1);
    EXPECT_EQ(b->rc.load(), 2u);
  }
  EXPECT_EQ(destroyed_b.load(), 1);
}

TEST(RcPtrTest, ResetReleasesEarly)
{
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroy_counter = &destroyed;
  rc_ptr<MockRefCounted> p(raw);
  p.reset();
  EXPECT_FALSE(p.valid());
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RcPtrTest, ConcurrentAddRefReleaseIsRaceFree)
{
  std::atomic<int> destroyed{0};
  auto* raw = new MockRefCounted{};
  raw->destroy_counter = &destroyed;
  rc_ptr<MockRefCounted> root(raw);

  constexpr int k_threads = 8;
  constexpr int k_iters = 10000;
  std::vector<std::thread> threads;
  threads.reserve(k_threads);
  for (int i = 0; i < k_threads; ++i) {
    threads.emplace_back([&] {
      for (int j = 0; j < k_iters; ++j) {
        rc_ptr<MockRefCounted> copy = root;
        (void)copy.get();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(root.valid());
  EXPECT_EQ(root->rc.load(), 1u);
  EXPECT_EQ(destroyed.load(), 0);
  root.reset();
  EXPECT_EQ(destroyed.load(), 1);
}
