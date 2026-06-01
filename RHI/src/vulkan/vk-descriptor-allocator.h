//
// vk-descriptor-allocator.h
// VkDescriptorSet allocator with optional ring-of-pools for frame-pipelined
// rendering. See docs/rhi-r7-swapchain-plan.md §4.
//
// Usage modes:
//
//   1. Single-pool (R0–R6, pure compute / tests):
//      Construct with just VkDevice. Use allocate() / reset() as before.
//      reset() recycles the entire pool — caller must have vkDeviceWaitIdle'd.
//
//   2. Ring-of-pools (R7+, frame-pipelined rendering):
//      After swapchain creation, call initRing(N) where N = imageCount.
//      Each frame:
//        - setActivePool(frameIdx) at frame start (after waitFence)
//        - resetActivePool() to recycle that frame's pool (safe: fence done)
//        - allocate() goes to the active pool
//
//   Computing path (submitAndWait) continues to use the base pool (index 0)
//   and reset() after sync — unaffected by ring-of-pools.
//

#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace sim::rhi::vulkan {

class DescriptorSetAllocator {
 public:
  explicit DescriptorSetAllocator(VkDevice device);
  ~DescriptorSetAllocator();

  struct Caps {
    uint32_t maxSets = 256;
    uint32_t storageBuffers = 1024;
    uint32_t uniformBuffers = 256;
    uint32_t storageImages = 512;
    uint32_t sampledImages = 512;
    uint32_t samplers = 128;
  };

  // ---- Single-pool API (unchanged from R4–R6) ------------------------------

  // Allocate a descriptor set from the currently active pool.
  // Throws std::runtime_error on pool exhaustion.
  VkDescriptorSet allocate(VkDescriptorSetLayout layout);

  // Reset ALL pools (single-pool mode) or just the base pool (ring mode).
  // Caller must ensure GPU is not referencing any allocated sets.
  void reset() noexcept;

  // ---- Ring-of-pools API (R7+) ---------------------------------------------

  // Initialize N additional pools for frame-pipelined usage.
  // pool[0] is the existing single pool. Pools [1..N-1] are created fresh.
  // If already initialized with a different count, destroys old ring and
  // rebuilds.
  void initRing(uint32_t poolCount);

  // Set which pool subsequent allocate() calls draw from.
  // index must be < ring size (asserts in debug).
  void setActivePool(uint32_t index) noexcept;

  // Reset the currently active pool. Safe to call after the corresponding
  // frame fence has been waited on.
  void resetActivePool() noexcept;

  // Number of pools in the ring (1 if ring not initialized).
  uint32_t ringSize() const noexcept {
    return static_cast<uint32_t>(m_pools.size());
  }

 private:
  VkDescriptorPool createPool();

  VkDevice m_device;
  std::vector<VkDescriptorPool> m_pools;  // m_pools[0] is the "base" pool
  uint32_t m_activePoolIdx = 0;
};

}  // namespace sim::rhi::vulkan
