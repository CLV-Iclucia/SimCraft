//
// vk-descriptor-allocator.cc
//

#include "vk-descriptor-allocator.h"

#include "vk-internals.h"

#include <array>
#include <cassert>

namespace sim::rhi::vulkan {

DescriptorSetAllocator::DescriptorSetAllocator(VkDevice device)
    : m_device(device) {
  // Start with a single pool (the "base" pool).
  m_pools.push_back(createPool());
}

DescriptorSetAllocator::~DescriptorSetAllocator() {
  for (VkDescriptorPool pool : m_pools) {
    if (pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(m_device, pool, nullptr);
    }
  }
  m_pools.clear();
}

// ---- Single-pool API -------------------------------------------------------

VkDescriptorSet DescriptorSetAllocator::allocate(VkDescriptorSetLayout layout) {
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = m_pools[m_activePoolIdx];
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &layout;

  VkDescriptorSet set = VK_NULL_HANDLE;
  VkResult r = vkAllocateDescriptorSets(m_device, &ai, &set);
  if (r != VK_SUCCESS) {
    spdlog::error(
        "[DescriptorSetAllocator] vkAllocateDescriptorSets failed (VkResult "
        "{}, pool index {}): pool exhausted? bump Caps or call reset/resetActivePool.",
        static_cast<int>(r), m_activePoolIdx);
    throw std::runtime_error(
        "DescriptorSetAllocator: vkAllocateDescriptorSets failed");
  }
  return set;
}

void DescriptorSetAllocator::reset() noexcept {
  // Reset all pools.
  for (VkDescriptorPool pool : m_pools) {
    if (pool != VK_NULL_HANDLE) {
      vkResetDescriptorPool(m_device, pool, 0);
    }
  }
  m_activePoolIdx = 0;
}

// ---- Ring-of-pools API -----------------------------------------------------

void DescriptorSetAllocator::initRing(uint32_t poolCount) {
  if (poolCount < 1) poolCount = 1;
  if (static_cast<uint32_t>(m_pools.size()) == poolCount) return;

  // Destroy extra pools if shrinking (rare).
  while (m_pools.size() > poolCount) {
    vkDestroyDescriptorPool(m_device, m_pools.back(), nullptr);
    m_pools.pop_back();
  }

  // Create additional pools if growing.
  while (m_pools.size() < poolCount) {
    m_pools.push_back(createPool());
  }

  m_activePoolIdx = 0;
}

void DescriptorSetAllocator::setActivePool(uint32_t index) noexcept {
  assert(index < m_pools.size() && "setActivePool: index out of range");
  m_activePoolIdx = index;
}

void DescriptorSetAllocator::resetActivePool() noexcept {
  VkDescriptorPool pool = m_pools[m_activePoolIdx];
  if (pool != VK_NULL_HANDLE) {
    vkResetDescriptorPool(m_device, pool, 0);
  }
}

// ---- Internal: createPool --------------------------------------------------

VkDescriptorPool DescriptorSetAllocator::createPool() {
  Caps caps;  // defaults

  std::array<VkDescriptorPoolSize, 5> sizes{{
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, caps.storageBuffers},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, caps.uniformBuffers},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, caps.storageImages},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, caps.sampledImages},
      {VK_DESCRIPTOR_TYPE_SAMPLER, caps.samplers},
  }};

  VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  ci.maxSets = caps.maxSets;
  ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
  ci.pPoolSizes = sizes.data();
  ci.flags = 0;  // No FREE_DESCRIPTOR_SET_BIT — reset only.

  VkDescriptorPool pool = VK_NULL_HANDLE;
  VK_CHECK(vkCreateDescriptorPool(m_device, &ci, nullptr, &pool));
  return pool;
}

}  // namespace sim::rhi::vulkan
