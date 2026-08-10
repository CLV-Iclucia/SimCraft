//
// pipeline.h
// Pipeline + ComputePipelineDesc. R4 covers compute only; graphics in R6.
// See docs/rhi-plan.md §3.4.2 / §3.4.3.
//
// What's stored on a Pipeline:
//   • Backend pipeline object (VkPipeline) + layout (VkPipelineLayout +
//     N × VkDescriptorSetLayout) — derived from the bound shader's reflection.
//   • A back-reference to the Shader (so CommandList::dispatch can re-fetch
//     reflection for the SHADER_PARAMS flow without going through Device).
//
// What's NOT here:
//   • PipelineCache. The Vulkan backend keeps a private VkPipelineCache to
//     speed up SPIR-V → ISA. Pipeline-deduplication via Device-side
//     std::map<PipelineDesc, PipelineRef> is plan §3.4.2 R22 work; a
//     description-keyed cache is added when graphics pipelines arrive (R6+).
//

#pragma once

#include <Core/rc-ptr.h>
#include <Core/properties.h>
#include <RHI/format.h>
#include <RHI/shader.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace ksk::rhi {

struct ComputePipelineDesc {
  ShaderRef shader;

  // Future: specialization constants, optional pipeline cache name, etc.

  // Pipeline cache comparison support
  bool operator<(const ComputePipelineDesc& o) const;
};

struct GraphicsPipelineDesc {
  ShaderRef vertexShader;
  ShaderRef fragmentShader;

  // Vertex input (must be explicitly declared; SPIR-V reflection can't infer stride)
  struct VertexAttribute {
    uint32_t location;
    Format format;
    uint32_t offset;

    bool operator<(const VertexAttribute& o) const {
      if (location != o.location) return location < o.location;
      if (format != o.format) return format < o.format;
      return offset < o.offset;
    }
  };
  struct VertexBinding {
    uint32_t binding;
    uint32_t stride;
    using Attributes = std::vector<VertexAttribute>;
    Attributes attributes;

    bool operator<(const VertexBinding& o) const {
      if (binding != o.binding) return binding < o.binding;
      if (stride != o.stride) return stride < o.stride;
      return attributes < o.attributes;
    }
  };
  using Bindings = std::vector<VertexBinding>;
  Bindings vertexBindings;

  enum class PrimitiveTopology { TriangleList, LineList, PointList };
  PrimitiveTopology topology = PrimitiveTopology::TriangleList;

  // ---- Packed pipeline-state flags ----
  // The boolean state (depth test/write, alpha blend) is folded into a single
  // word instead of N separate bools: keeps the desc compact (better cache
  // behaviour as a pipeline-cache map key) and lets the whole flag set be
  // compared as one integer in operator<. 32 bits available — add new flags by
  // defining a kFlag* constant plus a matching accessor.
  using Flags = uint32_t;
  static constexpr Flags kFlagDepthTest  = 1u << 0;
  static constexpr Flags kFlagDepthWrite = 1u << 1;
  static constexpr Flags kFlagAlphaBlend = 1u << 2;

  // Default: depth test + write enabled, alpha blending disabled.
  Flags flags = kFlagDepthTest | kFlagDepthWrite;

  bool depthTest() const noexcept { return (flags & kFlagDepthTest) != 0; }
  bool depthWrite() const noexcept { return (flags & kFlagDepthWrite) != 0; }
  bool alphaBlend() const noexcept { return (flags & kFlagAlphaBlend) != 0; }

  void setDepthTest(bool b) noexcept {
    flags = static_cast<Flags>((flags & ~kFlagDepthTest) | (b ? kFlagDepthTest : 0));
  }
  void setDepthWrite(bool b) noexcept {
    flags = static_cast<Flags>((flags & ~kFlagDepthWrite) | (b ? kFlagDepthWrite : 0));
  }
  void setAlphaBlend(bool b) noexcept {
    flags = static_cast<Flags>((flags & ~kFlagAlphaBlend) | (b ? kFlagAlphaBlend : 0));
  }

  // Render target format description
  using Formats = std::vector<Format>;
  Formats colorFormats;
  Format depthFormat = Format::Undefined;

  // Pipeline cache comparison support
  bool operator<(const GraphicsPipelineDesc& o) const;
};

  struct RayTracingPipelineDesc
  {

  };

// ComputePipelineDesc comparison (simple: just shader pointer)
inline bool ComputePipelineDesc::operator<(const ComputePipelineDesc& o) const {
  return shader.get() < o.shader.get();
}

// GraphicsPipelineDesc comparison (see plan §1b cookbook)
inline bool GraphicsPipelineDesc::operator<(const GraphicsPipelineDesc& o) const {
  // 1. Compare shader pointers (fastest differentiation)
  if (vertexShader.get() != o.vertexShader.get())
    return vertexShader.get() < o.vertexShader.get();
  if (fragmentShader.get() != o.fragmentShader.get())
    return fragmentShader.get() < o.fragmentShader.get();
  // 2. Compare topology + state flags (whole flag set as one integer)
  if (topology != o.topology) return topology < o.topology;
  if (flags != o.flags) return flags < o.flags;
  // 3. Compare render target formats
  if (colorFormats != o.colorFormats) return colorFormats < o.colorFormats;
  if (depthFormat != o.depthFormat) return depthFormat < o.depthFormat;
  // 4. Compare vertex layout (last, as shader pointer usually differentiates)
  return vertexBindings < o.vertexBindings;
}

class Pipeline : public core::NonCopyable {
 public:
  void addRef() noexcept { m_rc.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept {
    if (m_rc.fetch_sub(1, std::memory_order_acq_rel) == 1) destroy();
  }

  virtual ~Pipeline() = default;

  // R6 new: Return merged reflection info (compute: single shader;
  // graphics: VS ∪ FS)
  virtual const class ReflectionInfo& reflection() const = 0;

  // R6 deprecated: Remove shader(). Reason: undefined for graphics/ray tracing
  // pipelines. Use reflection() instead for SHADER_PARAMS resolution.
  // See docs/rhi-r6-plan.md §3 design decisions.

 protected:
  Pipeline() = default;
  virtual void destroy() noexcept = 0;

 private:
  std::atomic<uint32_t> m_rc{0};
};

using PipelineRef = core::rc_ptr<Pipeline>;

}  // namespace ksk::rhi
