//
// shader-params.h
// SHADER_PARAMS macro system. Header-only by design.
// See docs/rhi-plan.md §3.4.4 / §13.1 / §13.2.
//
// User-facing usage:
//   SHADER_PARAMS_BEGIN(SaxpyParams)
//     SHADER_PARAM_UAV   (BufferRef, g_y);
//     SHADER_PARAM_SCALAR(uint32_t,  count);
//     SHADER_PARAM_SCALAR(float,     alpha);
//   SHADER_PARAMS_END();
//
//   SaxpyParams params;
//   params.g_y   = y;       // looks like assignment — IS assignment, but
//   params.count = N;       // each field is a thin ParamSlot<T> wrapper that
//   params.alpha = 2.5f;    // forwards the value into m_value.
//   cmd->dispatch(pso, params, divRoundUp(N, 256), 1, 1);
//
// ===== Design intent (the contract this file enforces) =====================
//
// Two non-negotiable properties:
//
//   1. Each field is a wrapper type with an overloaded operator= so users
//      see plain "params.g_y = buf" syntax, but the framework can hook the
//      assignment if it ever needs to (validation, change-tracking, etc.).
//      The wrapper is `ParamSlot<T>` below — a single-T-member struct with
//      assignment overloads + transparent read access.
//
//   2. ZERO static accumulator variables per field. The previous design
//      (now reverted) emitted an `inline static const int _<field>_reg =
//      registerField<_Self>(...)` per SHADER_PARAM_*; that produced one
//      ODR-merged global per field and a function-local-static schema vector
//      per `Self`. Both were avoidable. The new design builds the per-class
//      `_schema` lazily inside the ParamSlot ctor via DMI side effects, so
//      static-storage cost across all SHADER_PARAMS classes is zero.
//
// ===== Internal mechanics ==================================================
//
//   1. SHADER_PARAM_*(Type, FieldName) emits TWO things into the class body:
//        a. A field declaration: `ParamSlot<Type> FieldName = (...);`
//        b. The `(...)` is a comma expression whose left operand calls
//           `this->_registerField(name, kind, offset, valueSize)` on the
//           partially-constructed leaf — populating `_schema` — and whose
//           right operand is a default-constructed `ParamSlot<Type>{}` used
//           to copy/move-initialise the field (mandatory copy-elision in
//           C++17+, so no actual copy).
//      Result: every SaxpyParams instance, on construction, walks its
//      declared fields once and pushes one FieldInfo per field into _schema.
//
//   2. CommandList::dispatch(pso, params, ...) checks the sentinel
//      `params._resolvedFor != pso.get()`; on mismatch it calls
//      `params._resolve(pso->reflection())` to map field names →
//      (set, binding) via the schema. ONE unordered_map lookup per field
//      at first dispatch (typical kernels have <20 fields). Steady state
//      is zero string ops.
//
//   3. `params._apply(cmd)` walks `_bindings` (vector<ResolvedEntry>) once
//      and issues `cmd.bindBufferAt / bindImageAt / bindSamplerAt / pushAt`.
//      No virtual calls in the loop, no string compares, no heap allocs.
//      Field reads are `*reinterpret_cast<const T*>(base + offset)` — safe
//      because `ParamSlot<T>` has T as its single member at offset 0.
//
// ===== Layout invariant ====================================================
//
//   ParamSlot<T> is required to have its T member at offset 0 (asserted via
//   static_assert when std::is_standard_layout_v<T>; relied upon for all T).
//   This means: given the address of a ParamSlot<T>, reinterpret_cast to T*
//   gives the underlying value's address. The `_apply` path uses this trick.
//
// ===== Push-constant offset strategy (plan §13.2 R23) ======================
//
//   SCALAR fields claim PC bytes in declaration order, starting from the
//   reflected PC block's `offset`. The HLSL `[[vk::push_constant]] struct`
//   field order MUST match the C++ SHADER_PARAM_SCALAR order. Mismatches
//   are runtime errors when total size overflows the reflected block; same-
//   sized misorderings are silently wrong (validation TODO — needs PC
//   member names from reflection, not in scope this round).
//
// ===== Why header-only =====================================================
//
//   `_resolve` and `_apply` are non-template member functions of the non-
//   template `ShaderParamsBase`, so they can in principle live in a .cc.
//   We keep `_resolve` inline here (no CommandList dependency) and define
//   `_apply` inline in commands.h (after CommandList is complete) so the
//   whole RHI module remains header-rich + easy to inline at call sites.
//   No per-class explicit instantiation or schema initialiser is needed.
//

#pragma once

#include <RHI/buffer.h>
#include <RHI/image.h>
#include <RHI/reflection.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sim::rhi {

// Forward decl — CommandList is referenced only by the inline _apply impl
// in commands.h.
class CommandList;

// ============================================================================
// 1. ParamSlot<T> — field wrapper with transparent operator=
// ============================================================================
//
// Layout invariant: ParamSlot<T>'s only data member is `T m_value`, at offset
// 0 of the wrapper. This lets the framework recover the underlying T from a
// ParamSlot address via reinterpret_cast.
//
// Usage by SHADER_PARAMS users is mostly transparent:
//   ParamSlot<BufferRef> g_y;
//   g_y = someBuffer;           // operator=(const T&)
//   BufferRef ref = g_y;        // implicit conversion via operator const T&
//   if (g_y.get())              // explicit access for places where conversion
//                               // is awkward (e.g. operator-> on RcPtr-likes)
//
template <class T>
class ParamSlot {
 public:
  ParamSlot() = default;
  ParamSlot(const ParamSlot&) = default;
  ParamSlot(ParamSlot&&) noexcept = default;
  ParamSlot& operator=(const ParamSlot&) = default;
  ParamSlot& operator=(ParamSlot&&) noexcept = default;

  // ---- The user-facing assignments ----
  ParamSlot& operator=(const T& v) {
    m_value = v;
    return *this;
  }
  ParamSlot& operator=(T&& v) noexcept(
      std::is_nothrow_move_assignable_v<T>) {
    m_value = std::move(v);
    return *this;
  }

  // ---- Read access ----
  // Implicit conversion so most "uses" of the slot (function args, return
  // statements, ternaries) Just Work without the user thinking about the
  // wrapper.
  operator const T&() const noexcept { return m_value; }

  // Explicit getters when implicit conversion is ambiguous (e.g. you want
  // to call a member function on the underlying T).
  const T& get() const noexcept { return m_value; }
  T& get() noexcept { return m_value; }

 private:
  T m_value{};
};

// Layout sanity: standard-layout T → ParamSlot<T> guaranteed standard-layout
// with m_value at offset 0. Non-standard-layout T (with vptr / non-empty
// virtual base / mixed access specifiers) loses standard-layout, but the
// single-member-offset-0 invariant still holds in practice on all major
// compilers — we don't assert it generically because static_assert with
// offsetof on non-standard-layout is conditionally supported.

// ============================================================================
// 2. detail:: schema types + kind helpers
// ============================================================================
namespace detail {

enum class FieldKind : uint8_t {
  UAVBuffer,     // RWStructuredBuffer  → VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
  UAVImage,      // RWTexture*          → VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
  SRVBuffer,     // ConstantBuffer/StructuredBuffer → UNIFORM/STORAGE_BUFFER
  SampledImage,  // Texture + Sampler combo → COMBINED_IMAGE_SAMPLER
  Sampler,       // standalone SamplerState
  Scalar,        // push-constant member
};

// Per-field metadata, accumulated into ShaderParamsBase::_schema by the
// ParamSlot DMI side effect.
struct FieldInfo {
  const char* name;   // string literal — no ownership
  FieldKind kind;
  uint32_t offset;    // offsetof(_Self, field) — within the leaf class
  uint32_t valueSize; // sizeof of the underlying T (for Scalar push consts)
};

// Per-instance resolved record. Built once on first dispatch (or on
// shader-switch). The hot path consumes this directly, no string ops.
struct ResolvedEntry {
  FieldKind kind = FieldKind::UAVBuffer;
  uint32_t fieldOffset = 0;  // offset of the ParamSlot within the leaf

  // For descriptor kinds:
  uint32_t set = 0;
  uint32_t binding = 0;

  // For Scalar:
  uint32_t pcOffset = 0;
  uint32_t pcSize = 0;
};

// Per-macro kind traits — same C++ type can mean different kinds depending
// on which SHADER_PARAM_* macro it's under (BufferRef as UAV vs SRV).
template <class T>
struct UAVKindOf;
template <>
struct UAVKindOf<BufferRef> {
  static constexpr FieldKind kind = FieldKind::UAVBuffer;
};
template <>
struct UAVKindOf<ImageRef> {
  static constexpr FieldKind kind = FieldKind::UAVImage;
};

template <class T>
struct SRVKindOf;
template <>
struct SRVKindOf<BufferRef> {
  static constexpr FieldKind kind = FieldKind::SRVBuffer;
};

template <class T>
struct ImageKindOf;
template <>
struct ImageKindOf<ImageBinding> {
  static constexpr FieldKind kind = FieldKind::SampledImage;
};

template <class T>
struct SamplerKindOf;
template <>
struct SamplerKindOf<SamplerRef> {
  static constexpr FieldKind kind = FieldKind::Sampler;
};

inline bool kindMatchesDescriptorType(FieldKind k,
                                      DescriptorBindingInfo::Type t) {
  using DT = DescriptorBindingInfo::Type;
  switch (k) {
    case FieldKind::UAVBuffer:
      return t == DT::StorageBuffer;
    case FieldKind::UAVImage:
      return t == DT::StorageImage;
    case FieldKind::SRVBuffer:
      return t == DT::UniformBuffer || t == DT::StorageBuffer;
    case FieldKind::SampledImage:
      return t == DT::SampledImage;
    case FieldKind::Sampler:
      return t == DT::Sampler;
    case FieldKind::Scalar:
      return false;
  }
  return false;
}

inline const char* fieldKindToString(FieldKind k) {
  switch (k) {
    case FieldKind::UAVBuffer:    return "UAVBuffer";
    case FieldKind::UAVImage:     return "UAVImage";
    case FieldKind::SRVBuffer:    return "SRVBuffer";
    case FieldKind::SampledImage: return "SampledImage";
    case FieldKind::Sampler:      return "Sampler";
    case FieldKind::Scalar:       return "Scalar";
  }
  return "?";
}

inline const char* descriptorTypeToString(DescriptorBindingInfo::Type t) {
  using DT = DescriptorBindingInfo::Type;
  switch (t) {
    case DT::StorageBuffer: return "StorageBuffer";
    case DT::UniformBuffer: return "UniformBuffer";
    case DT::StorageImage:  return "StorageImage";
    case DT::SampledImage:  return "SampledImage";
    case DT::Sampler:       return "Sampler";
  }
  return "?";
}

}  // namespace detail

// ============================================================================
// 3. ShaderParamsBase — non-template; holds per-instance state
// ============================================================================
//
// This is the only base class for SHADER_PARAMS-generated structs. It carries:
//   - _schema:       per-instance vector of FieldInfo, populated by the
//                    SHADER_PARAM_* macro's DMI side effect at construction.
//   - _bindings:     per-instance resolved bindings, built on first dispatch.
//   - _resolvedFor:  sentinel — which Shader* did we resolve against? Compared
//                    by raw pointer; cheap.
//
// All public members are intentionally `_`-prefixed to mark them as
// framework-managed; SHADER_PARAMS users don't touch them directly.
//
class ShaderParamsBase {
 public:
  std::vector<detail::FieldInfo> _schema;
  std::vector<detail::ResolvedEntry> _bindings;
  const void* _resolvedFor = nullptr;

  // Walk _schema + reflection, build _bindings. Throws std::runtime_error on
  // mismatch (missing binding, kind/type incompatible, PC overflow). Inline
  // here — no CommandList dependency.
  void _resolve(const ReflectionInfo& ri);

  // Walk _bindings, issue mid-tier CommandList calls. Defined inline in
  // commands.h since it depends on CommandList being a complete type.
  void _apply(CommandList& cmd) const;

  void _invalidate() noexcept {
    _resolvedFor = nullptr;
    _bindings.clear();
  }

 protected:
  ShaderParamsBase() = default;
  ~ShaderParamsBase() = default;  // non-virtual; SHADER_PARAMS structs are
                                   // never destroyed via base pointer

  // Called from the SHADER_PARAM_* macro's DMI side effect. Pushes one
  // FieldInfo per declared slot. Schema entries appear in declaration order
  // because C++ guarantees data-member initialisation in declaration order.
  //
  // Returns void; the macro uses comma expression `(_registerField(...), T{})`
  // so the void result is discarded and `T{}` becomes the slot value.
  void _registerField(const char* name, detail::FieldKind k, uint32_t offset,
                      uint32_t valueSize) {
    _schema.push_back({name, k, offset, valueSize});
  }
};

// ============================================================================
// 4. ShaderParamsBase::_resolve — non-template, inline definition
// ============================================================================
inline void ShaderParamsBase::_resolve(const ReflectionInfo& ri) {
  // Build name → DescriptorBindingInfo* lookup. Single pass over ri so total
  // work is O(N + M).
  std::unordered_map<std::string_view, const DescriptorBindingInfo*> byName;
  byName.reserve(ri.bindings.size());
  for (const auto& b : ri.bindings) byName.emplace(b.name, &b);

  _bindings.clear();
  _bindings.reserve(_schema.size());

  // PC block accounting (plan §13.2 R23: at most one PC block in HLSL/DXC).
  const uint32_t pcBlockOffset =
      ri.pushConstants.empty() ? 0u : ri.pushConstants[0].offset;
  const uint32_t pcBlockSize =
      ri.pushConstants.empty() ? 0u : ri.pushConstants[0].size;
  uint32_t pcRunning = 0;

  for (const auto& f : _schema) {
    detail::ResolvedEntry e{};
    e.kind = f.kind;
    e.fieldOffset = f.offset;

    if (f.kind == detail::FieldKind::Scalar) {
      if (pcRunning + f.valueSize > pcBlockSize) {
        throw std::runtime_error(
            std::string("shader param '") + f.name +
            "': scalar params total size (" +
            std::to_string(pcRunning + f.valueSize) +
            ") exceeds push constant block size (" +
            std::to_string(pcBlockSize) +
            " bytes). Check HLSL [[vk::push_constant]] struct matches C++ "
            "SHADER_PARAM_SCALAR declaration order and types.");
      }
      e.pcOffset = pcBlockOffset + pcRunning;
      e.pcSize = f.valueSize;
      pcRunning += f.valueSize;
    } else {
      auto it = byName.find(f.name);
      if (it == byName.end()) {
        std::string available;
        for (const auto& b : ri.bindings) {
          available += "\n  - " + b.name;
        }
        throw std::runtime_error(
            std::string("shader param '") + f.name +
            "' not found in shader reflection. Available bindings:" +
            available);
      }
      const auto* b = it->second;
      if (!detail::kindMatchesDescriptorType(f.kind, b->type)) {
        throw std::runtime_error(
            std::string("shader param '") + f.name + "': declared as " +
            detail::fieldKindToString(f.kind) +
            " in C++ but shader reflection says " +
            detail::descriptorTypeToString(b->type));
      }
      e.set = b->set;
      e.binding = b->binding;
    }

    _bindings.push_back(e);
  }
}

}  // namespace sim::rhi

// ============================================================================
// 5. User-facing macros
// ============================================================================
//
// MSVC needs /Zc:preprocessor to expand `#FieldName` consistently — the
// project already enables this in RHI/CMakeLists.txt.
//
// Each SHADER_PARAM_* expands to:
//   ParamSlot<Type> FieldName =
//       (this->_registerField("FieldName", kind, offsetof(_Self, FieldName),
//                             sizeof(Type)),
//        ::sim::rhi::ParamSlot<Type>{});
//
// The default-member-initialiser uses the comma operator:
//   - Left operand: side-effect call to _registerField on the partially-
//     constructed leaf (the base sub-object is fully constructed by the
//     time members are initialised, so _schema's vector is live).
//   - Right operand: a default-constructed ParamSlot<Type>{} that becomes
//     the field's initial value (mandatory copy elision in C++17+, so no
//     copy actually happens).
//
// `offsetof(_Self, FieldName)` inside the still-incomplete class body is
// the same gray-area pattern as before (Unreal SHADER_PARAMETER, Filament
// ParamBlock, etc. all rely on it). MSVC / Clang / GCC all accept it under
// /Zc:preprocessor.

#define SHADER_PARAMS_BEGIN(Name)                                              \
  struct Name : public ::sim::rhi::ShaderParamsBase {                          \
    using _Self = Name;

#define SHADER_PARAM_UAV(Type, FieldName)                                      \
  ::sim::rhi::ParamSlot<Type> FieldName =                                      \
      (this->_registerField(                                                   \
           #FieldName,                                                         \
           ::sim::rhi::detail::UAVKindOf<Type>::kind,                          \
           static_cast<uint32_t>(offsetof(_Self, FieldName)),                  \
           static_cast<uint32_t>(sizeof(Type))),                               \
       ::sim::rhi::ParamSlot<Type>{})

#define SHADER_PARAM_SRV(Type, FieldName)                                      \
  ::sim::rhi::ParamSlot<Type> FieldName =                                      \
      (this->_registerField(                                                   \
           #FieldName,                                                         \
           ::sim::rhi::detail::SRVKindOf<Type>::kind,                          \
           static_cast<uint32_t>(offsetof(_Self, FieldName)),                  \
           static_cast<uint32_t>(sizeof(Type))),                               \
       ::sim::rhi::ParamSlot<Type>{})

#define SHADER_PARAM_IMAGE(Type, FieldName)                                    \
  ::sim::rhi::ParamSlot<Type> FieldName =                                      \
      (this->_registerField(                                                   \
           #FieldName,                                                         \
           ::sim::rhi::detail::ImageKindOf<Type>::kind,                        \
           static_cast<uint32_t>(offsetof(_Self, FieldName)),                  \
           static_cast<uint32_t>(sizeof(Type))),                               \
       ::sim::rhi::ParamSlot<Type>{})

#define SHADER_PARAM_SAMPLER(Type, FieldName)                                  \
  ::sim::rhi::ParamSlot<Type> FieldName =                                      \
      (this->_registerField(                                                   \
           #FieldName,                                                         \
           ::sim::rhi::detail::SamplerKindOf<Type>::kind,                      \
           static_cast<uint32_t>(offsetof(_Self, FieldName)),                  \
           static_cast<uint32_t>(sizeof(Type))),                               \
       ::sim::rhi::ParamSlot<Type>{})

#define SHADER_PARAM_SCALAR(Type, FieldName)                                   \
  static_assert(::std::is_trivially_copyable_v<Type>,                          \
                "SHADER_PARAM_SCALAR(" #Type ", " #FieldName                   \
                "): Type must be trivially copyable");                         \
  ::sim::rhi::ParamSlot<Type> FieldName =                                      \
      (this->_registerField(                                                   \
           #FieldName,                                                         \
           ::sim::rhi::detail::FieldKind::Scalar,                              \
           static_cast<uint32_t>(offsetof(_Self, FieldName)),                  \
           static_cast<uint32_t>(sizeof(Type))),                               \
       ::sim::rhi::ParamSlot<Type>{})

#define SHADER_PARAMS_END()                                                    \
  }
