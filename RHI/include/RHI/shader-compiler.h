//
// shader-compiler.h
// HLSL → SPIR-V / DXIL compiler. Pure CPU; no Device required.
// See docs/rhi-plan.md §5.1.
//
// The compiler is intentionally decoupled from `Device`:
//   • R3 can be exercised in unit tests before any Vulkan device exists.
//   • Build-time HLSL embed (future) doesn't drag GPU init into the toolchain.
//   • Tests can feed mock bytecode straight into Device::createShader (R4) and
//     skip DXC entirely.
//
// Backend selection still drives codegen, but the choice is instance-scoped
// rather than tied to any particular runtime object:
//   • Bind a default once via ShaderCompiler::create(Backend) when the compiler
//     shadows a known runtime backend.
//   • Leave the compiler unbound via ShaderCompiler::create() when it is used
//     standalone, then set ShaderCompileOptions::targetBackend per call.
//   • If neither path provides a backend, compilation fails fast instead of
//     silently producing the wrong bytecode format.
//   • Backend::Vulkan → SPIR-V via `-spirv -fspv-target-env=vulkan1.3`.
//   • Backend::Dx12   → DXIL (best-effort on non-Windows; signing requires
//                       dxil.dll which is Windows-only).
//

#pragma once

#include <Core/properties.h>
#include <RHI/backend.h>
#include <RHI/shader.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sim::rhi {

struct ShaderCompileOptions {
  std::string entryPoint = "main";
  ShaderStage stage = ShaderStage::Compute;

  // Optional per-call override. When unset, ShaderCompiler::create(Backend)
  // provides the default codegen target for the compiler instance. Leaving both
  // unset is an error.
  std::optional<Backend> targetBackend;

  // Preprocessor defines: each pair is (name, value). Empty value emits `-D NAME`.
  std::vector<std::pair<std::string, std::string>> defines;

  // Additional include search paths (`-I`).
  std::vector<std::filesystem::path> includeDirs;

  // When true, populate CompiledShader::disassembly with textual disassembly.
  // R3 limitation: only DXIL output (Backend::Dx12) gets a populated
  // disassembly. Feeding SPIR-V to IDxcCompiler3::Disassemble fails
  // (E_FAIL); SPIR-V disassembly would need SPIRV-Tools, which isn't pulled
  // in until R4. For Vulkan output the field is left empty and an info-level
  // log is emitted. Adds one extra DXC call when set; disabled by default.
  bool generateDisassembly = false;

  // -Od -Zi (preserve debug info, disable optimization). Otherwise -O3.
  bool enableDebugInfo = false;
};

struct CompiledShader {
  // Final bytecode (SPIR-V if Backend::Vulkan, DXIL if Backend::Dx12).
  std::vector<std::byte> bytecode;

  // Populated only when ShaderCompileOptions::generateDisassembly == true.
  std::string disassembly;
};

class ShaderCompiler : public sim::core::NonCopyable {
 public:
  // Returns nullptr if dxcompiler is unavailable (DLL/SO not on the loader
  // path, or DxcCreateInstance fails).
  static std::unique_ptr<ShaderCompiler> create();

  // Same as create(), but binds a default codegen backend to the compiler
  // instance so most call sites don't need to repeat it.
  static std::unique_ptr<ShaderCompiler> create(Backend defaultBackend);

  virtual ~ShaderCompiler() = default;

  // Compile HLSL source. On failure returns std::nullopt and logs the DXC
  // diagnostics via spdlog at error level. On success the result is also
  // memoised on (source, options) so a second call returns the same bytecode
  // without recompiling. If options.targetBackend is unset, the compiler's
  // instance default is used instead.
  virtual std::optional<CompiledShader> compileHlsl(
      std::string_view source,
      const ShaderCompileOptions& options) = 0;

  // Read `path` and forward to compileHlsl(). The file's parent directory is
  // automatically prepended to includeDirs so `#include "neighbor.hlsl"` works
  // with no extra ceremony from the caller.
  virtual std::optional<CompiledShader> compileHlslFile(
      const std::filesystem::path& path,
      const ShaderCompileOptions& options) = 0;
};

}  // namespace sim::rhi
