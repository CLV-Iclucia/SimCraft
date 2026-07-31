//
// Kiseki Profiler — thin wrapper over Tracy
//
// Usage:
//   KSK_PROFILE_FUNCTION()
//   KSK_PROFILE_SCOPE("name")
//   KSK_PROFILE_FRAME_MARK()
//   KSK_PROFILE_VALUE("name", val)
//
//

#pragma once

#ifdef TRACY_ENABLE

#include <tracy/Tracy.hpp>

#define KSK_PROFILE_FUNCTION() ZoneScoped
#define KSK_PROFILE_SCOPE(name) ZoneScopedN(name)
#define KSK_PROFILE_FRAME_MARK() FrameMark
#define KSK_PROFILE_VALUE(name, val)           \
  do {                                         \
    TracyPlot(name, static_cast<double>(val)); \
  } while (0)
#define KSK_PROFILE_SCOPE_COLOR(name, color) ZoneScopedNC(name, color)

#define SIM_PROFILE_FUNCTION() KSK_PROFILE_FUNCTION()
#define SIM_PROFILE_SCOPE(name) KSK_PROFILE_SCOPE(name)
#define SIM_PROFILE_FRAME_MARK() KSK_PROFILE_FRAME_MARK()
#define SIM_PROFILE_VALUE(name, val) KSK_PROFILE_VALUE(name, val)
#define SIM_PROFILE_SCOPE_COLOR(name, color) \
  KSK_PROFILE_SCOPE_COLOR(name, color)

namespace ksk::core::profiler_colors {
inline constexpr unsigned int kRed = 0xFF6B6B;
inline constexpr unsigned int kCyan = 0x4ECDC4;
inline constexpr unsigned int kBlue = 0x45B7D1;
inline constexpr unsigned int kOrange = 0xFFA07A;
inline constexpr unsigned int kGreen = 0x98D8C8;
inline constexpr unsigned int kPurple = 0xDDA0DD;
}  // namespace ksk::core::profiler_colors

#else  // TRACY_ENABLE not defined

#define SIM_PROFILE_FUNCTION() (void)0
#define SIM_PROFILE_SCOPE(name) (void)0
#define SIM_PROFILE_FRAME_MARK() (void)0
#define SIM_PROFILE_VALUE(name, val) (void)0
#define KSK_PROFILE_FUNCTION() (void)0
#define KSK_PROFILE_SCOPE(name) (void)0
#define KSK_PROFILE_FRAME_MARK() (void)0
#define KSK_PROFILE_VALUE(name, val) (void)0
#define KSK_PROFILE_SCOPE_COLOR(name, color) (void)0
#define SIM_PROFILE_SCOPE_COLOR(name, color) (void)0

namespace ksk::core::profiler_colors {
inline constexpr unsigned int kGradient = 0;
inline constexpr unsigned int kHessian = 0;
inline constexpr unsigned int kSolver = 0;
inline constexpr unsigned int kCollision = 0;
inline constexpr unsigned int kLineSearch = 0;
inline constexpr unsigned int kEnergy = 0;
}  // namespace ksk::core::profiler_colors

#endif  // TRACY_ENABLE
