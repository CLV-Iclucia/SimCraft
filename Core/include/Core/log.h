//
// Created by creeper on 5/24/24.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_LOG_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_LOG_H_
#include <cstdint>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <source_location>
#include <format>
#include <chrono>
#include <thread>
namespace core {
#define LOG_FOREACH_LEVEL(replace) \
  replace(Trace)                   \
  replace(Debug)                   \
  replace(Info)                    \
  replace(Critical)                \
  replace(Warning)                  \
  replace(Error)                    \
  replace(Fatal)
#define ENUM_ENTRY(entry) entry,
enum LogLevel : uint8_t {
  LOG_FOREACH_LEVEL(ENUM_ENTRY)
};
#undef ENUM_ENTRY
namespace details {
template<typename T>
struct with_source_location {
 private:
  T inner;
  std::source_location loc;
 public:
  template<typename U>
  requires std::constructible_from<T, U>
  consteval with_source_location(U &&inner, std::source_location loc = std::source_location::current())
      : inner(std::forward<U>(inner)), loc(loc) {}
  constexpr T const &format() const { return inner; }
  [[nodiscard]] constexpr std::source_location const &location() const { return loc; }
};

inline std::string_view levelString(LogLevel level) {
#define SWITCH_CASE_ENTRY(entry) case(entry): \
  return #entry;
  switch (level) {
    LOG_FOREACH_LEVEL(SWITCH_CASE_ENTRY)
  }
  std::cerr << "Unknown log level\n";
  return "";
}
#undef SWITCH_CASE_ENTRY

inline std::string message(LogLevel level, std::string_view msg, const std::source_location &loc) {
  std::chrono::zoned_time now{std::chrono::current_zone(), std::chrono::high_resolution_clock::now()};
  return std::format("{} {}:{} [Thread {}] [{}] {}\n",
                     now,
                     loc.file_name(),
                     loc.line(),
                     std::hash<std::thread::id>{}(std::this_thread::get_id()),
                     levelString(level),
                     msg);
}

template<typename... Args>
inline void generic_log(std::fstream &stream,
                        LogLevel level,
                        LogLevel global_level,
                        with_source_location<std::format_string<Args...>> fmt,
                        Args &&... args) {
  if (level < global_level) return;
  const auto &loc = fmt.location();
  auto msg = std::vformat(fmt.format().get(), std::make_format_args(args...));
  stream << message(level, msg, loc);
}

template<typename... Args>
inline void generic_log_terminal(LogLevel level,
                                 LogLevel global_level,
                                 with_source_location<std::format_string<Args...>> fmt,
                                 Args &&... args) {
  if (level < global_level) return;
  const auto &loc = fmt.location();
  auto msg = std::vformat(fmt.format().get(), std::make_format_args(args...));
  if (level >= LogLevel::Error)
    std::cerr << message(level, msg, loc);
  else
    std::cout << message(level, msg, loc);
}
}
struct Logger {
  LogLevel log_level{Info};
  std::filesystem::path output_file{"./log.txt"};
  std::fstream stream;
  Logger() : stream(output_file) {}
  Logger(LogLevel level, const std::filesystem::path &path) : log_level(level), output_file(path), stream(path) {}
#define LOG_METHOD(level) \
  template <typename... Args> \
  const Logger& log##level(details::with_source_location<std::format_string<Args...>> fmt, Args&&... args) const { \
    details::generic_log(stream, level, log_level, std::move(fmt), std::forward<Args>(args)...);                \
    return *this;                      \
  }
  LOG_FOREACH_LEVEL(LOG_METHOD)
#undef LOG_METHOD

#define LOG_TERMINAL_METHOD(level) \
  template <typename... Args> \
  const Logger& logTerminal##level(details::with_source_location<std::format_string<Args...>> fmt, Args&&... args) const { \
    details::generic_log_terminal(level, log_level, std::move(fmt), std::forward<Args>(args)...);                \
    return *this;                      \
  }
  LOG_FOREACH_LEVEL(LOG_TERMINAL_METHOD)
};
#undef LOG_TERMINAL_METHOD
#undef LOG_FOREACH_LEVEL
}
#endif //SIMCRAFT_CORE_INCLUDE_CORE_LOG_H_
