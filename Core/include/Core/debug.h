//
// Created by creeper on 24-3-20.
//

#ifndef DEBUG_H
#define DEBUG_H
#include <iostream>
#include <Core/log.h>
namespace core {
template<typename... Args>
void ERROR(details::with_source_location<std::format_string<Args...>> fmt, Args &&... args) {
  details::generic_log_terminal(LogLevel::Error, LogLevel::Error, std::move(fmt), std::forward<Args>(args)...);
}
}
#endif //DEBUG_H
