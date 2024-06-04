//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_PROPERTIES_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_PROPERTIES_H_
namespace core {
struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};

struct Singleton : NonCopyable {
  Singleton() = default;
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;
};
}
#endif // SIMCRAFT_CORE_INCLUDE_CORE_PROPERTIES_H_
