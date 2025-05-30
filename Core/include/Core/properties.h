//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_PROPERTIES_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_PROPERTIES_H_
namespace sim::core {
struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
  NonCopyable(NonCopyable &&) = default;
};
struct Resource {
  Resource(Resource &&) = delete;
};
}
#endif // SIMCRAFT_CORE_INCLUDE_CORE_PROPERTIES_H_
