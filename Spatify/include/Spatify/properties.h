//
// Created by creeper on 6/20/24.
//

#ifndef SIMCRAFT_SPATIFY_INCLUDE_SPATIFY_PROPERTIES_H_
#define SIMCRAFT_SPATIFY_INCLUDE_SPATIFY_PROPERTIES_H_
namespace spatify {

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};
}
#endif //SIMCRAFT_SPATIFY_INCLUDE_SPATIFY_PROPERTIES_H_
