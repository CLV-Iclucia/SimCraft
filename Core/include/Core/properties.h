//
// Created by creeper on 23-9-1.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_PROPERTIES_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_PROPERTIES_H_
namespace core {
struct DisableCopy {
  DisableCopy() = default;
  DisableCopy(const DisableCopy &) = delete;
  DisableCopy &operator=(const DisableCopy &) = delete;
};
}
#endif // SIMCRAFT_CORE_INCLUDE_CORE_PROPERTIES_H_
