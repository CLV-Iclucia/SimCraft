//
// Created by creeper on 23-9-1.
//

#pragma once
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
