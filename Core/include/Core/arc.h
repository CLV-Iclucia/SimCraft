//
// Created by CreeperIclucia-Vader on 25-5-26.
//
#pragma once
#include <memory>
namespace sim::core {
template <typename T>
using arc = std::shared_ptr<const T>;
}
