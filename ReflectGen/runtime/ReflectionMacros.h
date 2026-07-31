#pragma once

#include <cstdint>

#if defined(__has_include)
#if __has_include(<Core/rtti.h>)
#include <Core/rtti.h>
#endif
#endif

#ifndef KISEKI_CORE_RTTI_TYPES_DEFINED
namespace RF {
struct TypeInfo {
    const TypeInfo* parent;
    uint32_t depth;
};

using TypeId = const TypeInfo*;
inline constexpr TypeId InvalidType = nullptr;
}
#define KISEKI_CORE_RTTI_TYPES_DEFINED 1
#endif

#if defined(__clang__)
#define RF_CLASS(...) [[clang::annotate("rf.class:" #__VA_ARGS__)]]
#else
#define RF_CLASS(...)
#endif

#if defined(__clang__)
#define RF_RTTI_METHOD_ANNOTATION [[clang::annotate("rf.rtti_body")]]
#else
#define RF_RTTI_METHOD_ANNOTATION
#endif

#define RF_RTTI_BODY() \
public: \
    RF_RTTI_METHOD_ANNOTATION virtual ::RF::TypeId GetTypeId() const;

#define KLASS(...) RF_CLASS(__VA_ARGS__)
#define KLASS_GENERATED_BODY() RF_RTTI_BODY()
