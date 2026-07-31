#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace RF {

struct TypeInfo {
  const TypeInfo* parent;
  uint32_t depth;
};

using TypeId = const TypeInfo*;
inline constexpr TypeId InvalidType = nullptr;

#define KISEKI_CORE_RTTI_TYPES_DEFINED 1

template <typename T>
struct StaticType;

template <typename T>
constexpr TypeId TypeOf()
{
  return StaticType<T>::Id();
}

constexpr TypeId ParentOf(TypeId id) noexcept
{
  return id ? id->parent : InvalidType;
}

constexpr bool IsA(TypeId self, TypeId target) noexcept
{
  if (!self || !target) {
    return false;
  }
  for (TypeId current = self; current; current = current->parent) {
    if (current == target) {
      return true;
    }
  }
  return false;
}

inline uintptr_t ToInteger(TypeId id) noexcept
{
  return reinterpret_cast<uintptr_t>(id);
}

#define KISEKI_CORE_RTTI_RUNTIME_DEFINED 1

namespace Detail {

template <typename T>
concept HasMemberTypeId = requires(const T* object) {
  { object->GetTypeId() } -> std::convertible_to<TypeId>;
};

template <typename T>
concept HasFreeTypeId = requires(const T* object) {
  { RF_GetTypeId(object) } -> std::convertible_to<TypeId>;
};

template <typename T>
TypeId DynamicTypeOfImpl(const T* object) noexcept
  requires HasMemberTypeId<T>
{
  return object ? object->GetTypeId() : InvalidType;
}

template <typename T>
TypeId DynamicTypeOfImpl(const T* object) noexcept
  requires(!HasMemberTypeId<T> && HasFreeTypeId<T>)
{
  return object ? RF_GetTypeId(object) : InvalidType;
}

}  // namespace Detail

template <typename T>
TypeId DynamicTypeOf(const T* object) noexcept
{
  return Detail::DynamicTypeOfImpl(object);
}

template <typename Target, typename Source>
Target* Cast(Source* object) noexcept
{
  if (!object || !IsA(DynamicTypeOf(object), TypeOf<Target>())) {
    return nullptr;
  }
  return static_cast<Target*>(object);
}

template <typename Target, typename Source>
const Target* Cast(const Source* object) noexcept
{
  if (!object || !IsA(DynamicTypeOf(object), TypeOf<Target>())) {
    return nullptr;
  }
  return static_cast<const Target*>(object);
}

template <typename Target, typename Source>
Target* DynamicCast(Source* object) noexcept
{
  return Cast<Target>(object);
}

template <typename Target, typename Source>
const Target* DynamicCast(const Source* object) noexcept
{
  return Cast<Target>(object);
}

}  // namespace RF

namespace ksk::core {

template <typename T>
constexpr std::string_view TypeName() noexcept
{
#if defined(_MSC_VER)
  constexpr std::string_view signature = __FUNCSIG__;
  constexpr std::string_view prefix =
      "class std::basic_string_view<char,struct std::char_traits<char> > __cdecl "
      "ksk::core::TypeName<";
  constexpr std::string_view suffix = ">(void) noexcept";
#elif defined(__clang__)
  constexpr std::string_view signature = __PRETTY_FUNCTION__;
  constexpr std::string_view prefix = "std::string_view ksk::core::TypeName() [T = ";
  constexpr std::string_view suffix = "]";
#else
  constexpr std::string_view signature = __PRETTY_FUNCTION__;
  constexpr std::string_view prefix =
      "constexpr std::string_view ksk::core::TypeName() [with T = ";
  constexpr std::string_view suffix = "; std::string_view = std::basic_string_view<char>]";
#endif

  const size_t start = signature.find(prefix);
  if (start == std::string_view::npos) {
    return signature;
  }
  const size_t name_start = start + prefix.size();
  const size_t end = signature.rfind(suffix);
  if (end == std::string_view::npos || end < name_start) {
    return signature.substr(name_start);
  }
  return signature.substr(name_start, end - name_start);
}

}  // namespace ksk::core
