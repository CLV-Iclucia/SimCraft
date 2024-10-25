//
// Created by creeper on 10/24/24.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_JSON_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_JSON_H_
#include <string>
#include <vector>
#include <variant>
#include <memory>
#include <Core/core.h>
namespace core {
struct JsonNode;
using JsonDict = std::unordered_map<std::string, JsonNode>;
using JsonList = std::vector<JsonNode>;
struct JsonNode {
  JsonNode() = default;
  explicit JsonNode(std::nullptr_t) : m_value(nullptr) {}
  explicit JsonNode(int value) : m_value(value) {}
  explicit JsonNode(Real value) : m_value(value) {}
  explicit JsonNode(bool value) : m_value(value) {}
  explicit JsonNode(const std::string &value) : m_value(value) {}
  explicit JsonNode(const JsonDict &value) : m_value(value) {}
  explicit JsonNode(const JsonList &value) : m_value(value) {}
  JsonNode(const JsonNode &other) = default;
  JsonNode(JsonNode &&other) noexcept: m_value(std::move(other.m_value)) { other.m_value = nullptr; }

  [[nodiscard]] bool isNull() const { return std::holds_alternative<std::nullptr_t>(m_value); }
  [[nodiscard]] bool isInt() const { return std::holds_alternative<int>(m_value); }
  [[nodiscard]] bool isReal() const { return std::holds_alternative<Real>(m_value); }
  [[nodiscard]] bool isBool() const { return std::holds_alternative<bool>(m_value); }
  [[nodiscard]] bool isString() const { return std::holds_alternative<std::string>(m_value); }
  [[nodiscard]] bool isDict() const { return std::holds_alternative<JsonDict>(m_value); }
  [[nodiscard]] bool isList() const { return std::holds_alternative<JsonList>(m_value); }

  JsonNode &value(const std::string &key) {
    if (!isDict())
      throw std::runtime_error("JsonNode is not a dictionary");
    auto &dict = std::get<JsonDict>(m_value);
    if (auto it = dict.find(key); it != dict.end())
      return it->second;
    throw std::runtime_error("Key not found");
  }
  const JsonNode &value(const std::string &key) const {
    if (!isDict())
      throw std::runtime_error("JsonNode is not a dictionary");
    auto &dict = std::get<JsonDict>(m_value);
    if (auto it = dict.find(key); it != dict.end())
      return it->second;
    throw std::runtime_error("Key not found");
  }

 private:
  std::variant<std::nullptr_t, int, Real, bool, std::string, JsonDict, JsonList> m_value{};
};
}
#endif //SIMCRAFT_CORE_INCLUDE_CORE_JSON_H_
