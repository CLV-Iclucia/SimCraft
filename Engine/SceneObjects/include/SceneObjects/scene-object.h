#pragma once

#include <Runtime/types.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ReflectionMacros.h>
#include "scene-object.generated.h"

namespace ksk::runtime {

enum class PropertyType
{
  Scalar,
  Vector,
};

struct PropertyDescriptor {
  std::string name;
  PropertyType type = PropertyType::Scalar;
  int componentCount = 1;
  int sampleCount = 0;
  std::string metadata;
  ObjectId ownerId = -1;
  std::string ownerTag;
};

struct KLASS() SceneObject {
  KLASS_GENERATED_BODY()
  explicit SceneObject(std::string tag = {})
      : tag(std::move(tag))
  {
  }

  virtual ~SceneObject() = default;

  [[nodiscard]] virtual std::vector<PropertyDescriptor> listProperties() const
  {
    return {};
  }
  [[nodiscard]] const std::string& getTag() const noexcept { return tag; }

  void addChildObject(ObjectId child)
  {
    childObjects.push_back(child);
  }

  [[nodiscard]] const std::vector<ObjectId>& children() const noexcept
  {
    return childObjects;
  }

 private:
  std::string tag;
  std::vector<ObjectId> childObjects;
};

}  // namespace ksk::runtime
