#pragma once

#include <DER/rod.h>
#include <Runtime/runtime-scene.h>
#include <SceneObjects/hair.h>
#include <SceneObjects/rod.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "der-scene.generated.h"

namespace ksk::der {

struct DERRodDesc {
  std::vector<RodDof> restBlocks;
  RodMaterial material;
};

struct KLASS() RodObject final : runtime::SceneObject {
  KLASS_GENERATED_BODY()

  explicit RodObject(std::string tag = {})
      : SceneObject(std::move(tag))
  {
  }

  [[nodiscard]] std::vector<runtime::PropertyDescriptor> listProperties() const override
  {
    const int vertexCount = static_cast<int>(restBlocks.size());
    const int twistCount = vertexCount > 0 ? vertexCount - 1 : 0;
    return {
        {"x", runtime::PropertyType::Scalar, 1, vertexCount,
         "Rod vertex x-position samples"},
        {"y", runtime::PropertyType::Scalar, 1, vertexCount,
         "Rod vertex y-position samples"},
        {"z", runtime::PropertyType::Scalar, 1, vertexCount,
         "Rod vertex z-position samples"},
        {"twist", runtime::PropertyType::Scalar, 1, twistCount,
         "Rod edge twist samples"},
    };
  }

  std::vector<RodDof> restBlocks;
  RodMaterial material;
};

runtime::ObjectRef addRod(runtime::RuntimeSceneDesc& scene, DERRodDesc rod);
}  // namespace ksk::der

namespace ksk::scene {

[[nodiscard]] runtime::ObjectRef addHair(runtime::RuntimeSceneDesc& scene,
                                         HairObject hair);

}  // namespace ksk::scene
