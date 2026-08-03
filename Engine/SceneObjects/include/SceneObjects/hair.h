#pragma once

#include <Runtime/runtime-scene.h>
#include <glm/glm.hpp>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hair.generated.h"

namespace ksk::scene {

using HairBlock = glm::dvec4;

struct HairMaterial {
  double density = 1300.0;
  double radius = 4e-5;
  double youngsModulus = 4e9;
  double shearModulus = 1.5e9;
};

struct KLASS() HairObject : runtime::SceneObject {
  KLASS_GENERATED_BODY()

  explicit HairObject(std::string tag = {})
      : SceneObject(std::move(tag))
  {
  }

  [[nodiscard]] std::vector<runtime::PropertyDescriptor> listProperties() const override
  {
    return {
        {"material.density", runtime::PropertyType::Scalar, 1, 1, "Hair density"},
        {"material.radius", runtime::PropertyType::Scalar, 1, 1, "Hair radius"},
        {"material.youngsModulus", runtime::PropertyType::Scalar, 1, 1,
         "Hair Young's modulus"},
        {"material.shearModulus", runtime::PropertyType::Scalar, 1, 1,
         "Hair shear modulus"},
    };
  }

  [[nodiscard]] std::vector<runtime::ObjectRef> rodRefs(
      const runtime::RuntimeSceneDesc& scene) const;

  HairMaterial material;
  std::vector<HairBlock> restBlocks;
};

}  // namespace ksk::scene
