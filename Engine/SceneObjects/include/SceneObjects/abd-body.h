#pragma once

#include <Runtime/global-geometry-manager.h>
#include <Runtime/runtime-scene.h>

#include <glm/glm.hpp>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "abd-body.generated.h"

namespace ksk::scene {

struct ABDBodyDesc {
  runtime::GeometryMeshDesc surface;
  glm::dmat4x3 initialTransform{1.0};
  glm::dmat4x3 initialVelocity{0.0};
  runtime::Real density = 1.0;
  runtime::Real orthogonalStiffness = 1.0e6;
  runtime::PhysicalMaterialId physicalMaterial =
      runtime::kDefaultPhysicalMaterial;
};

struct KLASS() ABDBodyObject final : runtime::SceneObject {
  KLASS_GENERATED_BODY()

  explicit ABDBodyObject(std::string tag = {}) : SceneObject(std::move(tag))
  {
  }

  [[nodiscard]] std::vector<runtime::PropertyDescriptor> listProperties()
      const override;

  ABDBodyDesc body;
};

}  // namespace ksk::scene
