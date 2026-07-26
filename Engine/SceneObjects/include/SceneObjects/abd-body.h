#pragma once

#include <Runtime/global-geometry-manager.h>
#include <Runtime/runtime-scene.h>

#include <glm/glm.hpp>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ksk::scene {

struct ABDBodyObject {};

struct ABDBodyDesc {
  runtime::GeometryMeshDesc surface;
  glm::dmat4x3 initialTransform{1.0};
  glm::dmat4x3 initialVelocity{0.0};
  runtime::Real density = 1.0;
  runtime::Real contactThickness = 0.0;
};

struct ABDBodyObjectDesc final : runtime::SceneObjectDesc {
  using ObjectType = ABDBodyObject;

  explicit ABDBodyObjectDesc(std::string tag = {})
      : SceneObjectDesc(std::move(tag))
  {
  }

  [[nodiscard]] runtime::ObjectTypeId typeId() const noexcept override
  {
    return runtime::elementTypeId<ABDBodyObject>();
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "ABDBodyObject";
  }

  [[nodiscard]] std::vector<runtime::PropertyDescriptor> listProperties()
      const override;

  ABDBodyDesc body;
};

}  // namespace ksk::scene
