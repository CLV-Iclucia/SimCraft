//
// Created by creeper on 10/24/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_SCENE_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_SCENE_H_

namespace fem {
struct SceneInfo {
  std::unique_ptr<System> system{};
  std::unique_ptr<Integrator> integrator{};
};

struct SceneBuilder {
  std::unique_ptr<SceneInfo> buildFromJson(const core::JsonNode& json);
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_SCENE_H_
