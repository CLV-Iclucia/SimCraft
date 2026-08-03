#pragma once

#include <Runtime/runtime-scene.h>
#include <SceneObjects/abd-body.h>

#include <string>

namespace ksk::abd {

using ABDBodyDesc = scene::ABDBodyDesc;
using ABDBodyObject = scene::ABDBodyObject;

[[nodiscard]] runtime::ObjectRef addABDBody(runtime::RuntimeSceneDesc& scene,
                                            ABDBodyDesc body,
                                            std::string tag = {});

}  // namespace ksk::abd
