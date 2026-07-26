#include <ABD/abd-scene.h>

#include <ABD/abd-subsystem.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ksk::abd {
namespace {

class ABDSubsystemDesc final : public runtime::SubsystemDesc {
 public:
  explicit ABDSubsystemDesc(std::vector<runtime::ObjectId> bodies)
      : bodies_(std::move(bodies))
  {
  }

  [[nodiscard]] std::string_view typeName() const noexcept override
  {
    return "abd";
  }

  void build(runtime::SubsystemBuildContext& context) const override;

 private:
  std::vector<runtime::ObjectId> bodies_;
};

std::unique_ptr<runtime::SubsystemDesc> createABDSubsystemDesc(
    std::vector<runtime::ObjectId> bodies)
{
  return std::make_unique<ABDSubsystemDesc>(std::move(bodies));
}

}  // namespace

void ABDSubsystemDesc::build(runtime::SubsystemBuildContext& context) const
{
  if (bodies_.empty()) {
    throw std::runtime_error("ABD subsystem requires at least one body");
  }

  std::vector<ABDBodyDesc> runtime_bodies;
  runtime_bodies.reserve(bodies_.size());
  const runtime::RuntimeSceneDesc& scene = context.sceneDesc();
  for (const runtime::ObjectId body_id : bodies_) {
    runtime::ObjectRef body_ref = scene.findObjectById(body_id);
    const ABDBodyObjectDesc* body_desc =
        scene.findObjectDescAs<ABDBodyObjectDesc>(body_ref);
    if (!body_desc) {
      throw std::runtime_error("ABD subsystem references a missing body object");
    }
    runtime_bodies.push_back(body_desc->body);
  }

  context.addSubsystem(
      std::make_unique<ABDSubsystem>(context.nextSubsystemId(),
                                     std::move(runtime_bodies),
                                     context.nextScalarOffset(),
                                     context.gravity()),
      std::string(typeName()));
}

runtime::ObjectRef addABDBody(runtime::RuntimeSceneDesc& scene,
                              ABDBodyDesc body,
                              std::string tag)
{
  auto body_desc = std::make_unique<ABDBodyObjectDesc>(std::move(tag));
  body_desc->body = std::move(body);

  runtime::ObjectRef body_ref = scene.registerObject(std::move(body_desc));
  scene.assignObjectToSubsystem("abd", body_ref.id, createABDSubsystemDesc);
  return body_ref;
}

}  // namespace ksk::abd
