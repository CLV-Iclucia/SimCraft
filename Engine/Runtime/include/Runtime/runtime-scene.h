#pragma once

#include <concepts>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>

#include <Runtime/contact-table.h>
#include <Runtime/dof-layout.h>
#include <Runtime/global-geometry-manager.h>
#include <Runtime/solver-config.h>
#include <Runtime/types.h>
#include <SceneObjects/scene-object.h>

namespace ksk::runtime {

class SimulationContext;
class Subsystem;
class SubsystemBuildContext;
struct RuntimeSceneDesc;

struct ObjectRef {
  ObjectId id = -1;
  ObjectTypeId type = nullptr;
  std::string tag;

  [[nodiscard]] bool isValid() const noexcept
  {
    return id >= 0;
  }

  [[nodiscard]] ObjectTypeId typeId() const noexcept
  {
    return type;
  }

  template <typename ObjectT>
  [[nodiscard]] bool isA() const noexcept
  {
    return RF::IsA(typeId(), elementTypeId<ObjectT>());
  }
};

struct SubsystemBatch {
  std::string type;
  int firstSubsystem = 0;
  int subsystemCount = 0;
  int qOffset = 0;
  int qCount = 0;
  int geometryOffset = 0;
  int geometryCount = 0;
  int auxOffset = 0;
  int auxCount = 0;
};

struct BufferLayout {
  int qScalars = 0;
  int qdotScalars = 0;
  int geometryPoints = 0;
};

struct RuntimeScene {
  DofLayout dofs;
  std::vector<SubsystemBatch> subsystemBatches;
  GlobalGeometryManager geometry;
  ContactStencils contacts;
  ContactStencils frictionReferenceContacts;
  FrictionContactStencils frictionContacts;
  std::unique_ptr<GlobalSolverConfig> solverConfig = makeDefaultSolverConfig();
  glm::dvec3 gravity{0.0, -9.81, 0.0};
  BufferLayout buffers;

  void refreshBufferLayout() noexcept;
};

using ScalarConstraintTarget = std::function<double(double)>;

struct SceneConstraintDesc {
  ObjectRef object;
  std::string property;
  int sample = 0;
  double stiffness = 0.0;
  ScalarConstraintTarget target;
};

struct SceneColliderDesc {
  GeometryMeshDesc mesh;
  glm::dmat4 transform{1.0};
  PhysicalMaterialId physicalMaterial = kDefaultPhysicalMaterial;
  std::string tag;
};

template <typename DescT>
concept SceneObjectDescType =
    std::is_base_of_v<SceneObject, DescT>;

class SubsystemDesc {
 public:
  virtual ~SubsystemDesc() = default;

  [[nodiscard]] virtual std::string_view typeName() const noexcept = 0;
  virtual void build(SubsystemBuildContext& context) const = 0;
};

using SubsystemDescFactory =
    std::function<std::unique_ptr<SubsystemDesc>(std::vector<ObjectId>)>;

class SubsystemBuildContext {
 public:
  // `solver` may be null; a default Newton+IPC config is used in that case.
  // The config is deep-cloned so the caller keeps ownership of its own object.
  explicit SubsystemBuildContext(const GlobalSolverConfig* solver = nullptr,
                                 glm::dvec3 gravity =
                                     glm::dvec3(0.0, -9.81, 0.0),
                                 const RuntimeSceneDesc* scene = nullptr);
  ~SubsystemBuildContext();

  [[nodiscard]] SubsystemIdx nextSubsystemId() const noexcept;
  [[nodiscard]] int nextScalarOffset() const noexcept;
  [[nodiscard]] const glm::dvec3& gravity() const noexcept;
  [[nodiscard]] const RuntimeSceneDesc& sceneDesc() const;

  void addSubsystem(std::unique_ptr<Subsystem> subsystem,
                    std::string typeName);
  [[nodiscard]] SimulationContext finish();

 private:
  std::unique_ptr<GlobalSolverConfig> solver_;
  glm::dvec3 gravity_;
  const RuntimeSceneDesc* scene_ = nullptr;
  std::vector<std::unique_ptr<Subsystem>> subsystems_;
  std::vector<std::string> subsystem_type_names_;
};

struct SceneObjectEntry {
  ObjectId id = -1;
  ObjectTypeId type = nullptr;
  std::string tag;
  std::unique_ptr<SceneObject> element;
};

struct SubsystemObjectGroup {
  std::string type;
  std::vector<ObjectId> objects;
  SubsystemDescFactory factory;
};

struct RuntimeSceneDesc {
  std::unique_ptr<GlobalSolverConfig> solverConfig = makeDefaultSolverConfig();
  glm::dvec3 gravity{0.0, -9.81, 0.0};
  double timeStep = 1.0 / 60.0;
  std::vector<std::unique_ptr<SubsystemDesc>> subsystems;
  std::vector<SubsystemObjectGroup> subsystemObjectGroups;
  std::vector<SceneObjectEntry> elements;
  std::vector<SceneConstraintDesc> constraints;
  std::vector<SceneColliderDesc> colliders;

  void addSubsystem(std::unique_ptr<SubsystemDesc> subsystem);
  void addCollider(SceneColliderDesc collider);
  void assignObjectToSubsystem(std::string subsystemType,
                               ObjectId object,
                               SubsystemDescFactory factory);
  void addConstraint(ObjectRef object,
                     std::string property,
                     int sample,
                     double stiffness,
                     ScalarConstraintTarget target);
  ObjectRef registerObject(ObjectTypeId type, std::string tag = {});
  ObjectRef registerObject(std::unique_ptr<SceneObject> element);
  ObjectRef findObjectById(ObjectId id) const;
  const SceneObject* findObjectDesc(ObjectId id) const;
  SceneObject* findObjectDesc(ObjectId id);
  template <typename ObjectT>
  const ObjectT* findObjectDescAs(ObjectRef ref) const
  {
    static_assert(SceneObjectDescType<ObjectT>,
                  "findObjectDescAs<T> expects a SceneObject type");
    if (!ref.isValid()) {
      return nullptr;
    }
    if (!ref.template isA<ObjectT>()) {
      return nullptr;
    }
    const SceneObject* desc = findObjectDesc(ref.id);
    if (!desc) {
      return nullptr;
    }
    return RF::Cast<ObjectT>(desc);
  }
  template <typename ObjectT>
  ObjectT* findObjectDescAs(ObjectRef ref)
  {
    static_assert(SceneObjectDescType<ObjectT>,
                  "findObjectDescAs<T> expects a SceneObject type");
    if (!ref.isValid()) {
      return nullptr;
    }
    if (!ref.template isA<ObjectT>()) {
      return nullptr;
    }
    SceneObject* desc = findObjectDesc(ref.id);
    if (!desc) {
      return nullptr;
    }
    return RF::Cast<ObjectT>(desc);
  }
  [[nodiscard]] std::vector<PropertyDescriptor> listProperties(ObjectRef ref) const;
  [[nodiscard]] std::vector<ObjectRef> findChildren(ObjectRef ref) const;
  std::vector<ObjectRef> findObjects(ObjectTypeId type) const;
  std::vector<ObjectRef> findObjectsByTag(const std::string& tag) const;
  template <typename ObjectT>
  std::vector<ObjectRef> findObjects() const
  {
    return findObjects(elementTypeId<ObjectT>());
  }
};

[[nodiscard]] SimulationContext buildSimulation(const RuntimeSceneDesc& scene);

}  // namespace ksk::runtime
