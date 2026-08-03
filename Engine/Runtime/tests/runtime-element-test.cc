#include <Runtime/runtime-scene.h>

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ksk::runtime::test_types {

struct BaseObject {};
struct DerivedObject {};

struct HairObject final : runtime::SceneObject {
  explicit HairObject(std::string tag = {})
      : runtime::SceneObject(std::move(tag))
  {
  }

  [[nodiscard]] ObjectTypeId GetTypeId() const override;
};

struct RodObject final : runtime::SceneObject {
  explicit RodObject(std::string tag = {})
      : runtime::SceneObject(std::move(tag))
  {
  }

  [[nodiscard]] ObjectTypeId GetTypeId() const override;
};

}  // namespace ksk::runtime::test_types

namespace RF {

template <>
struct StaticType<ksk::runtime::test_types::BaseObject> {
  inline static constexpr TypeInfo Info{
      InvalidType, 0, "ksk::runtime::test_types::BaseObject"};

  static constexpr TypeId Id() noexcept { return &Info; }
  static constexpr TypeId Parent() noexcept { return InvalidType; }
};

template <>
struct StaticType<ksk::runtime::test_types::DerivedObject> {
  inline static constexpr TypeInfo Info{
      TypeOf<ksk::runtime::test_types::BaseObject>(),
      1,
      "ksk::runtime::test_types::DerivedObject"};

  static constexpr TypeId Id() noexcept { return &Info; }
  static constexpr TypeId Parent() noexcept
  {
    return TypeOf<ksk::runtime::test_types::BaseObject>();
  }
};

template <>
struct StaticType<ksk::runtime::test_types::HairObject> {
  inline static constexpr TypeInfo Info{
      InvalidType, 0, "ksk::runtime::test_types::HairObject"};

  static constexpr TypeId Id() noexcept { return &Info; }
  static constexpr TypeId Parent() noexcept { return InvalidType; }
};

template <>
struct StaticType<ksk::runtime::test_types::RodObject> {
  inline static constexpr TypeInfo Info{
      InvalidType, 0, "ksk::runtime::test_types::RodObject"};

  static constexpr TypeId Id() noexcept { return &Info; }
  static constexpr TypeId Parent() noexcept { return InvalidType; }
};

}  // namespace RF

namespace ksk::runtime::test_types {

ObjectTypeId HairObject::GetTypeId() const
{
  return elementTypeId<HairObject>();
}

ObjectTypeId RodObject::GetTypeId() const
{
  return elementTypeId<RodObject>();
}

}  // namespace ksk::runtime::test_types

namespace ksk::runtime {
namespace {

using test_types::HairObject;
using test_types::RodObject;

TEST(RuntimeSceneDesc, RegistersAndFindsObjects)
{
  RuntimeSceneDesc scene;
  ObjectRef hair =
      scene.registerObject(std::make_unique<HairObject>("hair"));
  ObjectRef rod =
      scene.registerObject(std::make_unique<RodObject>("der"));

  EXPECT_TRUE(hair.isValid());
  EXPECT_TRUE(rod.isValid());
  EXPECT_NE(hair.id, rod.id);
  EXPECT_TRUE(hair.isA<HairObject>());
  EXPECT_FALSE(hair.isA<RodObject>());

  ObjectRef foundHair = scene.findObjectById(hair.id);
  EXPECT_TRUE(foundHair.isValid());
  EXPECT_EQ(foundHair.tag, "hair");
  EXPECT_TRUE(foundHair.isA<HairObject>());
  EXPECT_EQ(RF::NameOf(foundHair.typeId()),
            std::string_view("ksk::runtime::test_types::HairObject"));

  const HairObject* hairDesc =
      scene.findObjectDescAs<HairObject>(foundHair);
  EXPECT_NE(hairDesc, nullptr);
  EXPECT_EQ(hairDesc->getTag(), "hair");

  ObjectRef foundRod = scene.findObjectById(rod.id);
  EXPECT_TRUE(foundRod.isValid());
  EXPECT_EQ(foundRod.tag, "der");
  EXPECT_TRUE(foundRod.isA<RodObject>());

  const RodObject* rodDesc =
      scene.findObjectDescAs<RodObject>(foundRod);
  EXPECT_NE(rodDesc, nullptr);
  EXPECT_EQ(rodDesc->getTag(), "der");

  EXPECT_FALSE(scene.findObjectById(-1).isValid());
}

TEST(RuntimeSceneDesc, QueriesObjectsByTypeAndTag)
{
  RuntimeSceneDesc scene;
  scene.registerObject(std::make_unique<HairObject>("hair"));
  scene.registerObject(std::make_unique<HairObject>("hair"));

  std::vector<ObjectRef> hairObjects = scene.findObjects<HairObject>();
  EXPECT_EQ(hairObjects.size(), 2);
  EXPECT_TRUE(std::all_of(hairObjects.begin(), hairObjects.end(),
                          [](const ObjectRef& element) {
                            return element.isA<HairObject>();
                          }));

  scene.registerObject(std::make_unique<RodObject>("der"));
  std::vector<ObjectRef> derObjects = scene.findObjectsByTag("der");
  EXPECT_EQ(derObjects.size(), 1);
  EXPECT_EQ(derObjects[0].tag, "der");

  std::vector<ObjectRef> missingTag = scene.findObjectsByTag("missing");
  EXPECT_TRUE(missingTag.empty());
}

TEST(RuntimeSceneDesc, ObjectTypeIdsSupportRttiParentQueries)
{
  RuntimeSceneDesc scene;
  ObjectRef derived =
      scene.registerObject(elementTypeId<test_types::DerivedObject>(),
                           "derived");

  EXPECT_TRUE(derived.isA<test_types::DerivedObject>());
  EXPECT_TRUE(derived.isA<test_types::BaseObject>());

  std::vector<ObjectRef> baseObjects =
      scene.findObjects<test_types::BaseObject>();
  ASSERT_EQ(baseObjects.size(), 1);
  EXPECT_EQ(baseObjects.front().id, derived.id);
}

}  // namespace
}  // namespace ksk::runtime
