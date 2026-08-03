#include <SceneObjects/abd-body.h>

namespace ksk::scene {

std::vector<runtime::PropertyDescriptor> ABDBodyObject::listProperties()
    const
{
  return {
      {"density", runtime::PropertyType::Scalar, 1, 1, "ABD body density"},
      {"orthogonalStiffness", runtime::PropertyType::Scalar, 1, 1,
       "ABD orthogonality stiffness"},
      {"physicalMaterial", runtime::PropertyType::Scalar, 1, 1,
       "ABD body physical material id"},
      {"affine", runtime::PropertyType::Scalar, 12, 1,
       "ABD affine transform block"},
  };
}

}  // namespace ksk::scene
