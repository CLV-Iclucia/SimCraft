#include <SceneObjects/abd-body.h>

namespace ksk::scene {

std::vector<runtime::PropertyDescriptor> ABDBodyObject::listProperties()
    const
{
  return {
      {"density", runtime::PropertyType::Scalar, 1, 1, "ABD body density"},
      {"orthogonalStiffness", runtime::PropertyType::Scalar, 1, 1,
       "ABD orthogonality stiffness"},
      {"contactThickness", runtime::PropertyType::Scalar, 1, 1,
       "ABD body contact thickness"},
      {"frictionCoefficient", runtime::PropertyType::Scalar, 1, 1,
       "ABD body friction coefficient override"},
      {"affine", runtime::PropertyType::Scalar, 12, 1,
       "ABD affine transform block"},
  };
}

}  // namespace ksk::scene
