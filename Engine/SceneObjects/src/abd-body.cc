#include <SceneObjects/abd-body.h>

namespace ksk::scene {

std::vector<runtime::PropertyDescriptor> ABDBodyObjectDesc::listProperties()
    const
{
  return {
      {"density", runtime::PropertyType::Scalar, 1, 1, "ABD body density"},
      {"contactThickness", runtime::PropertyType::Scalar, 1, 1,
       "ABD body contact thickness"},
      {"affine", runtime::PropertyType::Scalar, 12, 1,
       "ABD affine transform block"},
  };
}

}  // namespace ksk::scene
