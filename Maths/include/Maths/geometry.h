//
// Created by creeper on 5/25/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_GEOMETRY_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_GEOMETRY_H_
#include <Maths/types.h>
namespace maths {
struct HalfPlane {
  Vector<Real, 3> p;
  Vector<Real, 3> n;
  Real unsignedDistance(const Vector<Real, 3>& x) const {
    return abs(n.dot(x - p));
  }
  Real signedDistance(const Vector<Real, 3>& x) const {
    return n.dot(x - p);
}
};
}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_GEOMETRY_H_
