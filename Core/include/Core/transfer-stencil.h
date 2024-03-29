//
// Created by creeper on 23-8-14.
//

#ifndef SIMCRAFT_MPM_INCLUDE_MPM_TRANSFER_STENCIL_H_
#define SIMCRAFT_MPM_INCLUDE_MPM_TRANSFER_STENCIL_H_
#include <Core/core.h>
#include <Core/utils.h>
namespace core {
struct CubicKernel {
  constexpr static Real kSupportRadius = 2.0;
  static Real N(Real x) {
    Real abs_x = std::abs(x);
    if (abs_x < 1.0)
      return 0.5 * core::cubic(abs_x) - core::sqr(abs_x) + 2.0 / 3.0;
    if (abs_x < 2.0)
      return 1.0 / 6.0 * core::cubic(2.0 - abs_x);
    return 0.0;
  }
  template<typename T, int Dim>
  static Real weight(T h, const Vector<T, Dim> &x) {
    if constexpr (Dim == 2)
      return N(x.x / h) * N(x.y / h);
    if constexpr (Dim == 3)
      return N(x.x / h) * N(x.y / h) * N(x.z / h);
  }
  template<typename T, int Dim>
  static Real weight(const Vector<T, Dim>& h, const Vector<T, Dim> &x) {
    if constexpr (Dim == 2)
      return N(x.x / h.x) * N(x.y / h.y);
    if constexpr (Dim == 3)
      return N(x.x / h.x) * N(x.y / h.y) * N(x.z / h.z);
  }
  static Real dN(Real x) {
    if (x > -1.0 && x < 1.0)
      return 1.5 * x * std::abs(x) - 2.0 * x;
    if (x > -2.0 && x < 2.0)
      return -0.5 * (2.0 - std::abs(x)) * (x > 0.0 ? 1.0 : -1.0);
    return 0.0;
  }
  template<int Dim>
  static Vector<Real, Dim> weightGradient(const Vector<Real, Dim> &h,
                                          const Vector<Real, Dim> &x) {
    Vector<Real, Dim> ret;
    if constexpr (Dim == 2) {
      ret.x = 1.0 / h.x * dN(x.x) * N(x.y);
      ret.y = 1.0 / h.y * N(x.x) * dN(x.y);
    }
    if constexpr (Dim == 3) {
      ret.x = 1.0 / h.x * dN(x.x) * N(x.y) * N(x.z);
      ret.y = 1.0 / h.y * N(x.x) * dN(x.y) * N(x.z);
      ret.z = 1.0 / h.z * N(x.x) * N(x.y) * dN(x.z);
    }
    return ret;
  }
};

struct BSplineKernel {
  constexpr static Real kSupportRadius = 2.0;
  static Real N(Real x) {
    Real abs_x = std::abs(x);
    if (abs_x < 1.0) {
      return 0.5 * core::cubic(abs_x) - core::sqr(abs_x) + 2.0 / 3.0;
    } else if (abs_x < 2.0) {
      return 1.0 / 6.0 * core::cubic(2.0 - abs_x);
    } else {
      return 0.0;
    }
  }
  template<typename T, int Dim>
  static Real weight(const Vector<T, Dim> &h, const Vector<T, Dim> &x) {
    if constexpr (Dim == 2) {
      return N(x.x / h.x) * N(x.y / h.y);
    }
    if constexpr (Dim == 3) {
      return N(x.x / h.x) * N(x.y / h.y) * N(x.z / h.z);
    }
  }
  template<typename T, int Dim>
  static Real weight(T h, const Vector<T, Dim> &x) {
    if constexpr (Dim == 2) {
      return N(x.x / h) * N(x.y / h);
    }
    if constexpr (Dim == 3) {
      return N(x.x / h) * N(x.y / h) * N(x.z / h);
    }
  }
  static Real dN(Real x) {
    if (x > -1.0 && x < 1.0) {
      return 1.5 * x * std::abs(x) - 2.0 * x;
    } else if (x > -2.0 && x < 2.0) {
      return -0.5 * (2.0 - std::abs(x)) * (x > 0.0 ? 1.0 : -1.0);
    } else {
      return 0.0;
    }
  }
  template<int Dim>
  static Vector<Real, Dim> weightGradient(const Vector<Real, Dim> &h,
                                          const Vector<Real, Dim> &x) {
    Vector<Real, Dim> ret;
    if constexpr (Dim == 2) {
      ret.x = 1.0 / h.x * dN(x.x) * N(x.y);
      ret.y = 1.0 / h.y * N(x.x) * dN(x.y);
    }
    if constexpr (Dim == 3) {
      ret.x = 1.0 / h.x * dN(x.x) * N(x.y) * N(x.z);
      ret.y = 1.0 / h.y * N(x.x) * dN(x.y) * N(x.z);
      ret.z = 1.0 / h.z * N(x.x) * N(x.y) * dN(x.z);
    }
  }
};

} // namespace mpm
#endif // SIMCRAFT_MPM_INCLUDE_MPM_TRANSFER_STENCIL_H_
