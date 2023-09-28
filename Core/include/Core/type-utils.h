//
// Created by creeper on 23-8-15.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_TYPE_UTILS_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_TYPE_UTILS_H_
#include <Core/core.h>
namespace core {
template <int a, int b> struct compile_time_gcd {
  static constexpr int value = b ? compile_time_gcd<b, a % b>::value : a;
};
// this enables us to use fractions in template arguments!
template <int Numerator, int Denominator> struct Fraction {
  static_assert(Numerator != 0 || Denominator != 1,
                "There is only one definition of zero vector");
  static_assert(Denominator != 0, "Denominator cannot be zero");
  static_assert(compile_time_gcd<Numerator, Denominator>::value == 1,
                "Fraction must be irreducible");
  static constexpr Real value = static_cast<Real>(Numerator) / Denominator;
};

// TODO: if needed, add support for vectors of higher dimensions
template <int NumeratorX, int DenominatorX, int NumeratorY, int DenominatorY>
struct compile_time_vec2 {
  static constexpr Real x = Fraction<NumeratorX, DenominatorX>::value;
  static constexpr Real y = Fraction<NumeratorY, DenominatorY>::value;
};

template <int NumeratorX, int DenominatorX, int NumeratorY, int DenominatorY,
          int NumeratorZ, int DenominatorZ>
struct compile_time_vec3 {
  static constexpr Real x = Fraction<NumeratorX, DenominatorX>::value;
  static constexpr Real y = Fraction<NumeratorY, DenominatorY>::value;
  static constexpr Real z = Fraction<NumeratorZ, DenominatorZ>::value;
};

template <typename T, int Dim> struct is_compile_time_vec {
  static constexpr bool value = false;
};
template <int Dim>
using Zeros = std::conditional_t<Dim == 2, compile_time_vec2<0, 1, 0, 1>,
                                 compile_time_vec3<0, 1, 0, 1, 0, 1>>;
// specialize is_compile_time_vec for compile_time_vec2 and compile_time_vec3
template <int NumeratorX, int DenominatorX, int NumeratorY, int DenominatorY>
struct is_compile_time_vec<
    compile_time_vec2<NumeratorX, DenominatorX, NumeratorY, DenominatorY>, 2> {
  static constexpr bool value = true;
};
template <int NumeratorX, int DenominatorX, int NumeratorY, int DenominatorY,
          int NumeratorZ, int DenominatorZ>
struct is_compile_time_vec<
    compile_time_vec3<NumeratorX, DenominatorX, NumeratorY, DenominatorY,
                      NumeratorZ, DenominatorZ>,
    3> {
  static constexpr bool value = true;
};

} // namespace core
#endif // SIMCRAFT_CORE_INCLUDE_CORE_TYPE_UTILS_H_
