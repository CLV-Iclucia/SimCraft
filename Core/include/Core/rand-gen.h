// add protection against multiple includes
#ifndef CORE_RAND_GEN_H_
#define CORE_RAND_GEN_H_
#include <Core/core.h>
#include <random>
#include <type_traits>
namespace core {
float randomFloat();
Real randomReal();
template <typename T>
inline T randomScalar() {
    static_assert(std::is_same_v<T, Real> || std::is_same_v<T, float>);
    if constexpr (std::is_same_v<T, Real>) {
        return randomReal();
    } else if constexpr (std::is_same_v<T, float>) {
        return randomFloat();
    }
}
template <typename T, int Dim>
inline Vector<T, Dim> randomVec() {
    Vector<T, Dim> ret;
    for (int i = 0; i < Dim; i++)
        ret[i] = randomScalar<T>();
    return ret;
}

}
#endif