//
// Created by creeper on 3/2/24.
//

#ifndef COL_DETECTER_H
#define COL_DETECTER_H
#include <memory>
#include <collision/lbvh.h>

namespace collision {
using Real = double;
template <typename T, int Dim>
class CollisionDetect {
private:
  std::unique_ptr<LBVH<T, Dim>> m_lbvh{};
  std::vector<Real> m_toi{};
  Real global_toi{};
  Real toi() {

  }
public:
  LBVH<T, Dim>& lbvh() {
    return *m_lbvh;
  }
  const LBVH<T, Dim>& lbvh() const {
    return *m_lbvh;
  }
};
}

#endif //COL_DETECTER_H