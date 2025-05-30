//
// Created by CreeperIclucia-Vader on 25-5-16.
//

#pragma once

#include <span>
#include <fem/types.h>
#include <fem/simplex.h>

namespace sim::fem {

struct PointTrajectory {
  const VecXd& x;
  const VecXd& p;
  Real toi = 1.0;
  int global_idx{};
  [[nodiscard]] BBox<Real, 3> bbox() const {
    auto v = x.segment<3>(global_idx * 3);
    auto u = p.segment<3>(global_idx * 3) * toi;
    return BBox<Real, 3>({v(0), v(1), v(2)}).expand({v(0) + u(0), v(1) + u(1), v(2) + u(2)});
  }
};

struct TriangleTrajectory {
  CSubVector<Real> dofView;
  const Triangle& triangle;
  CSubVector<Real> p;
  Real toi = 1.0;
};

}