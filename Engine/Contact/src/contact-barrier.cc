#include <Contact/contact-barrier.h>
#include <Contact/gipc/barrier.h>
#include <Contact/gipc/hessian.h>
#include <Contact/gipc/pfpx.h>
#include <Contact/shared/gipc-mollifier.h>
#include <Contact/shared/ipc-distance.h>

#include <Eigen/Dense>
#include <algorithm>

namespace ksk::engine::contact {
namespace {
Real effectiveI5(Real distance_sqr, Real reserved_dist, Real d_hat) {
  const Real distance = std::sqrt(std::max(distance_sqr, 0.0));
  const Real gap = std::max(distance - reserved_dist, d_hat * 1.0e-12);
  return (gap * gap) / (d_hat * d_hat);
}

Real stencilI5(const GIPCBarrierStencil& stencil) {
  using namespace ksk::engine::contact::shared;
  switch (stencil.type) {
    case EBarrierStencil::PP:
      return effectiveI5(shDistanceSqrPointPoint(stencil.x[0], stencil.x[1]),
                         stencil.reservedDist, stencil.dHat);
    case EBarrierStencil::PE:
      return effectiveI5(
          shDistanceSqrPointLine(stencil.x[0], stencil.x[1], stencil.x[2]),
          stencil.reservedDist, stencil.dHat);
    case EBarrierStencil::PT:
      return effectiveI5(shDistanceSqrPointPlane(stencil.x[0], stencil.x[1],
                                                 stencil.x[2], stencil.x[3]),
                         stencil.reservedDist, stencil.dHat);
    case EBarrierStencil::EE:
      return effectiveI5(shDistanceSqrLineLine(stencil.x[0], stencil.x[1],
                                               stencil.x[2], stencil.x[3]),
                         stencil.reservedDist, stencil.dHat);
  }
  return 1.0;
}

template <int N>
void scatterFlat(std::array<glm::dvec3, 4>& out,
                 const Eigen::Matrix<Real, N * 3, 1>& flat);

template <int N>
LocalBarrierHessian scatterFlatHessian(
    const Eigen::Matrix<Real, N * 3, N * 3>& flat);

bool hasMollifierDistance(const GIPCBarrierStencil& stencil) {
  return stencil.mollifierDistance != EBarrierMollifierDistance::None;
}

Real mollifierFeatureDistanceSqr(const GIPCBarrierStencil& stencil) {
  using namespace ksk::engine::contact::shared;
  switch (stencil.mollifierDistance) {
    case EBarrierMollifierDistance::PP:
      return shDistanceSqrPointPoint(stencil.x[0], stencil.x[2]);
    case EBarrierMollifierDistance::PE:
      return shDistanceSqrPointLine(stencil.x[0], stencil.x[2], stencil.x[3]);
    case EBarrierMollifierDistance::EE:
      return shDistanceSqrLineLine(stencil.x[0], stencil.x[1], stencil.x[2],
                                   stencil.x[3]);
    case EBarrierMollifierDistance::None:
      break;
  }
  return 0.0;
}

struct MollifiedFeatureGeometry {
  Real i1 = 0.0;
  Real i2 = 1.0;
  Real epsX = 0.0;
  Real c = 0.0;
  Real f33 = 0.0;
  bool active = false;
};

MollifiedFeatureGeometry mollifiedFeatureGeometry(
    const GIPCBarrierStencil& stencil) {
  MollifiedFeatureGeometry geometry;
  const Real d_hat_sqr = stencil.dHat * stencil.dHat;
  if (stencil.dHat <= 0.0 || d_hat_sqr <= 0.0) {
    return geometry;
  }

  const glm::dvec3 edge_a = stencil.x[1] - stencil.x[0];
  const glm::dvec3 edge_b = stencil.x[3] - stencil.x[2];
  geometry.i1 =
      glm::dot(glm::cross(edge_a, edge_b), glm::cross(edge_a, edge_b));
  if (geometry.i1 <= 0.0) {
    return geometry;
  }

  geometry.epsX = gipc::computeEpsCross(stencil.restX[0], stencil.restX[1],
                                        stencil.restX[2], stencil.restX[3]);
  if (geometry.epsX <= 0.0) {
    return geometry;
  }

  const Real distance_sqr = mollifierFeatureDistanceSqr(stencil);
  const Real gap =
      std::max(std::sqrt(std::max(distance_sqr, 0.0)) - stencil.reservedDist,
               stencil.dHat * 1.0e-12);
  geometry.i2 = (gap * gap) / d_hat_sqr;
  if (geometry.i2 <= 0.0 || geometry.i2 >= 1.0) {
    return geometry;
  }

  geometry.c = std::sqrt(geometry.i1);
  geometry.f33 = std::sqrt(geometry.i2);
  geometry.active = true;
  return geometry;
}

Real mollifiedFeatureEnergy(const GIPCBarrierStencil& stencil) {
  const MollifiedFeatureGeometry geometry = mollifiedFeatureGeometry(stencil);
  if (!geometry.active) {
    return 0.0;
  }

  const gipc::Barrier barrier(stencil.dHat);
  const Real mollifier = gipc::mollifierValue(geometry.i1, geometry.epsX);
  const Real log_i2 = std::log(geometry.i2);
  const Real one_minus_i2 = 1.0 - geometry.i2;
  return stencil.kappa * mollifier * barrier.dHatSqr() * barrier.dHatSqr() *
         one_minus_i2 * one_minus_i2 * log_i2 * log_i2;
}

Real addMollifiedFeatureGradient(const GIPCBarrierStencil& stencil,
                                 LocalBarrierGradient& result) {
  const MollifiedFeatureGeometry geometry = mollifiedFeatureGeometry(stencil);
  if (!geometry.active) {
    return 0.0;
  }

  const auto pfpx = gipc::computePFPx_PEE(
      stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3], stencil.dHat);
  if (!pfpx.valid) {
    return 0.0;
  }

  const gipc::Barrier barrier(stencil.dHat);
  Eigen::Matrix<Real, 9, 1> flatten_pk1 = Eigen::Matrix<Real, 9, 1>::Zero();
  flatten_pk1(4) =
      stencil.kappa *
      barrier.mollifierP1(geometry.i1, geometry.i2, geometry.epsX) * geometry.c;
  flatten_pk1(8) =
      stencil.kappa *
      barrier.mollifierP2(geometry.i1, geometry.i2, geometry.epsX) *
      geometry.f33;

  const Eigen::Matrix<Real, 12, 1> flat_gradient =
      pfpx.PFPx.transpose() * flatten_pk1;
  scatterFlat<4>(result.gradient, flat_gradient);
  return mollifiedFeatureEnergy(stencil);
}

LocalBarrierHessian mollifiedFeatureHessian(const GIPCBarrierStencil& stencil) {
  LocalBarrierHessian result;
  const MollifiedFeatureGeometry geometry = mollifiedFeatureGeometry(stencil);
  if (!geometry.active) {
    return result;
  }

  const auto pfpx = gipc::computePFPx_PEE(
      stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3], stencil.dHat);
  if (!pfpx.valid) {
    return result;
  }

  const gipc::Barrier barrier(stencil.dHat);
  Eigen::Matrix<Real, 9, 9> projected_hessian =
      Eigen::Matrix<Real, 9, 9>::Zero();

  const Real lambda11 =
      stencil.kappa *
      barrier.mollifierLambda11(geometry.i1, geometry.i2, geometry.epsX);
  if (lambda11 > 0.0) {
    projected_hessian(3, 3) += lambda11;
    projected_hessian(5, 5) += lambda11;
  }

  const Real lambda10 =
      stencil.kappa *
      barrier.mollifierLambda10(geometry.i1, geometry.i2, geometry.epsX);
  const Real lambda20 =
      stencil.kappa *
      barrier.mollifierLambda20(geometry.i1, geometry.i2, geometry.epsX);
  const Real lambda_g1g =
      stencil.kappa *
      barrier.mollifierLambdaG1G(geometry.i1, geometry.i2, geometry.epsX);
  const gipc::PD2x2Result pd = gipc::makePD2x2(lambda10, lambda_g1g, lambda20);
  for (int index = 0; index < pd.numPositive; ++index) {
    const Real a = pd.eigenVecs[index].x;
    const Real b = pd.eigenVecs[index].y;
    const Real lambda = pd.eigenValues[index];
    projected_hessian(4, 4) += lambda * a * a;
    projected_hessian(4, 8) += lambda * a * b;
    projected_hessian(8, 4) += lambda * b * a;
    projected_hessian(8, 8) += lambda * b * b;
  }

  return scatterFlatHessian<4>(pfpx.PFPx.transpose() * projected_hessian *
                               pfpx.PFPx);
}

template <int N>
void scatterFlat(std::array<glm::dvec3, 4>& out,
                 const Eigen::Matrix<Real, N * 3, 1>& flat) {
  for (int i = 0; i < N; i++)
    out[i] += glm::dvec3(flat[3 * i + 0], flat[3 * i + 1], flat[3 * i + 2]);
}

template <int N>
LocalBarrierHessian scatterFlatHessian(
    const Eigen::Matrix<Real, N * 3, N * 3>& flat) {
  LocalBarrierHessian result;
  for (int row_point = 0; row_point < N; row_point++) {
    for (int col_point = 0; col_point < N; col_point++) {
      glm::dmat3 block(0.0);
      for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
          block[col][row] = flat(3 * row_point + row, 3 * col_point + col);
        }
      }
      result.blocks[row_point][col_point] = block;
    }
  }
  return result;
}

std::array<glm::dvec3, 4> multiplyHessian(
    const LocalBarrierHessian& hessian,
    const std::array<glm::dvec3, 4>& direction, int count) {
  std::array result{glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0),
                    glm::dvec3(0.0)};
  for (int row = 0; row < count; row++) {
    for (int col = 0; col < count; col++) {
      result[row] += hessian.blocks[row][col] * direction[col];
    }
  }
  return result;
}

template <int N>
Eigen::Matrix<Real, N * 3, 1> flattenDirection(
    const std::array<glm::dvec3, 4>& direction) {
  Eigen::Matrix<Real, N * 3, 1> flat;
  for (int i = 0; i < N; i++) {
    flat[3 * i + 0] = direction[i].x;
    flat[3 * i + 1] = direction[i].y;
    flat[3 * i + 2] = direction[i].z;
  }
  return flat;
}

template <int N, int VecDim>
void appendRankOneGradient(const GIPCBarrierStencil& stencil,
                           const Eigen::Matrix<Real, VecDim, N * 3>& pfpx,
                           const Eigen::Matrix<Real, VecDim, 1>& q0, Real i5,
                           LocalBarrierGradient& result) {
  if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0 || i5 <= 0.0 || i5 >= 1.0) {
    return;
  }

  const gipc::Barrier barrier(stencil.dHat);
  const Real sqrt_i5 = std::sqrt(i5);
  const Eigen::Matrix<Real, VecDim, 1> pk1 =
      q0 * (barrier.gradCoeff(i5) * sqrt_i5);
  const Eigen::Matrix<Real, N * 3, 1> flat_gradient =
      stencil.kappa * pfpx.transpose() * pk1;
  scatterFlat<N>(result.gradient, flat_gradient);
}

template <int N, int VecDim>
LocalBarrierHessian rankOneHessian(
    const GIPCBarrierStencil& stencil,
    const Eigen::Matrix<Real, VecDim, N * 3>& pfpx,
    const Eigen::Matrix<Real, VecDim, 1>& q0, Real i5) {
  LocalBarrierHessian result;
  if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0 || i5 <= 0.0 || i5 >= 1.0) {
    return result;
  }

  const gipc::Barrier barrier(stencil.dHat);
  const Real lambda = stencil.kappa * barrier.clampedLambda0(i5);
  if (lambda <= 0.0) {
    return result;
  }
  const Eigen::Matrix<Real, N * 3, 1> v = pfpx.transpose() * q0;
  const Eigen::Matrix<Real, N * 3, N * 3> flat_hessian =
      lambda * v * v.transpose();
  return scatterFlatHessian<N>(flat_hessian);
}

Real addMollifiedEEGradient(const GIPCBarrierStencil& stencil,
                            LocalBarrierGradient& result) {
  const auto mollified = shared::shMollifiedBarrierEEWithOffset(
      stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3], stencil.restX[0],
      stencil.restX[1], stencil.restX[2], stencil.restX[3], stencil.dHat,
      stencil.kappa, stencil.reservedDist);
  if (!mollified.active) {
    return 0.0;
  }

  Eigen::Matrix<Real, 12, 1> flat_gradient;
  for (int i = 0; i < 12; i++) {
    flat_gradient[i] = mollified.grad[i];
  }
  scatterFlat<4>(result.gradient, flat_gradient);

  return shared::shMollifiedEnergyEEWithOffset(
      stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3], stencil.restX[0],
      stencil.restX[1], stencil.restX[2], stencil.restX[3], stencil.dHat,
      stencil.kappa, stencil.reservedDist);
}

LocalBarrierHessian mollifiedEEHessian(const GIPCBarrierStencil& stencil) {
  LocalBarrierHessian result;
  const auto mollified = shared::shMollifiedBarrierEEWithOffset(
      stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3], stencil.restX[0],
      stencil.restX[1], stencil.restX[2], stencil.restX[3], stencil.dHat,
      stencil.kappa, stencil.reservedDist);
  if (!mollified.active) {
    return result;
  }

  Eigen::Matrix<Real, 12, 12> flat_hessian =
      Eigen::Matrix<Real, 12, 12>::Zero();
  for (int rank = 0; rank < mollified.rank; rank++) {
    Eigen::Matrix<Real, 12, 1> w;
    for (int i = 0; i < 12; i++) {
      w[i] = mollified.w[rank][i];
    }
    flat_hessian += mollified.lam[rank] * w * w.transpose();
  }
  return scatterFlatHessian<4>(flat_hessian);
}
}  // namespace

Real computeGIPCBarrierEnergy(const GIPCBarrierStencil& stencil) {
  if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0) {
    return 0.0;
  }
  if (hasMollifierDistance(stencil)) {
    return mollifiedFeatureEnergy(stencil);
  }
  if (stencil.type == EBarrierStencil::EE &&
      shared::shEEUsesMollifier(stencil.x[0], stencil.x[1], stencil.x[2],
                                stencil.x[3], stencil.restX[0],
                                stencil.restX[1], stencil.restX[2],
                                stencil.restX[3])) {
    return shared::shMollifiedEnergyEEWithOffset(
        stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3],
        stencil.restX[0], stencil.restX[1], stencil.restX[2], stencil.restX[3],
        stencil.dHat, stencil.kappa, stencil.reservedDist);
  }
  gipc::Barrier barrier(stencil.dHat);
  Real i5 = stencilI5(stencil);
  return stencil.kappa * barrier.energy(i5 * stencil.dHat * stencil.dHat);
}

LocalBarrierGradient computeGIPCBarrierGradient(
    const GIPCBarrierStencil& stencil) {
  LocalBarrierGradient result;
  if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0) {
    return result;
  }
  result.energy = computeGIPCBarrierEnergy(stencil);

  switch (stencil.type) {
    case EBarrierStencil::PP: {
      const auto pfpx = gipc::computePFPx_PP(
          stencil.x[0], stencil.x[1], stencil.dHat, stencil.reservedDist);
      if (pfpx.valid)
        appendRankOneGradient<2, 3>(stencil, pfpx.PFPx, pfpx.q0,
                                    stencilI5(stencil), result);
      break;
    }
    case EBarrierStencil::PE: {
      const auto pfpx = gipc::computePFPx_PE(stencil.x[0], stencil.x[1],
                                             stencil.x[2], stencil.dHat);
      if (pfpx.valid)
        appendRankOneGradient<3, 6>(stencil, pfpx.PFPx, pfpx.q0,
                                    stencilI5(stencil), result);
      break;
    }
    case EBarrierStencil::PT: {
      const auto pfpx = gipc::computePFPx_PT(
          stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3], stencil.dHat);
      if (pfpx.valid)
        appendRankOneGradient<4, 9>(stencil, pfpx.PFPx, pfpx.q0,
                                    stencilI5(stencil), result);
      break;
    }
    case EBarrierStencil::EE: {
      if (hasMollifierDistance(stencil)) {
        result.energy = addMollifiedFeatureGradient(stencil, result);
        break;
      }
      if (shared::shEEUsesMollifier(stencil.x[0], stencil.x[1], stencil.x[2],
                                    stencil.x[3], stencil.restX[0],
                                    stencil.restX[1], stencil.restX[2],
                                    stencil.restX[3])) {
        result.energy = addMollifiedEEGradient(stencil, result);
        break;
      }
      const auto pfpx = gipc::computePFPx_EE(
          stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3], stencil.dHat);
      if (pfpx.valid) {
        appendRankOneGradient<4, 9>(stencil, pfpx.PFPx, pfpx.q0,
                                    stencilI5(stencil), result);
      }
      break;
    }
  }
  return result;
}

LocalBarrierHessian computeGIPCBarrierHessian(
    const GIPCBarrierStencil& stencil) {
  LocalBarrierHessian result;
  if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0) {
    return result;
  }

  switch (stencil.type) {
    case EBarrierStencil::PP: {
      const auto pfpx = gipc::computePFPx_PP(
          stencil.x[0], stencil.x[1], stencil.dHat, stencil.reservedDist);
      if (pfpx.valid) {
        return rankOneHessian<2, 3>(stencil, pfpx.PFPx, pfpx.q0,
                                    stencilI5(stencil));
      }
      break;
    }
    case EBarrierStencil::PE: {
      const auto pfpx = gipc::computePFPx_PE(stencil.x[0], stencil.x[1],
                                             stencil.x[2], stencil.dHat);
      if (pfpx.valid) {
        return rankOneHessian<3, 6>(stencil, pfpx.PFPx, pfpx.q0,
                                    stencilI5(stencil));
      }
      break;
    }
    case EBarrierStencil::PT: {
      const auto pfpx = gipc::computePFPx_PT(
          stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3], stencil.dHat);
      if (pfpx.valid) {
        return rankOneHessian<4, 9>(stencil, pfpx.PFPx, pfpx.q0,
                                    stencilI5(stencil));
      }
      break;
    }
    case EBarrierStencil::EE: {
      if (hasMollifierDistance(stencil)) {
        return mollifiedFeatureHessian(stencil);
      }
      if (shared::shEEUsesMollifier(stencil.x[0], stencil.x[1], stencil.x[2],
                                    stencil.x[3], stencil.restX[0],
                                    stencil.restX[1], stencil.restX[2],
                                    stencil.restX[3])) {
        return mollifiedEEHessian(stencil);
      }
      const auto pfpx = gipc::computePFPx_EE(
          stencil.x[0], stencil.x[1], stencil.x[2], stencil.x[3], stencil.dHat);
      if (pfpx.valid) {
        return rankOneHessian<4, 9>(stencil, pfpx.PFPx, pfpx.q0,
                                    stencilI5(stencil));
      }
      break;
    }
  }

  return result;
}

std::array<glm::dvec3, 4> computeGIPCBarrierHessianProduct(
    const GIPCBarrierStencil& stencil,
    const std::array<glm::dvec3, 4>& direction) {
  if (stencil.dHat <= 0.0 || stencil.kappa <= 0.0) {
    return {glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0), glm::dvec3(0.0)};
  }

  return multiplyHessian(computeGIPCBarrierHessian(stencil), direction,
                         stencilPointCount(stencil.type));
}
}  // namespace ksk::engine::contact
