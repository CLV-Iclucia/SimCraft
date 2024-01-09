#include <Core/transfer-stencil.h>
#include <FluidSim/cpu/util.h>
#include <FluidSim/cpu/advect-solver.h>

namespace fluid {
void PicAdvector3D::solveG2P(const std::span<Vec3d> pos,
                             const FaceCentredGrid<Real, Real, 3, 0>& ug,
                             const FaceCentredGrid<Real, Real, 3, 1>& vg,
                             const FaceCentredGrid<Real, Real, 3, 2>& wg,
                             const SDF<3>& collider_sdf,
                             Real dt) {
  Real h = ug.gridSpacing().x;
  for (auto& p : pos)
    vel(&p - pos.data()) = sampleVelocity(p, ug, vg, wg);
}
void PicAdvector3D::solveP2G(const std::span<Vec3d> pos,
                             const FaceCentredGrid<Real, Real, 3, 0>& ug,
                             const FaceCentredGrid<Real, Real, 3, 1>& vg,
                             const FaceCentredGrid<Real, Real, 3, 2>& wg,
                             const SDF<3>& collider_sdf,
                             spatify::Array3D<char>& uValid,
                             spatify::Array3D<char>& vValid,
                             spatify::Array3D<char>& wValid,
                             Real dt) {
  Real h = ug.gridSpacing().x;
  uValid.fill(0);
  vValid.fill(0);
  wValid.fill(0);
  for (auto& p : pos) {
    auto& h = ug.gridSpacing();
    Vec3i u_idx = ug.nearest(p);
    Vec3i v_idx = vg.nearest(p);
    Vec3i w_idx = wg.nearest(p);
    Real u = 0.0, v = 0.0, w = 0.0;
    Real w_u = 0.0, w_v = 0.0, w_w = 0.0;
    for (int j = -1; j <= 1; j++) {
      for (int k = -1; k <= 1; k++) {
        for (int l = -1; l <= 1; l++) {
          if (u_idx.x + j >= 0 && u_idx.x + j < ug.width() &&
              u_idx.y + k >= 0 && u_idx.y + k < ug.height() &&
              u_idx.z + l >= 0 && u_idx.z + l < ug.depth()) {
            u += ug(u_idx + Vec3i(j, k, l)) *
                core::CubicKernel::weight<Real, 3>(
                    h, p - ug.indexToCoord(u_idx + Vec3i(j, k, l)));
            w_u += core::CubicKernel::weight<Real, 3>(
                h, p - ug.indexToCoord(u_idx + Vec3i(j, k, l)));
            uValid(u_idx + Vec3i(j, k, l)) = 1;
          }
          if (v_idx.x + j >= 0 && v_idx.x + j < vg.width() &&
              v_idx.y + k >= 0 && v_idx.y + k < vg.height() &&
              v_idx.z + l >= 0 && v_idx.z + l < vg.depth()) {
            v += vg.at(v_idx + Vec3i(j, k, l)) *
                core::CubicKernel::weight<Real, 3>(
                    h, p - vg.indexToCoord(v_idx + Vec3i(j, k, l)));
            w_v += core::CubicKernel::weight<Real, 3>(
                h, p - vg.indexToCoord(v_idx + Vec3i(j, k, l)));
            vValid(v_idx + Vec3i(j, k, l)) = 1;
          }
          if (w_idx.x + j >= 0 && w_idx.x + j < wg.width() &&
              w_idx.y + k >= 0 && w_idx.y + k < wg.height() &&
              w_idx.z + l >= 0 && w_idx.z + l < wg.depth()) {
            w += wg(w_idx + Vec3i(j, k, l)) *
                core::CubicKernel::weight<Real, 3>(
                    h, p - wg.indexToCoord(w_idx + Vec3i(j, k, l)));
            w_w += core::CubicKernel::weight<Real, 3>(
                h, p - wg.indexToCoord(w_idx + Vec3i(j, k, l)));
            wValid(w_idx + Vec3i(j, k, l)) = 1;
          }
        }
      }
    }
    int i = &p - pos.data();
    vel(i).x = w_u > 0.0 ? u / w_u : 0.0;
    assert(notNan(vel(i).x));
    vel(i).y = w_v > 0.0 ? v / w_v : 0.0;
    assert(notNan(vel(i).y));
    vel(i).z = w_w > 0.0 ? w / w_w : 0.0;
    assert(notNan(vel(i).z));
  }
}

void PicAdvector3D::handleCollision(const SDF<3>& collider_sdf, Vec3d& p,
                                    Vec3d& v) const {
  Real d = collider_sdf.eval(p);
  if (d < 0.0) {
    Vec3d normal = normalize(collider_sdf.grad(p));
    p -= d * normal;
    v -= dot(v, normal) * normal;
  }
  if (p.x < 0.0) {
    p.x = 0.0;
    v.x = 0.0;
  }
  if (p.x > width) {
    p.x = width;
    v.x = 0.0;
  }
  if (p.y < 0.0) {
    p.y = 0.0;
    v.y = 0.0;
  }
  if (p.y > height) {
    p.y = height;
    v.y = 0.0;
  }
  if (p.z < 0.0) {
    p.z = 0.0;
    v.z = 0.0;
  }
  if (p.z > depth) {
    p.z = depth;
    v.z = 0.0;
  }
}
void PicAdvector3D::advect(const std::span<Vec3d> pos,
                           const FaceCentredGrid<Real, Real, 3, 0>& ug,
                           const FaceCentredGrid<Real, Real, 3, 1>& vg,
                           const FaceCentredGrid<Real, Real, 3, 2>& wg,
                           const SDF<3>& collider_sdf,
                           Real dt) {
  for (auto& p : pos) {
    int i = &p - pos.data();
    Vec3d mid_p = p + 0.5 * vel(i) * dt;
    handleCollision(collider_sdf, p, vel(i));
    Vec3d mid_v = sampleVelocity(mid_p, ug, vg, wg);
    p += mid_v * 0.5 * dt;
    handleCollision(collider_sdf, p, vel(i));
  }
}
} // namespace fluid