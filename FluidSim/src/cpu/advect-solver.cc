#include <FluidSim/cpu/stencil.h>
#include <FluidSim/cpu/util.h>
#include <FluidSim/cpu/advect-solver.h>

namespace fluid::cpu {
void PicAdvector3D::solveGtoP(const GtoPDetails &details) {
  auto &&[pos, ug, vg, wg, collider_sdf, dt] = details;
  int n = pos.size();
  for (int i = 0; i < n; i++) {
    vel(i) = sampleVelocity(pos[i], ug, vg, wg);
  }
}
void PicAdvector3D::solvePtoG(const PtoGDetails &details) {
  auto &&[pos, ug, vg, wg, collider_sdf, uw, vw, ww, uValid, vValid, wValid, dt] = details;
  int n = pos.size();
  for (int idx = 0; idx < n; idx++) {
    auto &p = pos[idx];
    auto &h = ug.gridSpacing();
    Vec3i u_idx = ug.nearest(p);
    Vec3i v_idx = vg.nearest(p);
    Vec3i w_idx = wg.nearest(p);
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        for (int k = -1; k <= 1; k++) {
          if (u_idx.x + i >= 0 && u_idx.x + i < ug.width() &&
              u_idx.y + j >= 0 && u_idx.y + j < ug.height() &&
              u_idx.z + k >= 0 && u_idx.z + k < ug.depth()) {
            Vec3d pn = ug.indexToCoord(u_idx + Vec3i(i, j, k));
            Real w_u = CubicKernel::weight<Real, 3>(h, p - pn);
            ug(u_idx + Vec3i(i, j, k)) += vel(idx).x * w_u;
            uw(u_idx + Vec3i(i, j, k)) += w_u;
          }
          if (v_idx.x + i >= 0 && v_idx.x + i < vg.width() &&
              v_idx.y + j >= 0 && v_idx.y + j < vg.height() &&
              v_idx.z + k >= 0 && v_idx.z + k < vg.depth()) {
            Vec3d pn = vg.indexToCoord(v_idx + Vec3i(i, j, k));
            Real w_v = CubicKernel::weight<Real, 3>(h, p - pn);
            vg(v_idx + Vec3i(i, j, k)) += vel(idx).y * w_v;
            vw(v_idx + Vec3i(i, j, k)) += w_v;
          }
          if (w_idx.x + i >= 0 && w_idx.x + i < wg.width() &&
              w_idx.y + j >= 0 && w_idx.y + j < wg.height() &&
              w_idx.z + k >= 0 && w_idx.z + k < wg.depth()) {
            Vec3d pn = wg.indexToCoord(w_idx + Vec3i(i, j, k));
            Real w_w = CubicKernel::weight<Real, 3>(h, p - pn);
            wg(w_idx + Vec3i(i, j, k)) += vel(idx).z * w_w;
            ww(w_idx + Vec3i(i, j, k)) += w_w;
          }
        }
      }
    }
  }
  uValid.fill(0);
  vValid.fill(0);
  wValid.fill(0);
  ug.parallelForEach([&](int i, int j, int k) {
    if (uw(i, j, k) > 0.0) {
      ug(i, j, k) /= uw(i, j, k);
      uValid(i, j, k) = 1;
    } else {
      ug(i, j, k) = 0.0;
      uValid(i, j, k) = 0;
    }
  });
  vg.parallelForEach([&](int i, int j, int k) {
    if (vw(i, j, k) > 0.0) {
      vg(i, j, k) /= vw(i, j, k);
      vValid(i, j, k) = 1;
    } else {
      vg(i, j, k) = 0.0;
      vValid(i, j, k) = 0;
    }
  });
  wg.parallelForEach([&](int i, int j, int k) {
    if (ww(i, j, k) > 0.0) {
      wg(i, j, k) /= ww(i, j, k);
      wValid(i, j, k) = 1;
    } else {
      wg(i, j, k) = 0.0;
      wValid(i, j, k) = 0;
    }
  });
}

void PicAdvector3D::handleCollision(const SDF<3> &collider_sdf, Vec3d &p,
                                    Vec3d &v) const {
  Real d = collider_sdf.eval(p);
  if (d < 0.0) {
    Vec3d normal = normalize(collider_sdf.grad(p));
    p -= d * normal;
    v -= std::min(dot(v, normal), 0.0) * normal;
  }
  if (p.x < 0.0) {
    p.x = 0.0;
    if (v.x < 0.0) v.x = 0.0;
  }
  if (p.x > width) {
    p.x = width;
    if (v.x > 0.0)
      v.x = 0.0;
  }
  if (p.y < 0.0) {
    p.y = 0.0;
    if (v.y < 0.0)
      v.y = 0.0;
  }
  if (p.y > height) {
    p.y = height;
    if (v.y > 0.0)
      v.y = 0.0;
  }
  if (p.z < 0.0) {
    p.z = 0.0;
    if (v.z < 0.0)
      v.z = 0.0;
  }
  if (p.z > depth) {
    p.z = depth;
    if (v.z > 0.0)
      v.z = 0.0;
  }
}
void PicAdvector3D::advect(const AdvectDetails &details) {
  auto&& [pos, ug, vg, wg, collider_sdf, dt] = details;
  int n = pos.size();
  for (int i = 0; i < n; i++) {
    auto &p = pos[i];
    Vec3d mid_p = p + 0.5 * vel(i) * dt;
    handleCollision(collider_sdf, p, vel(i));
    Vec3d mid_v = sampleVelocity(mid_p, ug, vg, wg);
    p = mid_p + mid_v * 0.5 * dt;
    handleCollision(collider_sdf, p, vel(i));
  }
}
} // namespace fluid