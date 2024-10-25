//
// Created by creeper on 10/27/23.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_ADVECT_SOLVER_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_ADVECT_SOLVER_H_
#include <Spatify/arrays.h>
#include <FluidSim/cpu/sdf.h>

namespace fluid::cpu {

struct GtoPDetails {
  std::span<Vec3d> pos;
  const FaceCentredGrid<Real, Real, 3, 0>& ug;
  const FaceCentredGrid<Real, Real, 3, 1>& vg;
  const FaceCentredGrid<Real, Real, 3, 2>& wg;
  const SDF<3>& collider_sdf;
  Real dt;
};

struct PtoGDetails {
  std::span<Vec3d> pos;
  FaceCentredGrid<Real, Real, 3, 0>& ug;
  FaceCentredGrid<Real, Real, 3, 1>& vg;
  FaceCentredGrid<Real, Real, 3, 2>& wg;
  const SDF<3>& collider_sdf;
  Array3D<Real>& uw;
  Array3D<Real>& vw;
  Array3D<Real>& ww;
  Array3D<char>& uValid;
  Array3D<char>& vValid;
  Array3D<char>& wValid;
  Real dt;
};

struct AdvectDetails {
  std::span<Vec3d> pos;
  const FaceCentredGrid<Real, Real, 3, 0> &ug;
  const FaceCentredGrid<Real, Real, 3, 1> &vg;
  const FaceCentredGrid<Real, Real, 3, 2> &wg;
  const SDF<3> &collider_sdf;
  Real dt;
};

class HybridAdvectionSolver3D {
  public:
    HybridAdvectionSolver3D(int n, Real w, Real h, Real d)
      : n_particles(n), width(w), height(h), depth(d) {
    }
    virtual void solveGtoP(const GtoPDetails &details) = 0;
    virtual void solvePtoG(const PtoGDetails &details) = 0;
    virtual void advect(const AdvectDetails &details) = 0;
    virtual ~HybridAdvectionSolver3D() = default;

  protected:
    [[nodiscard]] const Vec3d& vel(int i) const { return velocities[i]; }
    Vec3d& vel(int i) { return velocities[i]; }
    std::vector<Vec3d> velocities{};
    int n_particles;
    Real width{};
    Real height{};
    Real depth{};
};

class PicAdvector3D final : public HybridAdvectionSolver3D {
  public:
    PicAdvector3D(int n, Real width, Real height, Real depth)
      : HybridAdvectionSolver3D(n, width, height, depth) {
      velocities.resize(n);
    }
    void solveGtoP(const GtoPDetails &details) override;
    void solvePtoG(const PtoGDetails &details) override;
    void advect(const AdvectDetails &details) override;

  private:
    void handleCollision(const SDF<3>& collider_sdf, Vec3d& p, Vec3d& v) const;
    std::unique_ptr<Array3D<Real>> uw{}, vw{}, ww{};
};
}
#endif //SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_ADVECT_SOLVER_H_