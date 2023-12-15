//
// Created by creeper on 10/27/23.
//

#ifndef SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_ADVECT_SOLVER_H_
#define SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_ADVECT_SOLVER_H_
#include <Spatify/arrays.h>
#include <FluidSim/common/fluid-sim.h>
#include <FluidSim/common/sdf.h>

namespace fluid {
class HybridAdvectionSolver2D {
  public:
    virtual void init(int n_particles, int width, int height) = 0;
    virtual void solveG2P(Vec2d* pos,
                          const FaceCentredGrid<Real, Real, 2, 0>* u_grid,
                          const FaceCentredGrid<Real, Real, 2, 1>* v_grid,
                          Real dt) = 0;
    virtual void solveP2G(const Vec2d* pos,
                          FaceCentredGrid<Real, Real, 2, 0>* u_grid,
                          FaceCentredGrid<Real, Real, 2, 1>* v_grid,
                          Real dt) = 0;
    virtual void advect(Vec2d* pos,
                        const FaceCentredGrid<Real, Real, 2, 0>* u_grid,
                        const FaceCentredGrid<Real, Real, 2, 1>* v_grid,
                        const SDF<2>& collider_sdf,
                        Real dt) = 0;
};

class PicAdvector2D : public HybridAdvectionSolver2D {
  public:
    void init(int n, int w, int h) override {
      n_particles = n;
      width = w;
      height = h;
    };
    void solveG2P(Vec2d* pos,
                  const FaceCentredGrid<Real, Real, 2, 0>* u_grid,
                  const FaceCentredGrid<Real, Real, 2, 1>* v_grid,
                  Real dt) override;
    void solveP2G(const Vec2d* pos,
                  FaceCentredGrid<Real, Real, 2, 0>* u_grid,
                  FaceCentredGrid<Real, Real, 2, 1>* v_grid,
                  Real dt) override;
    void advect(Vec2d* pos,
                const FaceCentredGrid<Real, Real, 2, 0>* u_grid,
                const FaceCentredGrid<Real, Real, 2, 1>* v_grid,
                const SDF<2>& collider_sdf,
                Real dt) override;

  private:
    const Vec2d& vel(int i) const { return velocities[i]; }
    Vec2d& vel(int i) { return velocities[i]; }
    int n_particles;
    int width;
    int height;
    std::vector<Vec2d> velocities;
};

class HybridAdvectionSolver3D {
  public:
    HybridAdvectionSolver3D(int n, Real w, Real h, Real d)
      : n_particles(n), width(w), height(h), depth(d) {
    }
    virtual void init(int n_particles, int width, int height, int depth) = 0;
    virtual void solveG2P(std::span<Vec3d> pos,
                          const FaceCentredGrid<Real, Real, 3, 0>* ug,
                          const FaceCentredGrid<Real, Real, 3, 1>* vg,
                          const FaceCentredGrid<Real, Real, 3, 2>* wg,
                          const SDF<3>& collider_sdf,
                          Real dt) = 0;
    virtual void solveP2G(std::span<Vec3d> pos,
                          const FaceCentredGrid<Real, Real, 3, 0>* ug,
                          const FaceCentredGrid<Real, Real, 3, 1>* vg,
                          const FaceCentredGrid<Real, Real, 3, 2>* wg,
                          const SDF<3>& collider_sdf,
                          spatify::Array3D<char>* uValid,
                          spatify::Array3D<char>* vValid,
                          spatify::Array3D<char>* wValid,
                          Real dt) = 0;
    virtual void advect(std::span<Vec3d> pos,
                        const FaceCentredGrid<Real, Real, 3, 0>* ug,
                        const FaceCentredGrid<Real, Real, 3, 1>* vg,
                        const FaceCentredGrid<Real, Real, 3, 2>* wg,
                        const SDF<3>& collider_sdf,
                        Real dt) = 0;
    virtual ~HybridAdvectionSolver3D() = default;

  protected:

    const Vec3d& vel(int i) const { return velocities[i]; }
    Vec3d& vel(int i) { return velocities[i]; }
    std::vector<Vec3d> velocities;
    int n_particles;
    Real width;
    Real height;
    Real depth;
};

class PicAdvector3D : public HybridAdvectionSolver3D {
  public:
    PicAdvector3D(int n, Real width, Real height, Real depth)
      : HybridAdvectionSolver3D(n, width, height, depth) {
      velocities.resize(n);
    }
    void solveG2P(std::span<Vec3d> pos,
                  const FaceCentredGrid<Real, Real, 3, 0>* ug,
                  const FaceCentredGrid<Real, Real, 3, 1>* vg,
                  const FaceCentredGrid<Real, Real, 3, 2>* wg,
                  const SDF<3>& collider_sdf,
                  Real dt) override;
    void solveP2G(std::span<Vec3d> pos,
                  const FaceCentredGrid<Real, Real, 3, 0>* ug,
                  const FaceCentredGrid<Real, Real, 3, 1>* vg,
                  const FaceCentredGrid<Real, Real, 3, 2>* wg,
                  const SDF<3>& collider_sdf,
                  spatify::Array3D<char>* uValid,
                  spatify::Array3D<char>* vValid,
                  spatify::Array3D<char>* wValid,
                  Real dt) override;
    void advect(std::span<Vec3d> pos,
                const FaceCentredGrid<Real, Real, 3, 0>* ug,
                const FaceCentredGrid<Real, Real, 3, 1>* vg,
                const FaceCentredGrid<Real, Real, 3, 2>* wg,
                const SDF<3>& collider_sdf,
                Real dt) override;
  private:
    void handleCollision(const SDF<3>& collider_sdf, Vec3d& p, Vec3d& v) const;
};
}
#endif //SIMCRAFT_FLUIDSIM_INCLUDE_FLUIDSIM_COMMON_ADVECT_SOLVER_H_