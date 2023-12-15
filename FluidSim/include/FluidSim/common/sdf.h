//
// Created by creeper on 10/25/23.
//

#ifndef JEOCRAFT_IMPLICITSURFACE_INCLUDE_IMPLICITSURFACE_SDF_H_
#define JEOCRAFT_IMPLICITSURFACE_INCLUDE_IMPLICITSURFACE_SDF_H_
#include <FluidSim/common/fluid-sim.h>
#include <Spatify/grids.h>
#include <Spatify/arrays.h>
#include <Spatify/ns-util.h>

namespace core {
struct Mesh;
}

namespace fluid {
// use grids to store the SDF
template <int Dim>
struct SDF : NonCopyable {
  SDF(const Vector<int, Dim>& resolution, const Vector<Real, Dim>& size,
      const Vector<Real, Dim>& origin = Vector<Real, Dim>(0.0))
    : grid(resolution, size, origin) {
  }
  void init(const Vector<int, Dim>& resolution, const Vector<Real, Dim>& size,
            const Vector<Real, Dim>& origin = Vector<Real, Dim>(0.0)) {
    grid.init(resolution, size, origin);
  }
  void clear() {
    grid.clear();
  }
  Real eval(const Vector<Real, Dim>& p) const {
    if constexpr (Dim == 2) {
      // use bilerp
    } else if constexpr (Dim == 3) {
      Vec3i idx = grid.coordToCellIndex(p);
      // use trilerp to get the value
      int x = idx.x, y = idx.y, z = idx.z;
      Real tx = (p.x - grid.indexToCoord(x, 0, 0).x) / grid.gridSpacing().x;
      Real ty = (p.y - grid.indexToCoord(0, y, 0).y) / grid.gridSpacing().y;
      Real tz = (p.z - grid.indexToCoord(0, 0, z).z) / grid.gridSpacing().z;
      return tx * ty * tz * grid(x, y, z) +
             tx * ty * (1 - tz) * grid(x, y, z + 1) +
             tx * (1 - ty) * tz * grid(x, y + 1, z) +
             tx * (1 - ty) * (1 - tz) * grid(x, y + 1, z + 1) +
             (1 - tx) * ty * tz * grid(x + 1, y, z) +
             (1 - tx) * ty * (1 - tz) * grid(x + 1, y, z + 1) +
             (1 - tx) * (1 - ty) * tz * grid(x + 1, y + 1, z) +
             (1 - tx) * (1 - ty) * (1 - tz) * grid(x + 1, y + 1, z + 1);
    }
  }
  Vector<Real, Dim> gradient(const Vector<int, Dim>& idx) const {
    if constexpr (Dim == 2) {
    } else if constexpr (Dim == 3) {
      int xn = idx.x < width() - 1 ? idx.x + 1 : idx.x;
      int xp = idx.x > 0 ? idx.x - 1 : idx.x;
      int yn = idx.y < height() - 1 ? idx.y + 1 : idx.y;
      int yp = idx.y > 0 ? idx.y - 1 : idx.y;
      int zn = idx.z < depth() - 1 ? idx.z + 1 : idx.z;
      int zp = idx.z > 0 ? idx.z - 1 : idx.z;
      Real dx = idx.x > 0 && idx.x < width() - 1
                  ? 2 * grid.gridSpacing().x
                  : grid.gridSpacing().x;
      Real dy = idx.y > 0 && idx.y < height() - 1
                  ? 2 * grid.gridSpacing().y
                  : grid.gridSpacing().y;
      Real dz = idx.z > 0 && idx.z < depth() - 1
                  ? 2 * grid.gridSpacing().z
                  : grid.gridSpacing().z;
      Real gx = grid(xn, idx.y, idx.z) - grid(xp, idx.y, idx.z);
      Real gy = grid(idx.x, yn, idx.z) - grid(idx.x, yp, idx.z);
      Real gz = grid(idx.x, idx.y, zn) - grid(idx.x, idx.y, zp);
      return Vec3d(gx / dx, gy / dy, gz / dz);
    }
  }
  Vector<Real, Dim> grad(const Vector<Real, Dim>& p) const {
    if constexpr (Dim == 2) {
      // use bilerp
    } else if constexpr (Dim == 3) {
      Vec3i idx = grid.coordToCellIndex(p);
      // use trilerp to get the value
      int x = idx.x, y = idx.y, z = idx.z;
      Real tx = (p.x - grid.indexToCoord(x, 0, 0).x) / grid.gridSpacing().x;
      Real ty = (p.y - grid.indexToCoord(0, y, 0).y) / grid.gridSpacing().y;
      Real tz = (p.z - grid.indexToCoord(0, 0, z).z) / grid.gridSpacing().z;
      Vec3d g000 = gradient(Vec3i(x, y, z));
      Vec3d g001 = gradient(Vec3i(x, y, z + 1));
      Vec3d g010 = gradient(Vec3i(x, y + 1, z));
      Vec3d g011 = gradient(Vec3i(x, y + 1, z + 1));
      Vec3d g100 = gradient(Vec3i(x + 1, y, z));
      Vec3d g101 = gradient(Vec3i(x + 1, y, z + 1));
      Vec3d g110 = gradient(Vec3i(x + 1, y + 1, z));
      Vec3d g111 = gradient(Vec3i(x + 1, y + 1, z + 1));
      return tx * ty * tz * g000 +
             tx * ty * (1 - tz) * g001 +
             tx * (1 - ty) * tz * g010 +
             tx * (1 - ty) * (1 - tz) * g011 +
             (1 - tx) * ty * tz * g100 +
             (1 - tx) * ty * (1 - tz) * g101 +
             (1 - tx) * (1 - ty) * tz * g110 +
             (1 - tx) * (1 - ty) * (1 - tz) * g111;
    }
  }
  int width() const {
    return grid.width();
  }
  int height() const {
    return grid.height();
  }
  int depth() const {
    static_assert(Dim == 3);
    return grid.depth();
  }
  std::vector<Vector<Real, Dim>> positionSamples() const {
    std::vector<Vector<Real, Dim>> position;
    if constexpr (Dim == 2) {
      for (int i = 0; i < grid.width(); i++)
        for (int j = 0; j < grid.height(); j++)
          position.emplace_back(grid.indexToCoord(i, j));
    } else if constexpr (Dim == 3) {
      for (int i = 0; i < grid.width(); i++)
        for (int j = 0; j < grid.height(); j++)
          for (int k = 0; k < grid.depth(); k++)
            position.emplace_back(grid.indexToCoord(i, j, k));
    }
    return position;
  }
  const Vector<Real, Dim>& spacing() const {
    return grid.gridSpacing();
  }
  const Vector<Real, Dim>& origin() const {
    return grid.origin();
  }
  CellCentredGrid<Real, Real, Dim> grid;
};

template <typename T, int Dim>
struct ParticleSystemReconstructor : NonCopyable {
  static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");
  virtual void reconstruct(int n, const Vector<T, Dim>* particles, Real radius,
                           SDF<3>* sdf) = 0;
  virtual SDF<Dim>& sdf() = 0;
  virtual const SDF<Dim>& sdf() const = 0;
  virtual ~ParticleSystemReconstructor() = default;
};

template <typename T, int Dim>
class NaiveReconstructor : ParticleSystemReconstructor<T, Dim> {
};

template <typename T>
class NaiveReconstructor<T, 3> : public ParticleSystemReconstructor<T, 3> {
  public:
    NaiveReconstructor() {
    }
    void reconstruct(int n, const Vector<T, 3>* particles, Real radius,
                     SDF<3>* sdf) {
      ns.resetGrid(sdf->width(), sdf->height(), sdf->depth(),
                   sdf->grid.gridSpacing());
      ns.update(n, particles);
      sdf->grid.fill(1e9);
      sdf->grid.parallelForEach([&](int i, int j, int k) {
        Vector<T, 3> p = sdf->grid.indexToCoord(i, j, k);
        ns.forNeighbours(p, 2 * radius, [&](int idx) {
          Real dis = glm::distance(p, particles[i]);
          if (dis < radius) {
            if (sdf->grid(i, j, k) > 0.0 || sdf->grid(i, j, k) < dis -
                radius)
              sdf->grid(i, j, k) = dis - radius;
          } else
            sdf->grid(i, j, k) = std::min(sdf->grid(i, j, k), dis - radius);
        });
      });
    }

  private:
    spatify::NeighbourSearcher<T, 3> ns;
};

void manifold2SDF(int exact_band, spatify::Array3D<int>& closest,
                  spatify::Array3D<int>& intersection_cnt,
                  const core::Mesh& mesh, SDF<3>* sdf);
}
#endif //JEOCRAFT_IMPLICITSURFACE_INCLUDE_IMPLICITSURFACE_SDF_H_