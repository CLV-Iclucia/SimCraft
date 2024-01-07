#ifndef SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_GRIDS_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_GRIDS_H_
#include <Spatify/types.h>
#include <Spatify/type-utils.h>
#include <Spatify/parallel.h>
#include <Spatify/arrays.h>
#include <algorithm>
#include <type_traits>

namespace spatify {
using std::max;
using std::min;
template <typename Scalar, typename T, int Dim, typename Offset, int PaddingSize
              = 0>
class TGrid {
  static_assert(Dim == 2 || Dim == 3, "Grid dimension must be 2 or 3");
  static_assert(
      is_compile_time_vec<Offset, Dim>::value,
      "Offset must be a compile time vector and dimensions must match");
  static_assert(Offset::value < 1.0 && Offset::value >= 0.0,
                "Offset must be in [0, 1)");
};
template <typename Scalar, typename T, typename Offset, int PaddingSize>
class TGrid<Scalar, T, 2, Offset, PaddingSize> {
  public:
    using Range = std::tuple<Vec2i, Vec2i>;
    static Vector<T, 2> offset() {
      return Vector<T, 2>(static_cast<T>(Offset::x), static_cast<T>(Offset::y));
    }
    TGrid() = default;
    explicit TGrid(const Vec2i& size)
      : m_size(size), m_data(size.x * size.y) {
    }
    explicit TGrid(const Vec2i& size, const Vector<T, 2>& grid_spacing)
      : m_size(size), m_grid_spacing(grid_spacing), m_data(size.x * size.y) {
    }
    explicit TGrid(const Vec2i& size, const Vector<T, 2>& grid_spacing,
                   const vector<T>& data)
      : m_size(size), m_grid_spacing(grid_spacing), m_data(data) {
    }
    explicit TGrid(const Vec2i& size, const Vector<T, 2>& grid_spacing,
                   vector<T>&& data)
      : m_size(size), m_grid_spacing(grid_spacing), m_data(std::move(data)) {
    }

    void init(const Vec2i& resolution, const Vector<T, 2>& size,
              const Vector<T, 2>& origin = Vector<Real, 2>(static_cast<T>(0))) {
      m_size = resolution;
      m_grid_spacing = Vector<T, 2>(size.x / resolution.x,
                                    size.y / resolution.y);
      m_data.resize(resolution.x * resolution.y);
      m_origin = origin;
    }

    const Vector<T, 2>& origin() const { return m_origin; }
    const Vec2i& size() const { return m_size; }
    int width() const { return m_size.x; }
    int height() const { return m_size.y; }
    Scalar operator()(int i, int j) const {
      assert(i >= 0 && i < m_size.x && j >= 0 && j < m_size.y);
      return m_data[i * m_size.y + j];
    }
    Scalar& operator()(int i, int j) {
      assert(i >= 0 && i < m_size.x && j >= 0 && j < m_size.y);
      return m_data[i * m_size.y + j];
    }
    Scalar at(int i, int j) const {
      assert(i >= 0 && i < m_size.x && j >= 0 && j < m_size.y);
      return m_data[i * m_size.y + j];
    }
    Scalar& at(int i, int j) {
      assert(i >= 0 && i < m_size.x && j >= 0 && j < m_size.y);
      return m_data[i * m_size.y + j];
    }
    Scalar at(const Vec2i& index) const {
      assert(index.x >= 0 && index.x < m_size.x && index.y >= 0 &&
          index.y < m_size.y);
      return m_data[index.x * m_size.y + index.y];
    }
    Scalar& at(const Vec2i& index) {
      assert(index.x >= 0 && index.x < m_size.x && index.y >= 0 &&
          index.y < m_size.y);
      return m_data[index.x * m_size.y + index.y];
    }
    const Vector<T, 2>& gridSpacing() const { return m_grid_spacing; }
    Vec2i nearest(const Vector<T, 2>& coord) const {
      // get nearest point for coord
      return Vec2i(
          (coord - m_origin - offset() * m_grid_spacing) / m_grid_spacing +
          Vector<T, 2>(0.5));
    }
    Vec2i coordToCellIndex(const Vector<T, 2>& coord) const {
      return Vec2i((coord - m_origin) / m_grid_spacing);
    }
    Vector<T, 2> indexToCoord(const Vec2i& index) const {
      return (Vector<T, 2>(index) + offset()) * m_grid_spacing + m_origin;
    }
    Vector<T, 2> indexToCoord(int x, int y) const {
      return Vector<T, 2>(x + Offset::x, y + Offset::y) * m_grid_spacing +
             m_origin;
    }
    void resize(const Vec2i& size) {
      m_size = size;
      m_data.resize(size.x * size.y);
    }
    void swap(TGrid& other) {
      std::swap(m_size, other.m_size);
      std::swap(m_grid_spacing, other.m_grid_spacing);
      std::swap(m_data, other.m_data);
    }
    template <typename Func>
    void forEach(Func&& func) const {
      for (int i = 0; i < m_size.x; i++)
        for (int j = 0; j < m_size.y; j++) func(i, j);
    }
    template <typename Func>
    void forEach(Func&& func) {
      for (int i = 0; i < m_size.x; i++)
        for (int j = 0; j < m_size.y; j++) func(i, j);
    }
    template <typename Func>
    void parallelForEach(Func&& func) {
      tbb::parallel_for(
          tbb::blocked_range<int>(0, m_size.x),
          [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); i++)
              for (int j = 0; j < m_size.y; j++) func(i, j);
          });
    }
    template <typename Func>
    void parallelForEach(Func&& func) const {
      tbb::parallel_for(
          tbb::blocked_range<int>(0, m_size.x),
          [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); i++)
              for (int j = 0; j < m_size.y; j++) func(i, j);
          });
    }
    // for inside
    template <typename Func>
    void forInside(Func&& func) const {
      for (int i = 1; i < m_size.x - 1; i++)
        for (int j = 1; j < m_size.y - 1; j++) func(i, j);
    }
    template <typename Func>
    void forInside(Func&& func) {
      for (int i = 1; i < m_size.x - 1; i++)
        for (int j = 1; j < m_size.y - 1; j++) func(Vec2i(i, j));
    }
    // TODO: For now this only suits zero offsets
    template <typename Func>
    void forNeighbours(const Vector<T, 2>& p, T r, Func&& func) {
      int lower_bound_x =
          max(static_cast<int>(std::ceil(p.x / m_grid_spacing.x - r)), 0);
      int upper_bound_x =
          min(static_cast<int>(p.x / m_grid_spacing.x + r), m_size.x - 1);
      int lower_bound_y =
          max(static_cast<int>(std::ceil(p.y / m_grid_spacing.y - r)), 0);
      int upper_bound_y =
          min(static_cast<int>(p.y / m_grid_spacing.y + r), m_size.y - 1);
      for (int i = lower_bound_x; i <= upper_bound_x; i++)
        for (int j = lower_bound_y; j <= upper_bound_y; j++) func(i, j);
    }
    template <typename Func>
    void forNeighbours(const Vec2i& p, T r, Func&& func) {
      int lower_bound_x =
          max(static_cast<int>(std::ceil(p.x / m_grid_spacing.x - r)), 0);
      int upper_bound_x =
          min(static_cast<int>(p.x / m_grid_spacing.x + r), m_size.x - 1);
      int lower_bound_y =
          max(static_cast<int>(std::ceil(p.y / m_grid_spacing.y - r)), 0);
      int upper_bound_y =
          min(static_cast<int>(p.y / m_grid_spacing.y + r), m_size.y - 1);
      for (int i = lower_bound_x; i <= upper_bound_x; i++)
        for (int j = lower_bound_y; j <= upper_bound_y; j++) func(Vec2i(i, j));
    }
    template <typename Func>
    void forGridNeighbours(const Vec2i& p, T r, Func&& func) {
      int lower_bound_x = max(p.x - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_x = min(p.x + static_cast<int>(std::ceil(r)),
                              m_size.x - 1);
      int lower_bound_y = max(p.y - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_y = min(p.y + static_cast<int>(std::ceil(r)),
                              m_size.y - 1);

      for (int i = lower_bound_x; i <= upper_bound_x; i++)
        for (int j = lower_bound_y; j <= upper_bound_y; j++) func(Vec2i(i, j));
    }
    Range computeIntersectionNeighbourhoods(const Vec2i& p1, const Vec2i& p2,
                                            T r) {
      int lower_bound_x1 = max(p1.x - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_x1 =
          min(p1.x + static_cast<int>(std::ceil(r)), m_size.x - 1);
      int lower_bound_y1 = max(p1.y - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_y1 =
          min(p1.y + static_cast<int>(std::ceil(r)), m_size.y - 1);
      int lower_bound_x2 = max(p2.x - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_x2 =
          min(p2.x + static_cast<int>(std::ceil(r)), m_size.x - 1);
      int lower_bound_y2 = max(p2.y - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_y2 =
          min(p2.y + static_cast<int>(std::ceil(r)), m_size.y - 1);
      int lower_bound_x = max(lower_bound_x1, lower_bound_x2);
      int upper_bound_x = min(upper_bound_x1, upper_bound_x2);
      int lower_bound_y = max(lower_bound_y1, lower_bound_y2);
      int upper_bound_y = min(upper_bound_y1, upper_bound_y2);
      return std::make_tuple(Vec2i(lower_bound_x, lower_bound_y),
                             Vec2i(upper_bound_x, upper_bound_y));
    }
    void clear() { m_data.clear(); }
    void fill(T value) {
      std::fill(m_data.begin(), m_data.end(), value);
    }
    Scalar* data() { return m_data.data(); }
    const Scalar* data() const { return m_data.data(); }
    virtual ~TGrid() = default;

  protected:
    Vec2i m_size{0, 0};
    Vector<T, 2> m_origin{0.0, 0.0};
    Vector<T, 2> m_grid_spacing{1.0, 1.0};
    vector<Scalar> m_data;
};

template <typename StoredType, typename T, typename Offset, int Padding>
class TGrid<StoredType, T, 3, Offset, Padding> final {
  public:
    using Range = std::tuple<Vec3i, Vec3i>;
    static Vector<T, 3> offset() {
      return Vector<T, 3>(static_cast<T>(Offset::x), static_cast<T>(Offset::y),
                          static_cast<T>(Offset::z));
    }
    TGrid() = default;
    explicit TGrid(const Vec3i& size)
      : m_size(size) {
    }
    explicit TGrid(const Vec3i& size, const Vector<T, 3>& grid_spacing,
                   const vector<StoredType>& data)
      : m_size(size), m_grid_spacing(grid_spacing), m_data(data) {
    }
    explicit TGrid(const Vec3i& size, const Vector<T, 3>& grid_spacing,
                   vector<StoredType>&& data)
      : m_size(size), m_grid_spacing(grid_spacing), m_data(std::move(data)) {
    }
    TGrid(const Vec3i& resolution, const Vector<T, 3>& size,
          const Vector<T, 3>& origin = Vector<Real, 3>(static_cast<T>(0)))
      : m_size(resolution), m_origin(origin),
        m_grid_spacing(size.x / resolution.x,
                       size.y / resolution.y,
                       size.z / resolution.z),
        m_data(resolution) {
    }

    const Vector<T, 3>& origin() const { return m_origin; }
    const Vec3i& size() const { return m_size; }
    int width() const { return m_size.x; }
    int height() const { return m_size.y; }
    int depth() const { return m_size.z; }
    StoredType operator()(int i, int j, int k) const {
      return m_data(i, j, k);
    }
    StoredType& operator()(int i, int j, int k) {
      return m_data(i, j, k);
    }
    const StoredType& operator()(const Vec3i& index) const {
      return m_data(index);
    }
    StoredType at(int i, int j, int k) const {
      return m_data(i, j, k);
    }
    StoredType& at(int i, int j, int k) {
      return m_data(i, j, k);
    }
    StoredType at(const Vec3i& index) const {
      return m_data(index);
    }
    StoredType& at(const Vec3i& index) {
      return m_data(index);
    }
    Vec3i nearest(const Vector<T, 3>& coord) const {
      // get nearest point for coord
      return Vec3i(
          (coord - m_origin - offset() * m_grid_spacing) / m_grid_spacing +
          Vector<T, 3>(0.5));
    }
    const Vector<T, 3>& gridSpacing() const { return m_grid_spacing; }
    Vec3i coordToCellIndex(const Vector<T, 3>& coord) const {
      return Vec3i((coord - m_origin) / m_grid_spacing);
    }
    Vector<T, 3> indexToCoord(const Vec3i& index) const {
      return (Vector<T, 3>(index) + offset()) * m_grid_spacing + m_origin;
    }
    Vector<T, 3> indexToCoord(int x, int y, int z) const {
      return Vector<T, 3>(static_cast<T>(x) + Offset::x,
                          static_cast<T>(y) + Offset::y,
                          static_cast<T>(z) + Offset::z) * m_grid_spacing +
             m_origin;
    }
    void resize(const Vec3i& size) {
      m_size = size;
      m_data.resize(size.x * size.y * size.z);
    }
    template <typename Func>
    void forEach(Func&& func) const {
      for (int i = 0; i < m_size.x; i++)
        for (int j = 0; j < m_size.y; j++)
          for (int k = 0; k < m_size.z; k++) func(i, j, k);
    }
    template <typename Func>
    void forEach(Func&& func) {
      for (int i = 0; i < m_size.x; i++)
        for (int j = 0; j < m_size.y; j++)
          for (int k = 0; k < m_size.z; k++) func(i, j, k);
    }
    template <typename Func>
    void parallelForEach(Func&& func) const {
      tbb::parallel_for(
          tbb::blocked_range<int>(0, m_size.x),
          [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); i++)
              for (int j = 0; j < m_size.y; j++)
                for (int k = 0; k < m_size.z; k++) func(i, j, k);
          });
    }
    template <typename Func>
    void parallelForEach(Func&& func) {
      tbb::parallel_for(
          tbb::blocked_range<int>(0, m_size.x),
          [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); i++)
              for (int j = 0; j < m_size.y; j++)
                for (int k = 0; k < m_size.z; k++) func(i, j, k);
          });
    }
    template <typename Func>
    void forInside(Func&& func) const {
      for (int i = 1; i < m_size.x - 1; i++)
        for (int j = 1; j < m_size.y - 1; j++)
          for (int k = 1; k < m_size.z - 1; k++) func(i, j, k);
    }
    // another for of forEach which supports argument as Vec3i
    template <typename Func>
    void forInside(Func&& func) {
      for (int i = 1; i < m_size.x - 1; i++)
        for (int j = 1; j < m_size.y - 1; j++)
          for (int k = 1; k < m_size.z - 1; k++) func(Vec3i(i, j, k));
    }
    // perform func on all neighbours of p within radius of grid spacing r
    template <typename Func>
    void forNeighbours(const Vector<T, 3>& p, T r, Func&& func) {
      int lower_bound_x =
          max(static_cast<int>(std::ceil(p.x / m_grid_spacing.x - r)), 0);
      int upper_bound_x =
          min(static_cast<int>(p.x / m_grid_spacing.x + r), m_size.x - 1);
      int lower_bound_y =
          max(static_cast<int>(std::ceil(p.y / m_grid_spacing.y - r)), 0);
      int upper_bound_y =
          min(static_cast<int>(p.y / m_grid_spacing.y + r), m_size.y - 1);
      int lower_bound_z =
          max(static_cast<int>(std::ceil(p.z / m_grid_spacing.z - r)), 0);
      int upper_bound_z =
          min(static_cast<int>(p.z / m_grid_spacing.z + r), m_size.z - 1);
      for (int i = lower_bound_x; i <= upper_bound_x; i++)
        for (int j = lower_bound_y; j <= upper_bound_y; j++)
          for (int k = lower_bound_z; k <= upper_bound_z; k++) func(i, j, k);
    }
    // another method to perform func on all neighbours of p within grid spacing r
    // the only difference is that func accepts a Vec3i instead of three ints
    template <typename Func>
    void forNeighbours(const Vec3i& p, T r, Func&& func) {
      int lower_bound_x =
          max(static_cast<int>(std::ceil(p.x / m_grid_spacing.x - r)), 0);
      int upper_bound_x =
          min(static_cast<int>(p.x / m_grid_spacing.x + r), m_size.x - 1);
      int lower_bound_y =
          max(static_cast<int>(std::ceil(p.y / m_grid_spacing.y - r)), 0);
      int upper_bound_y =
          min(static_cast<int>(p.y / m_grid_spacing.y + r), m_size.y - 1);
      int lower_bound_z =
          max(static_cast<int>(std::ceil(p.z / m_grid_spacing.z - r)), 0);
      int upper_bound_z =
          min(static_cast<int>(p.z / m_grid_spacing.z + r), m_size.z - 1);
      for (int i = lower_bound_x; i <= upper_bound_x; i++)
        for (int j = lower_bound_y; j <= upper_bound_y; j++)
          for (int k = lower_bound_z; k <= upper_bound_z; k++)
            func(Vec3i(i, j, k));
    }
    // another method to perform func on all neighbours within grid spacing r
    // the difference is that the centre must be grid indices
    template <typename Func>
    void forGridNeighbours(const Vec3i& p, T r, Func&& func) {
      int lower_bound_x = max(p.x - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_x = min(p.x + static_cast<int>(std::ceil(r)),
                              m_size.x - 1);
      int lower_bound_y = max(p.y - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_y = min(p.y + static_cast<int>(std::ceil(r)),
                              m_size.y - 1);
      int lower_bound_z = max(p.z - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_z = min(p.z + static_cast<int>(std::ceil(r)),
                              m_size.z - 1);
      for (int i = lower_bound_x; i <= upper_bound_x; i++)
        for (int j = lower_bound_y; j <= upper_bound_y; j++)
          for (int k = lower_bound_z; k <= upper_bound_z; k++)
            func(Vec3i(i, j, k));
    }
    // add a method to compute the intersection of two neighbourhoods
    // that's actually the intersection of two cubes
    Range computeIntersectionNeighbourhoods(const Vec3i& p1, const Vec3i& p2,
                                            T r) {
      int lower_bound_x1 = max(p1.x - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_x1 =
          min(p1.x + static_cast<int>(std::ceil(r)), m_size.x - 1);
      int lower_bound_y1 = max(p1.y - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_y1 =
          min(p1.y + static_cast<int>(std::ceil(r)), m_size.y - 1);
      int lower_bound_z1 = max(p1.z - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_z1 =
          min(p1.z + static_cast<int>(std::ceil(r)), m_size.z - 1);
      int lower_bound_x2 = max(p2.x - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_x2 =
          min(p2.x + static_cast<int>(std::ceil(r)), m_size.x - 1);
      int lower_bound_y2 = max(p2.y - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_y2 =
          min(p2.y + static_cast<int>(std::ceil(r)), m_size.y - 1);
      int lower_bound_z2 = max(p2.z - static_cast<int>(std::ceil(r)), 0);
      int upper_bound_z2 =
          min(p2.z + static_cast<int>(std::ceil(r)), m_size.z - 1);
      int lower_bound_x = max(lower_bound_x1, lower_bound_x2);
      int upper_bound_x = min(upper_bound_x1, upper_bound_x2);
      int lower_bound_y = max(lower_bound_y1, lower_bound_y2);
      int upper_bound_y = min(upper_bound_y1, upper_bound_y2);
      int lower_bound_z = max(lower_bound_z1, lower_bound_z2);
      int upper_bound_z = min(upper_bound_z1, upper_bound_z2);
      return std::make_tuple(Vec3i(lower_bound_x, lower_bound_y, lower_bound_z),
                             Vec3i(upper_bound_x, upper_bound_y,
                                   upper_bound_z));
    }
    void clear() { m_data.clear(); }
    void fill(StoredType value) {
      m_data.fill(value);
    }
    void fill(StoredType value, StoredType padding_value) {
      m_data.fill(value, padding_value);
    }
    virtual ~TGrid() = default;

  protected:
    Vec3i m_size{0, 0, 0};
    Vector<T, 3> m_origin{0.0, 0.0, 0.0};
    Vector<T, 3> m_grid_spacing{1.0, 1.0, 1.0};
    GhostArray3D<StoredType, Padding> m_data;
};

template <typename StoredType, typename T, int Dim>
using Grid = TGrid<StoredType, T, Dim, Zeros<Dim>>;

template <typename StoredType, typename T, int Dim>
using CellCentredGrid = TGrid<StoredType, T, Dim, Halfs<Dim>>;

// Specifically used for SDF
template <typename StoredType, typename T, int Dim, int Padding>
using PaddedCellCentredGrid = TGrid<StoredType, T, Dim, Halfs<Dim>, Padding>;

template <typename Scalar, typename T, int Dim, int Axis>
using EdgeCentredGrid = TGrid<Scalar, T, Dim, Half<Dim, Axis>>;

template <int Dim, int Axis>
using FaceCentreOffset = std::conditional_t<
  Dim == 3, compile_time_vec3<Axis != 0, 2, Axis != 1, 2, Axis != 2, 2>,
  compile_time_vec2<Axis != 0, 2, Axis != 1, 2>>;

template <typename Scalar, typename T, int Dim, int Axis>
using FaceCentredGrid = TGrid<Scalar, T, Dim, FaceCentreOffset<Dim, Axis>>;

template <typename Scalar, int Dim>
using ScalarGrid = Grid<Scalar, Real, Dim>;

template <typename Scalar, int Dim>
using VectorGrid = Grid<Vector<Scalar, Dim>, Real, Dim>;
} // namespace core
#endif  // SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_GRIDS_H_