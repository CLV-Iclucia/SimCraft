#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Maths/types.h>
#include <vector>
#include <cassert>
#include <cmath>

namespace sim::maths {

/// Contiguous array of glm::vec<N, Real> blocks.
/// Memory layout: [x0 y0 z0 | x1 y1 z1 | ...] — identical to double[numBlocks*N].
/// Zero-copy bridgeable to Eigen::Map<VectorXd>.
template <int N>
class BlockVector {
  static_assert(N >= 2 && N <= 4, "Block size must be 2, 3, or 4");

public:
  using Block = glm::vec<N, Real>;

  BlockVector() = default;
  explicit BlockVector(int numBlocks) : m_data(numBlocks, Block(0)) {}
  BlockVector(int numBlocks, const Block &init) : m_data(numBlocks, init) {}

  // --- Size ---
  [[nodiscard]] int numBlocks() const { return static_cast<int>(m_data.size()); }
  [[nodiscard]] int scalarSize() const { return numBlocks() * N; }
  [[nodiscard]] bool empty() const { return m_data.empty(); }

  void resize(int numBlocks) { m_data.resize(numBlocks, Block(0)); }
  void resizeLike(const BlockVector &other) { m_data.resize(other.numBlocks(), Block(0)); }

  // --- Block access ---
  Block &operator[](int i) {
    assert(i >= 0 && i < numBlocks());
    return m_data[i];
  }
  const Block &operator[](int i) const {
    assert(i >= 0 && i < numBlocks());
    return m_data[i];
  }

  // --- Scalar data access ---
  Real *data() { return glm::value_ptr(m_data[0]); }
  const Real *data() const { return glm::value_ptr(m_data[0]); }

  // --- Eigen bridge (zero-copy) ---
  Eigen::Map<Vector<Real, Dynamic>> asEigen() {
    return {data(), scalarSize()};
  }
  Eigen::Map<const Vector<Real, Dynamic>> asEigen() const {
    return {data(), scalarSize()};
  }

  // --- Vector space operations ---
  [[nodiscard]] Real dot(const BlockVector &other) const {
    assert(numBlocks() == other.numBlocks());
    Real sum = 0.0;
    for (int i = 0; i < numBlocks(); i++)
      sum += glm::dot(m_data[i], other.m_data[i]);
    return sum;
  }

  [[nodiscard]] Real squaredNorm() const {
    Real sum = 0.0;
    for (int i = 0; i < numBlocks(); i++)
      sum += glm::dot(m_data[i], m_data[i]);
    return sum;
  }

  [[nodiscard]] Real norm() const { return std::sqrt(squaredNorm()); }

  /// Infinity norm (max absolute value across all components)
  [[nodiscard]] Real infNorm() const {
    Real maxVal = 0.0;
    for (int i = 0; i < numBlocks(); i++)
      for (int d = 0; d < N; d++)
        maxVal = std::max(maxVal, std::abs(m_data[i][d]));
    return maxVal;
  }

  /// this += a * other
  void axpy(Real a, const BlockVector &other) {
    assert(numBlocks() == other.numBlocks());
    for (int i = 0; i < numBlocks(); i++)
      m_data[i] += static_cast<Real>(a) * other.m_data[i];
  }

  void setZero() {
    for (auto &b : m_data)
      b = Block(0);
  }

  void copyFrom(const BlockVector &other) {
    m_data = other.m_data;
  }

  // --- Arithmetic operators ---
  BlockVector &operator+=(const BlockVector &other) {
    axpy(1.0, other);
    return *this;
  }

  BlockVector &operator-=(const BlockVector &other) {
    axpy(-1.0, other);
    return *this;
  }

  BlockVector &operator*=(Real s) {
    for (auto &b : m_data)
      b *= s;
    return *this;
  }

private:
  std::vector<Block> m_data;
};

using BlockVector3 = BlockVector<3>;

} // namespace sim::maths
