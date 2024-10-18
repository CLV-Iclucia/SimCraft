//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_SPARSE_MATRIX_BUILDER_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_SPARSE_MATRIX_BUILDER_H_
#include <Maths/types.h>
#include <iostream>
#include <vector>
#include <concepts>
namespace maths {

template<typename T>
concept IntType = std::is_integral_v<T> || std::is_convertible_v<T, int>;

template<typename Scalar>
class SparseMatrixBuilder {
 public:
  using Triplet = Eigen::Triplet<Scalar>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  SparseMatrixBuilder() = default;
  SparseMatrixBuilder(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {}
  SparseMatrixBuilder(size_t rows, size_t cols, size_t nnz) : m_rows(rows), m_cols(cols) {
    m_triplets.reserve(nnz);
  }

  void addElement(int row, int col, Scalar value) {
    if (row >= m_rows || col >= m_cols) {
      std::cerr << "invalid row or col" << std::endl;
      return;
    }
    m_triplets.emplace_back(row, col, value);
  }

  template<int Dim, IntType... Indices>
  void assemble(const Matrix<Scalar, Dim, Dim> &matrix, Indices... indices) {
    static_assert(Dim > 0, "dimension must be greater than 0");
    static_assert(sizeof...(indices) == Dim, "number of indices must match the dimension of the matrix");
    std::array<int, Dim> index_array{indices...};
    for (int i = 0; i < Dim; i++)
      for (int j = 0; j < Dim; j++)
        addElement(index_array[i], index_array[j], matrix(i, j));
  }

  template<int Dim, int BlockSize, IntType... BlockIndices>
  void assembleBlock(const Matrix<Scalar, Dim, Dim> &matrix, BlockIndices... block_indices) {
    static_assert(Dim > 0, "dimension must be greater than 0");
    static_assert(BlockSize > 0, "block size must be greater than 0");
    assert(m_rows % BlockSize == 0 && m_cols % BlockSize == 0);
    assert(Dim % BlockSize == 0);
    constexpr int num_blocks = Dim / BlockSize;
    static_assert(sizeof...(block_indices) == num_blocks, "number of block indices must match the number of blocks");
    std::array<int, num_blocks> block_index_array{block_indices...};
    for (int i = 0; i < num_blocks; i++)
      for (int j = 0; j < num_blocks; j++)
        for (int k = 0; k < BlockSize; k++)
          for (int l = 0; l < BlockSize; l++)
            addElement(block_index_array[i] * BlockSize + k,
                       block_index_array[j] * BlockSize + l,
                       matrix(i * BlockSize + k, j * BlockSize + l));
  }

  [[nodiscard]] int rows() const {
    return m_rows;
  }
  [[nodiscard]] int cols() const {
    return m_cols;
  }

  SparseMatrixBuilder& clear() {
    m_rows = 0;
    m_cols = 0;
    m_triplets.clear();
    return *this;
  }

  SparseMatrixBuilder& setRows(size_t rows) {
    m_rows = rows;
    return *this;
  }

  SparseMatrixBuilder& setColumns(size_t cols) {
    m_cols = cols;
    return *this;
  }

  SparseMatrixBuilder& reserveNonZeros(size_t nnz) {
    m_triplets.reserve(nnz);
    return *this;
  }

  SparseMatrix build() const {
    if (!m_rows || !m_cols)
      throw std::runtime_error("matrix size not set properly");
    SparseMatrix matrix(m_rows, m_cols);
    matrix.setFromTriplets(m_triplets.begin(), m_triplets.end());
    return matrix;
  }
 private:
  size_t m_rows{}, m_cols{};
  std::vector<Triplet> m_triplets{};
};
}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_SPARSE_MATRIX_BUILDER_H_
