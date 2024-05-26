//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_SPARSE_MATRIX_BUILDER_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_SPARSE_MATRIX_BUILDER_H_
#include <Maths/types.h>
#include <iostream>
#include <vector>
namespace maths {
template<typename Scalar>
class SparseMatrixBuilder {
 public:
  using Triplet = Eigen::Triplet<Scalar>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  SparseMatrixBuilder(int rows, int cols, int nnz) : m_rows(rows), m_cols(cols), m_triplets(nnz) {}
  void addElement(int row, int col, Scalar value) {
    if (row >= m_rows || col >= m_cols) {
      std::cerr << "invalid row or col" << std::endl;
      exit(1);
    }
    m_triplets.emplace_back(row, col, value);
  }
  [[nodiscard]] int rows() const {
    return m_rows;
  }
  [[nodiscard]] int cols() const {
    return m_cols;
  }
  void reset() {
    m_triplets.clear();
  }
  void reset(int rows, int cols) {
    m_rows = rows;
    m_cols = cols;
  }
  void reserveNonZeros(int nnz) {
    m_triplets.reserve(nnz);
  }
  SparseMatrix build() const {
    SparseMatrix matrix(m_rows, m_cols);
    matrix.setFromTriplets(m_triplets.begin(), m_triplets.end());
    return matrix;
  }
 private:
  int m_rows{}, m_cols{};
  std::vector<Triplet> m_triplets{};
};
}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_SPARSE_MATRIX_BUILDER_H_
