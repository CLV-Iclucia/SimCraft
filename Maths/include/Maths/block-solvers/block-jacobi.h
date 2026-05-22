#pragma once
#include <Maths/block-linear-solver.h>
#include <glm/glm.hpp>

namespace sim::maths {

struct BlockJacobiPreconditioner final : BlockPreconditioner {
  void setup(const BlockSparseMatrix<3>& A) override {
    auto diag = A.extractDiagonal();
    m_invDiag.resize(diag.size());
    for (size_t i = 0; i < diag.size(); i++) {
      Real det = glm::determinant(diag[i]);
      if (std::abs(det) < 1e-12) {
        // Regularization: add small perturbation to singular diagonal blocks
        diag[i] += glm::dmat3(1e-10);
      }
      m_invDiag[i] = glm::inverse(diag[i]);
    }
  }

  void apply(const BlockVector<3>& r, BlockVector<3>& z) const override {
    assert(r.numBlocks() == static_cast<int>(m_invDiag.size()));
    for (int i = 0; i < r.numBlocks(); i++)
      z[i] = m_invDiag[i] * r[i];
  }

private:
  std::vector<glm::dmat3> m_invDiag;
};

} // namespace sim::maths
