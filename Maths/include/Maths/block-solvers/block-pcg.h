#pragma once
#include <Maths/block-linear-solver.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace sim::maths {

/// Block PCG solver with built-in Block Jacobi preconditioning.
///
/// Default behavior: uses inline Block Jacobi (no virtual dispatch).
/// If a custom preconditioner is assigned, falls back to virtual dispatch.
struct BlockPCGSolver final : BlockLinearSolver {
  int maxIterations = 10000;
  Real tolerance = 1e-6;

  /// Optional custom preconditioner. If null, uses built-in Block Jacobi.
  std::unique_ptr<BlockPreconditioner> customPreconditioner = nullptr;

  BlockPCGSolver() = default;

  explicit BlockPCGSolver(int maxIter, Real tol)
      : maxIterations(maxIter), tolerance(tol) {}

  SolveResult solve(const BlockSparseMatrix<3>& A,
                    const BlockVector<3>& b,
                    BlockVector<3>& x) override {
    const int n = b.numBlocks();
    BlockVector<3> r(n), z(n), p(n), Ap(n);

    // Setup preconditioner
    if (customPreconditioner) {
      customPreconditioner->setup(A);
    } else {
      setupJacobi(A);
    }

    // r = b - A*x
    A.apply(x, r);
    r *= Real(-1);
    r += b;

    // z = M^{-1} r
    applyPrecond(r, z);
    p.copyFrom(z);

    Real rz = r.dot(z);
    Real b_norm = b.norm();
    if (b_norm < 1e-30) return {true, 0, 0.0};

    for (int iter = 0; iter < maxIterations; iter++) {
      A.apply(p, Ap);
      Real pAp = p.dot(Ap);

      if (std::abs(pAp) < 1e-30) {
        spdlog::warn("BlockPCG: pAp near zero at iter {}", iter);
        return {false, iter, r.norm()};
      }

      Real alpha = rz / pAp;

      x.axpy(alpha, p);
      r.axpy(-alpha, Ap);

      Real r_norm = r.norm();
      if (r_norm / b_norm < tolerance)
        return {true, iter + 1, r_norm};

      applyPrecond(r, z);
      Real rz_new = r.dot(z);
      Real beta = rz_new / rz;
      rz = rz_new;

      p *= beta;
      p += z;
    }
    return {false, maxIterations, r.norm()};
  }

private:
  // --- Built-in Block Jacobi (no virtual dispatch) ---
  std::vector<glm::dmat3> m_invDiag;

  void setupJacobi(const BlockSparseMatrix<3>& A) {
    auto diag = A.extractDiagonal();
    const int n = static_cast<int>(diag.size());
    m_invDiag.resize(n);

    if (n < 1000) {
      for (int i = 0; i < n; i++) {
        if (std::abs(glm::determinant(diag[i])) < 1e-12)
          diag[i] += glm::dmat3(1e-10);
        m_invDiag[i] = glm::inverse(diag[i]);
      }
    } else {
      tbb::parallel_for(tbb::blocked_range<int>(0, n),
          [&](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); i++) {
              if (std::abs(glm::determinant(diag[i])) < 1e-12)
                diag[i] += glm::dmat3(1e-10);
              m_invDiag[i] = glm::inverse(diag[i]);
            }
          });
    }
  }

  void applyPrecond(const BlockVector<3>& r, BlockVector<3>& z) const {
    if (customPreconditioner) {
      customPreconditioner->apply(r, z);
      return;
    }
    // Inline Block Jacobi: z[i] = invDiag[i] * r[i]
    const int n = r.numBlocks();
    if (n < 1000) {
      for (int i = 0; i < n; i++)
        z[i] = m_invDiag[i] * r[i];
    } else {
      tbb::parallel_for(tbb::blocked_range<int>(0, n),
          [&](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); i++)
              z[i] = m_invDiag[i] * r[i];
          });
    }
  }
};

} // namespace sim::maths
