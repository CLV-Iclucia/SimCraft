#pragma once
#include <Maths/block-linear-solver.h>
#include <Maths/block-solvers/block-jacobi.h>
#include <spdlog/spdlog.h>

namespace sim::maths {

struct BlockPCGSolver final : BlockLinearSolver {
  int maxIterations = 1000;
  Real tolerance = 1e-6;
  std::unique_ptr<BlockPreconditioner> preconditioner;

  BlockPCGSolver() {
    preconditioner = std::make_unique<BlockJacobiPreconditioner>();
  }

  explicit BlockPCGSolver(int maxIter, Real tol)
      : maxIterations(maxIter), tolerance(tol) {
    preconditioner = std::make_unique<BlockJacobiPreconditioner>();
  }

  SolveResult solve(const BlockSparseMatrix<3>& A,
                    const BlockVector<3>& b,
                    BlockVector<3>& x) override {
    const int n = b.numBlocks();
    BlockVector<3> r(n), z(n), p(n), Ap(n);

    // r = b - A*x  (if x initially zero then r = b)
    A.apply(x, r);         // r = A*x
    r *= Real(-1);
    r += b;                // r = b - A*x

    preconditioner->setup(A);
    preconditioner->apply(r, z);   // z = M^{-1} r
    p.copyFrom(z);                 // p = z

    Real rz = r.dot(z);
    Real b_norm = b.norm();
    if (b_norm < 1e-30) return {true, 0, 0.0};  // zero RHS

    for (int iter = 0; iter < maxIterations; iter++) {
      A.apply(p, Ap);              // Ap = A*p
      Real pAp = p.dot(Ap);

      if (std::abs(pAp) < 1e-30) {
        spdlog::warn("BlockPCG: pAp near zero at iter {}", iter);
        return {false, iter, r.norm()};
      }

      Real alpha = rz / pAp;

      x.axpy(alpha, p);           // x += alpha * p
      r.axpy(-alpha, Ap);         // r -= alpha * Ap

      Real r_norm = r.norm();
      if (r_norm / b_norm < tolerance)
        return {true, iter + 1, r_norm};

      preconditioner->apply(r, z); // z = M^{-1} r
      Real rz_new = r.dot(z);
      Real beta = rz_new / rz;
      rz = rz_new;

      // p = z + beta * p
      p *= beta;
      p += z;
    }
    return {false, maxIterations, r.norm()};
  }
};

} // namespace sim::maths
