#include <gtest/gtest.h>
#include <Maths/block-sparse-matrix.h>
#include <Maths/block-types.h>
#include <Maths/block-solvers/block-pcg.h>

using namespace sim::maths;

// Helper: build a known SPD block matrix in full (non-symmetric) mode
static BlockSparseMatrix<3> buildFullSPD(int nBlocks) {
  BlockSparseMatrix<3> A(nBlocks, nBlocks);
  glm::dmat3 I(1.0);

  // Diagonal-dominant: A[i][i] = (nBlocks+1)*I, A[i][j] = random symmetric block for |i-j|==1
  for (int i = 0; i < nBlocks; i++) {
    A.addBlock(i, i, I * static_cast<Real>(nBlocks + 1));
  }
  // Off-diagonal: symmetric blocks (B and B^T)
  for (int i = 0; i + 1 < nBlocks; i++) {
    glm::dmat3 B(0.0);
    // Deterministic "random" symmetric block
    B[0][0] = 0.5 * (i + 1);  B[0][1] = 0.1 * i;        B[0][2] = 0.05;
    B[1][0] = 0.1 * i;        B[1][1] = 0.3 * (i + 1);  B[1][2] = 0.02 * i;
    B[2][0] = 0.05;           B[2][1] = 0.02 * i;        B[2][2] = 0.2 * (i + 1);
    A.addBlock(i, i + 1, B);
    A.addBlock(i + 1, i, glm::transpose(B));
  }
  return A;
}

// Same matrix, built in symmetric mode (only upper triangle stored)
static BlockSparseMatrix<3> buildSymmetricSPD(int nBlocks) {
  BlockSparseMatrix<3> A(nBlocks, nBlocks);
  A.setSymmetric(true);
  glm::dmat3 I(1.0);

  for (int i = 0; i < nBlocks; i++) {
    A.addBlock(i, i, I * static_cast<Real>(nBlocks + 1));
  }
  for (int i = 0; i + 1 < nBlocks; i++) {
    glm::dmat3 B(0.0);
    B[0][0] = 0.5 * (i + 1);  B[0][1] = 0.1 * i;        B[0][2] = 0.05;
    B[1][0] = 0.1 * i;        B[1][1] = 0.3 * (i + 1);  B[1][2] = 0.02 * i;
    B[2][0] = 0.05;           B[2][1] = 0.02 * i;        B[2][2] = 0.2 * (i + 1);
    // Only add the upper-triangle entry (i < i+1)
    A.addBlock(i, i + 1, B);
  }
  return A;
}

// ============================================================
// Test 1: Symmetric SpMV matches full SpMV
// ============================================================
TEST(SymmetricMode, SpMVMatchesFull) {
  constexpr int N = 5;
  auto fullA = buildFullSPD(N);
  auto symA = buildSymmetricSPD(N);

  // Storage savings: symmetric should have fewer entries
  // Full: N diagonal + 2*(N-1) off-diag = 3N - 2
  // Sym:  N diagonal + (N-1) off-diag = 2N - 1
  EXPECT_EQ(fullA.numEntries(), 3 * N - 2);
  EXPECT_EQ(symA.numEntries(), 2 * N - 1);

  // Build a test vector
  BlockVector<3> x(N);
  for (int i = 0; i < N; i++)
    x[i] = glm::dvec3(1.0 + i, 2.0 - 0.5 * i, 0.3 * i * i);

  // Apply both
  BlockVector<3> y_full(N), y_sym(N);
  fullA.apply(x, y_full);
  symA.apply(x, y_sym);

  // Results should match
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(y_full[i].x, y_sym[i].x, 1e-12)
        << "Mismatch at block " << i << " component x";
    EXPECT_NEAR(y_full[i].y, y_sym[i].y, 1e-12)
        << "Mismatch at block " << i << " component y";
    EXPECT_NEAR(y_full[i].z, y_sym[i].z, 1e-12)
        << "Mismatch at block " << i << " component z";
  }
}

// ============================================================
// Test 2: addBlock canonicalization
// Even if we add the lower-triangle entry, symmetric mode
// canonicalizes to upper triangle and apply() still works.
// ============================================================
TEST(SymmetricMode, CanonicalizeLowerTriangle) {
  constexpr int N = 3;
  auto symA = buildSymmetricSPD(N);

  // Build another symmetric matrix, but add entries in reverse order
  BlockSparseMatrix<3> symB(N, N);
  symB.setSymmetric(true);
  glm::dmat3 I(1.0);

  for (int i = 0; i < N; i++)
    symB.addBlock(i, i, I * static_cast<Real>(N + 1));

  for (int i = 0; i + 1 < N; i++) {
    glm::dmat3 B(0.0);
    B[0][0] = 0.5 * (i + 1);  B[0][1] = 0.1 * i;        B[0][2] = 0.05;
    B[1][0] = 0.1 * i;        B[1][1] = 0.3 * (i + 1);  B[1][2] = 0.02 * i;
    B[2][0] = 0.05;           B[2][1] = 0.02 * i;        B[2][2] = 0.2 * (i + 1);
    // Intentionally add as lower triangle (i+1, i) with transposed block
    // Symmetric mode should canonicalize this to (i, i+1) with the original block
    symB.addBlock(i + 1, i, glm::transpose(B));
  }

  // Both should have same number of entries
  EXPECT_EQ(symA.numEntries(), symB.numEntries());

  // Apply both to same vector
  BlockVector<3> x(N);
  x[0] = glm::dvec3(1.0, 2.0, 3.0);
  x[1] = glm::dvec3(4.0, 5.0, 6.0);
  x[2] = glm::dvec3(7.0, 8.0, 9.0);

  BlockVector<3> yA(N), yB(N);
  symA.apply(x, yA);
  symB.apply(x, yB);

  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(yA[i].x, yB[i].x, 1e-12);
    EXPECT_NEAR(yA[i].y, yB[i].y, 1e-12);
    EXPECT_NEAR(yA[i].z, yB[i].z, 1e-12);
  }
}

// ============================================================
// Test 3: PCG solver works with symmetric matrix
// ============================================================
TEST(SymmetricMode, PCGSolveMatchesFull) {
  constexpr int N = 4;
  auto fullA = buildFullSPD(N);
  auto symA = buildSymmetricSPD(N);

  // Known solution
  BlockVector<3> x_true(N);
  x_true[0] = glm::dvec3(1.0, 2.0, 3.0);
  x_true[1] = glm::dvec3(4.0, 5.0, 6.0);
  x_true[2] = glm::dvec3(7.0, 8.0, 9.0);
  x_true[3] = glm::dvec3(10.0, 11.0, 12.0);

  // b = A * x_true
  BlockVector<3> b_full(N), b_sym(N);
  fullA.apply(x_true, b_full);
  symA.apply(x_true, b_sym);

  // b should match
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(b_full[i].x, b_sym[i].x, 1e-12);
    EXPECT_NEAR(b_full[i].y, b_sym[i].y, 1e-12);
    EXPECT_NEAR(b_full[i].z, b_sym[i].z, 1e-12);
  }

  // Solve with full matrix
  BlockVector<3> x_full(N);
  x_full.setZero();
  BlockPCGSolver solver(1000, 1e-10);
  auto result_full = solver.solve(fullA, b_full, x_full);
  EXPECT_TRUE(result_full.converged);

  // Solve with symmetric matrix
  BlockVector<3> x_sym(N);
  x_sym.setZero();
  auto result_sym = solver.solve(symA, b_sym, x_sym);
  EXPECT_TRUE(result_sym.converged);

  // Solutions should match
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(x_full[i].x, x_sym[i].x, 1e-10);
    EXPECT_NEAR(x_full[i].y, x_sym[i].y, 1e-10);
    EXPECT_NEAR(x_full[i].z, x_sym[i].z, 1e-10);
  }

  // Both should recover x_true
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(x_sym[i].x, x_true[i].x, 1e-9);
    EXPECT_NEAR(x_sym[i].y, x_true[i].y, 1e-9);
    EXPECT_NEAR(x_sym[i].z, x_true[i].z, 1e-9);
  }
}

// ============================================================
// Test 4: assembleBlock<4> respects symmetric mode
// ============================================================
TEST(SymmetricMode, AssembleBlockSkipsLowerTriangle) {
  // A 4×4 block local Hessian (e.g., from a tetrahedron)
  Eigen::Matrix<Real, 12, 12> localMat;
  localMat.setZero();
  // Fill with a symmetric pattern
  for (int i = 0; i < 12; i++) {
    localMat(i, i) = 10.0 + i;
    for (int j = i + 1; j < 12; j++) {
      Real val = 0.1 * (i + 1) * (j + 1);
      localMat(i, j) = val;
      localMat(j, i) = val;
    }
  }

  std::array<int, 4> blockIndices = {0, 2, 5, 7};  // non-contiguous

  // Full mode: all 16 blocks stored
  BlockSparseMatrix<3> fullH(8, 8);
  fullH.assembleBlock<4>(localMat, blockIndices);

  // Symmetric mode: only upper triangle (~10 blocks)
  BlockSparseMatrix<3> symH(8, 8);
  symH.setSymmetric(true);
  symH.assembleBlock<4>(localMat, blockIndices);

  // Symmetric should have fewer entries (upper triangle of 4x4 = 10 blocks)
  EXPECT_EQ(fullH.numEntries(), 16);
  EXPECT_EQ(symH.numEntries(), 10);  // 4 diagonal + 6 upper off-diagonal

  // But SpMV should produce the same result
  BlockVector<3> x(8);
  for (int i = 0; i < 8; i++)
    x[i] = glm::dvec3(0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1));

  BlockVector<3> y_full(8), y_sym(8);
  fullH.apply(x, y_full);
  symH.apply(x, y_sym);

  for (int i = 0; i < 8; i++) {
    EXPECT_NEAR(y_full[i].x, y_sym[i].x, 1e-10)
        << "Mismatch at block " << i << ".x";
    EXPECT_NEAR(y_full[i].y, y_sym[i].y, 1e-10)
        << "Mismatch at block " << i << ".y";
    EXPECT_NEAR(y_full[i].z, y_sym[i].z, 1e-10)
        << "Mismatch at block " << i << ".z";
  }
}

// ============================================================
// Test 5: assembleLocalHessian respects symmetric mode
// ============================================================
TEST(SymmetricMode, AssembleLocalHessianSymmetric) {
  constexpr int N = 3;

  // Symmetric local hessian
  LocalHessian<N> hess{};
  for (int i = 0; i < N; i++) {
    hess[i][i] = glm::dmat3(2.0);  // diagonal
    for (int j = i + 1; j < N; j++) {
      glm::dmat3 B(0.0);
      B[0][0] = 0.3 * (i + j);
      B[1][1] = 0.2 * (i + j);
      B[2][2] = 0.1 * (i + j);
      hess[i][j] = B;
      hess[j][i] = glm::transpose(B);
    }
  }

  // Symmetric local gradient
  LocalGrad<N> grad{};
  grad[0] = glm::dvec3(1.0, 0.5, 0.2);
  grad[1] = glm::dvec3(0.3, 1.0, 0.4);
  grad[2] = glm::dvec3(0.1, 0.2, 1.0);

  std::array<int, N> globalIdx = {1, 3, 5};
  Real bGrad = 0.5, bHess = 0.1, kappa = 100.0;

  // Full mode
  BlockSparseMatrix<3> fullH(6, 6);
  assembleLocalHessian<N>(fullH, globalIdx, hess, grad, bGrad, bHess, kappa);

  // Symmetric mode
  BlockSparseMatrix<3> symH(6, 6);
  symH.setSymmetric(true);
  assembleLocalHessian<N>(symH, globalIdx, hess, grad, bGrad, bHess, kappa);

  // Symmetric should store fewer entries
  EXPECT_EQ(fullH.numEntries(), N * N);       // 9
  EXPECT_EQ(symH.numEntries(), N * (N + 1) / 2);  // 6

  // SpMV should match
  BlockVector<3> x(6);
  for (int i = 0; i < 6; i++)
    x[i] = glm::dvec3(1.0 + 0.5 * i, 2.0 - 0.3 * i, 0.1 * i * i);

  BlockVector<3> y_full(6), y_sym(6);
  fullH.apply(x, y_full);
  symH.apply(x, y_sym);

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(y_full[i].x, y_sym[i].x, 1e-10)
        << "Mismatch at block " << i << ".x";
    EXPECT_NEAR(y_full[i].y, y_sym[i].y, 1e-10)
        << "Mismatch at block " << i << ".y";
    EXPECT_NEAR(y_full[i].z, y_sym[i].z, 1e-10)
        << "Mismatch at block " << i << ".z";
  }
}

// ============================================================
// Test 6: addFrom with mixed symmetric modes
// (non-symmetric mass added to symmetric Hessian)
// ============================================================
TEST(SymmetricMode, AddFromNonSymmetricToSymmetric) {
  constexpr int N = 3;

  // Symmetric Hessian
  BlockSparseMatrix<3> symH(N, N);
  symH.setSymmetric(true);
  glm::dmat3 I(1.0);
  symH.addBlock(0, 0, I * 5.0);
  symH.addBlock(1, 1, I * 5.0);
  symH.addBlock(2, 2, I * 5.0);
  symH.addBlock(0, 1, I * 0.5);
  symH.addBlock(1, 2, I * 0.3);

  // Non-symmetric diagonal mass matrix (like blockMass)
  BlockSparseMatrix<3> mass(N, N);
  mass.addBlock(0, 0, I * 2.0);
  mass.addBlock(1, 1, I * 3.0);
  mass.addBlock(2, 2, I * 4.0);

  // addFrom should work: diagonal blocks just add normally
  symH.addFrom(mass);

  // Reference: build the same thing as full matrix
  BlockSparseMatrix<3> fullH(N, N);
  fullH.addBlock(0, 0, I * 5.0);
  fullH.addBlock(1, 1, I * 5.0);
  fullH.addBlock(2, 2, I * 5.0);
  fullH.addBlock(0, 1, I * 0.5);
  fullH.addBlock(1, 0, I * 0.5);  // symmetric counterpart
  fullH.addBlock(1, 2, I * 0.3);
  fullH.addBlock(2, 1, I * 0.3);  // symmetric counterpart
  fullH.addFrom(mass);

  // Verify SpMV
  BlockVector<3> x(N);
  x[0] = glm::dvec3(1.0, 2.0, 3.0);
  x[1] = glm::dvec3(4.0, 5.0, 6.0);
  x[2] = glm::dvec3(7.0, 8.0, 9.0);

  BlockVector<3> y_full(N), y_sym(N);
  fullH.apply(x, y_full);
  symH.apply(x, y_sym);

  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(y_full[i].x, y_sym[i].x, 1e-12);
    EXPECT_NEAR(y_full[i].y, y_sym[i].y, 1e-12);
    EXPECT_NEAR(y_full[i].z, y_sym[i].z, 1e-12);
  }
}
