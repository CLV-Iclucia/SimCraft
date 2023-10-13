#include <PoissonSolver/multigrid-solver.h>
#include <PoissonSolver/poisson-solver.h>
#include <PoissonSolver/cg-solver.h>
#include <PoissonSolver/util.h>
#include <cstddef>

namespace poisson {

void relax(float* u, float* f, int n, int m, const MultigridOption& option) {
  for (int i = 0; i < option.smooth_iter; i++) {
    rbGaussSeidel(u, f, n, m, 0);
    rbGaussSeidel(u, f, n, m, 1);
  }
}

// e = f - Au
// we directly override f
void calcResidual(float* u, float* f, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++)
      f[i * m + j] -= 4 * u[i * m + j] - (i > 0 ? u[(i - 1) * m + j] : 0) -
                      (i < n - 1 ? u[(i + 1) * m + j] : 0) -
                      (j > 0 ? u[i * m + j - 1] : 0) -
                      (j < m - 1 ? u[i * m + j + 1] : 0);
  }
}

// nf = I(2h, h)f
// we use the restriction operator from "A Parallel Multigrid Poisson Solver for
// Fluids Simulation on Large Grids" by McAdams et al. the stencil is: 1/64 3/64
// 3/64 1/64 3/64 9/64 9/64 3/64 3/64 9/64 9/64 3/64 1/64 3/64 3/64 1/64 the n
// and m is the size of the finer grid
void restrict(float* f, float* nf, int n, int m) {
  int nn = n >> 1, nm = m >> 1;
  for (int i = 0; i < nn; i++) {
    for (int j = 0; j < nm; j++) {
#define F(i, j) f[(i)*m + (j)]
#define NF(i, j) nf[(i)*nm + (j)]
      NF(i, j) = 0;
      if (i > 0) {
        if (j > 0) NF(i, j) += F(2 * i - 1, 2 * j - 1) * 3 / 64;
        NF(i, j) += F(2 * i - 1, 2 * j) * 9 / 64;
        if (j < m - 1) NF(i, j) += F(2 * i - 1, 2 * j + 1) * 3 / 64;
      }
      if (i < nn - 1) {
        if (j > 0) NF(i, j) += F(2 * i + 1, 2 * j - 1) * 3 / 64;
        NF(i, j) += F(2 * i + 1, 2 * j) * 9 / 64;
        if (j < nm - 1) NF(i, j) += F(2 * i + 1, 2 * j + 1) * 3 / 64;
      }
      if (j > 0) NF(i, j) += F(2 * i, 2 * j - 1) * 9 / 64;
      NF(i, j) += F(2 * i, 2 * j) * 36 / 64;
      if (j < nm - 1) NF(i, j) += F(2 * i, 2 * j + 1) * 9 / 64;
#undef NF
#undef F
    }
  }
}

// solve Au = f on the coarsest grid
// n and m is the size of the coarsest grid
// still use rbGauss
void bottomSolve(float* u, float* f, int n, int m,
                 const MultigridOption& option) {
  for (int i = 0; i < option.bottom_iter; i++) {
    rbGaussSeidel(u, f, n, m, 0);
    rbGaussSeidel(u, f, n, m, 1);
  }
}

// nu += I(h, 2h)u
// n, m is the size of the coarser grid
// the stencil is double the restriction operator
// the stencil is:
// 1/32 3/32 3/32 1/32
// 3/32 9/32 9/32 3/32
// 3/32 9/32 9/32 3/32
// 1/32 3/32 3/32 1/32
void correct(float* u, float* nu, int n, int m) {
  int nn = n << 1, nm = m << 1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
#define U(i, j) u[(i)*m + (j)]
#define NU(i, j) nu[(i)*nm + (j)]
      NU(2 * i, 2 * j) += U(i, j) * 9 / 32;
      if (i > 0) {
        if (j > 0) NU(2 * i - 1, 2 * j - 1) += U(i, j) * 3 / 32;
        NU(2 * i - 1, 2 * j) += U(i, j) * 9 / 32;
        if (j < m - 1) NU(2 * i - 1, 2 * j + 1) += U(i, j) * 3 / 32;
      }
      if (i < n - 1) {
        if (j > 0) NU(2 * i + 1, 2 * j - 1) += U(i, j) * 3 / 32;
        NU(2 * i + 1, 2 * j) += U(i, j) * 9 / 32;
        if (j < m - 1) NU(2 * i + 1, 2 * j + 1) += U(i, j) * 3 / 32;
      }
      if (j > 0) NU(2 * i, 2 * j - 1) += U(i, j) * 9 / 32;
      if (j < m - 1) NU(2 * i, 2 * j + 1) += U(i, j) * 9 / 32;
#undef NU
#undef U
    }
  }
}

void multigrid(float* u, float* f, int n, int m, const MultigridOption& mg_option) {
  int max_level = mg_option.max_level;
  // take the pointers
  float* aux_vars = mg_option.aux_vars;
  float* aux_rhs = mg_option.aux_rhs;
  // calc the residual
  for (int level = 1; level < max_level; level++) {
    zeroFill(u, n, m);
    relax(u, f, n, m, mg_option);
    calcResidual(u, f, n, m);
    restrict(f, f + n * m, n, m);
    u += n * m;
    f += n * m;
    n >>= 1;
    m >>= 1;
  }
  // solve on the coarsest grid
  bottomSolve(u, f, n, m, mg_option);
  for (int level = 0; level < max_level - 1; level++) {
    correct(u, u + n * m, n, m);
    relax(u, f, n, m, mg_option);
    n <<= 1;
    m <<= 1;
    u -= n * m;
    f -= n * m;
  }
}

// this almost follows the V-Cycle scheme of "A Multigrid Tutorial" by Briggs et
// al.
void multigridSolve(const PoissonSolverOption& option,
                    const MultigridOption& mg_option) {
  // take the arguments from option and call multigrid
  int n = option.width, m = option.height;
  float* u = option.input_vars;
  float* f = option.input_rhs;
  multigrid(u, f, n, m, mg_option);
}

float dot(float* u, float* v, int n, int m) {
  float res = 0;
  for (int i = 0; i < n * m; i++) 
    res += u[i] * v[i];
  return res;
}

float precondAndDot(float* p, float* r, float mu, int n, int m,
             const PoissonSolverOption& option,
             const MultigridOption& mg_option) {
  for (int i = 0; i < n * m; i++) r[i] -= mu;
  multigrid(p, r, n, m, mg_option);
  return dot(p, r, n, m);
}

float computeNullSpace(float* r, int n, int m) {
  float mu = 0;
  for (int i = 0; i < n * m; i++) mu += r[i];
  mu /= n * m;
  return mu;
}

// implmented based on "A Parallel Multigrid Poisson Solver for Fluids
// Simulation on Large Grids" by McAdams et al.
void mgpcgSolve(const PoissonSolverOption& option,
                const MultigridOption& mg_option,
                const CgSolverOption& cg_option) {
  float* x = option.input_vars;
  float* f = option.input_rhs;
  float* r = option.residual;
  int n = option.width, m = option.height;
  float mu = computeNullSpace(r, n, m);
  float* p = cg_option.aux_var_step;
  float* Ap = cg_option.aux_var_Ap;
  float rho = precondAndDot(p, r, mu, n, m, option, mg_option);
  // start iteration
  for (int i = 0; i < cg_option.max_iter; i++) {
    applyLaplacian(Ap, p, n, m);
    float alpha = rho / dot(p, Ap, n, m);
    
  }
}

}  // namespace poisson