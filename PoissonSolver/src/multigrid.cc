#include <PoissonSolver/multigrid-solver.h>
#include <PoissonSolver/poisson-solver.h>
#include <PoissonSolver/cg-solver.h>
#include <PoissonSolver/util.h>
#include <cstdio>
#include <algorithm>
#include <cstring>

namespace poisson {

void relax(double* u, double* f, int n, int m, const MultigridOption& option, int dir) {
  if (dir == 0) {
    for (int iter = 0; iter < option.smooth_iter; iter++) {
      rbGaussSeidel(u, f, n, m, 0);
      rbGaussSeidel(u, f, n, m, 1);
    }
  } else {
    for (int iter = 0; iter < option.smooth_iter; iter++) {
      rbGaussSeidelReverse(u, f, n, m, 1);
      rbGaussSeidelReverse(u, f, n, m, 0);
    }
  }
}

// nf = I(2h, h)f
// we use the restriction operator from "A Parallel Multigrid Poisson Solver for
// Fluids Simulation on Large Grids" by McAdams et al. the stencil is: 1/64 3/64
// 3/64 1/64 3/64 9/64 9/64 3/64 3/64 9/64 9/64 3/64 1/64 3/64 3/64 1/64 the n
// and m is the size of the finer grid
void restrict(double* __restrict__ f, double* __restrict__ nf, int n, int m) {
  int nn = n >> 1, nm = m >> 1;
  for (int i = 0; i < nn; i++) {
    for (int j = 0; j < nm; j++) {
#define F(i, j) f[(i)*m + (j)]
#define NF(i, j) nf[(i)*nm + (j)]
      NF(i, j) = F(2 * i, 2 * j) * 9.f / 64;
      NF(i, j) += F(2 * i, 2 * j + 1) * 9.f / 64;
      NF(i, j) += F(2 * i + 1, 2 * j) * 9.f / 64;
      NF(i, j) += F(2 * i + 1, 2 * j + 1) * 9.f / 64;
      if (i > 0) {
        if (j > 0) NF(i, j) += F(2 * i - 1, 2 * j - 1) * 1.f / 64;
        NF(i, j) += F(2 * i - 1, 2 * j) * 3.f / 64;
        NF(i, j) += F(2 * i - 1, 2 * j + 1) * 3.f / 64;
        if (j < nm - 1) NF(i, j) += F(2 * i - 1, 2 * j + 2) * 1.f / 64;
      }
      if (j > 0) {
        NF(i, j) += F(2 * i, 2 * j - 1) * 3.f / 64;
        NF(i, j) += F(2 * i + 1, 2 * j - 1) * 3.f / 64;
      }
      if (i < nn - 1) {
        if (j > 0) NF(i, j) += F(2 * i + 2, 2 * j - 1) * 1.f / 64;
        NF(i, j) += F(2 * i + 2, 2 * j) * 3.f / 64;
        NF(i, j) += F(2 * i + 2, 2 * j + 1) * 3.f / 64;
        if (j < nm - 1) NF(i, j) += F(2 * i + 2, 2 * j + 2) * 1.f / 64;
      }
      if (j < nm - 1) {
        NF(i, j) += F(2 * i, 2 * j + 2) * 3.f / 64;
        NF(i, j) += F(2 * i + 1, 2 * j + 2) * 3.f / 64;
      }
#undef NF
#undef F
    }
  }
}

// solve Au = f on the coarsest grid
// n and m is the size of the coarsest grid
// still use rbGauss
void bottomSolve(double* __restrict__ u, double* __restrict__ f, int n, int m,
                 const MultigridOption& option) {
  for (int i = 0; i < option.bottom_iter; i++) {
    rbGaussSeidel(u, f, n, m, 0);
    rbGaussSeidel(u, f, n, m, 1);
  }
}

// nu += I(h, 2h)u
// n, m is the size of the coarser grid
void correct(double* __restrict__ u, double* __restrict__ nu, int n, int m) {
  int nm = m << 1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
#define U(i, j) u[(i)*m + (j)]
#define NU(i, j) nu[(i)*nm + (j)]
      NU(2 * i, 2 * j) += U(i, j) * 9.f / 16;
      NU(2 * i, 2 * j + 1) += U(i, j) * 9.f / 16;
      NU(2 * i + 1, 2 * j) += U(i, j) * 9.f / 16;
      NU(2 * i + 1, 2 * j + 1) += U(i, j) * 9.f / 16;
      if (i > 0) {
        if (j > 0) NU(2 * i - 1, 2 * j - 1) += U(i, j) * 1.f / 16;
        NU(2 * i - 1, 2 * j) += U(i, j) * 3.f / 16;
        NU(2 * i - 1, 2 * j + 1) += U(i, j) * 3.f / 16;
        if (j < m - 1) NU(2 * i - 1, 2 * j + 2) += U(i, j) * 1.f / 16;
      }
      if (j > 0) {
        NU(2 * i, 2 * j - 1) += U(i, j) * 3.f / 16;
        NU(2 * i + 1, 2 * j - 1) += U(i, j) * 3.f / 16;
      }
      if (i < n - 1) {
        if (j > 0) NU(2 * i + 2, 2 * j - 1) += U(i, j) * 1.f / 16;
        NU(2 * i + 2, 2 * j) += U(i, j) * 3.f / 16;
        NU(2 * i + 2, 2 * j + 1) += U(i, j) * 3.f / 16;
        if (j < m - 1) NU(2 * i + 2, 2 * j + 2) += U(i, j) * 1.f / 16;
      }
      if (j < m - 1) {
        NU(2 * i, 2 * j + 2) += U(i, j) * 3.f / 16;
        NU(2 * i + 1, 2 * j + 2) += U(i, j) * 3.f / 16;
      }
#undef NU
#undef U
    }
  }
}

void multigrid(double* u, double* f, int n, int m, const MultigridOption& mg_option) {
  int max_level = mg_option.max_level;
  double *input_vars = u, *input_rhs = f;
  double* aux_vars = mg_option.aux_vars;
  double* aux_rhs = mg_option.aux_rhs;
  double* r = mg_option.residual;
  for (int level = 0; level < max_level; level++) {
    zeroFill(u, n, m);
    relax(u, f, n, m, mg_option, 0);
    computeResidual(u, f, level == 0 ? r : f, n, m);
    printf("%f\n", normLinf(level == 0 ? r : f, n, m));
    double* nf = level == 0 ? aux_rhs : f + n * m;
    double* nu = level == 0 ? aux_vars : u + n * m;
    restrict(level == 0 ? r : f, nf, n, m);
    f = nf;
    u = nu;
    n >>= 1;
    m >>= 1;
  }
  bottomSolve(u, f, n, m, mg_option);
  for (int level = max_level - 1; level >= 0; level--) {
    double* nu = level == 0 ? input_vars : u - (n << 1) * (m << 1);
    double* nf = level == 0 ? input_rhs : f - (n << 1) * (m << 1);
    correct(u, nu, n, m);
    n <<= 1;
    m <<= 1;
    u = nu;
    f = nf;
    relax(u, f, n, m, mg_option, 1);
  }
  computeResidual(u, input_rhs, r, n, m);
  std::printf("multigrid: residual = %f\n", normLinf(r, n, m));
}

// this almost follows the V-Cycle scheme of "A Multigrid Tutorial" by Briggs et
// al.
void multigridSolve(const PoissonSolverOption& option,
                    const MultigridOption& mg_option) {
  // take the arguments from option and call multigrid
  int n = option.width, m = option.height;
  double* u = option.input_vars;
  double* f = option.input_rhs;
  multigrid(u, f, n, m, mg_option);
}

double precondAndDot(double* p, double* r, double mu, int n, int m,
             const MultigridOption& mg_option) {
  for (int i = 0; i < n * m; i++)
    r[i] -= mu;
  multigrid(p, r, n, m, mg_option);
  return dot(p, r, n, m);
}

static double normSubAve(double* v, double mu, int n, int m) {
  double res = 0.0;
  for (int i = 0; i < n * m; i++)
    res = std::max(std::abs(v[i] - mu), res);
  return res;
}

// implmented based on "A Parallel Multigrid Poisson Solver for Fluids
// Simulation on Large Grids" by McAdams et al.
// Note: this solver only handles pure Neumann boundary condition
// TODO: add an option for handling Dirichlet boundary condition
void mgpcgSolve(const PoissonSolverOption& option,
                const MultigridOption& mg_option,
                const CgSolverOption& cg_option) {
  std::printf("start mgpcg\n");
  double* x = option.input_vars;
  double* f = option.input_rhs;
  double* r = option.residual;
  int n = option.width, m = option.height;
  double scale = normLinf(f, n, m);
  scaleDiv(f, scale, n, m);
  scaleDiv(x, scale, n, m);
  double tolerance = option.tolerance / scale;
  if (tolerance < 1e-5) tolerance = 1e-5;
  computeResidual(x, f, r, n, m);
  double mu = computeNullSpace(r, n, m);
  double* p = cg_option.aux_var_step;
  double* z = cg_option.aux_var_Ap;
  double v = normSubAve(r, mu, n, m);
  if (v < tolerance) {
    scaleMul(x, scale, n, m);
    return ;
  }
  double rho = precondAndDot(p, r, mu, n, m, mg_option);
  for (int i = 0; ; i++) {
    printf("iter %d, res = %f\n", i, v);
    applyLaplacian(z, p, n, m);
    double sigma = dot(p, z, n, m);
    double alpha = rho / sigma;
    saxpy(r, -alpha, z, n, m);
    mu = computeNullSpace(r, n, m);
    v = normSubAve(r, mu, n, m);
    if (v < tolerance || i == cg_option.max_iter - 1) {
      saxpy(x, alpha, p, n, m);
      scaleMul(x, scale, n, m);
      return ;
    }
    double rho_new = precondAndDot(z, r, mu, n, m, mg_option);
    double beta = rho_new / rho;
    rho = rho_new;
    saxpy(x, alpha, p, n, m);
    scaleAndAdd(p, beta, z, n, m);
  }
}

}  // namespace poisson