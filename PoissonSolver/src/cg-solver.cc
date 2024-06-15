#include <PoissonSolver/poisson-solver.h>
#include <PoissonSolver/cg-solver.h>
#include <PoissonSolver/util.h>
#include <cstring>
#include <cstdio>
#include <cmath>
namespace poisson {
void cgSolve(const PoissonSolverOption &option, const CgSolverOption &cg_option) {
  double *x = option.input_vars;
  double *f = option.input_rhs;
  double *r = option.residual;
  int n = option.width, m = option.height;
  double scale = normLinf(f, n, m);
  for (int i = 0; i < n * m; i++) {
    f[i] /= scale;
    x[i] /= scale;
  }
  computeResidual(x, f, r, n, m);
  double mu = computeNullSpace(r, n, m);
  for (int i = 0; i < n * m; i++)
    r[i] -= mu;
  double *p = cg_option.aux_var_step;
  memcpy(p, r, sizeof(double) * n * m);
  double *Ap = cg_option.aux_var_Ap;
  double v = normLinf(r, n, m);
  double rho = normSqr(r, n, m);
  double tolerance = static_cast<double>(option.tolerance) / static_cast<double>(scale);
  if (v < tolerance) {
    for (int i = 0; i < n * m; i++)
      x[i] *= scale;
    return;
  }
  for (int i = 0; ; i++) {
    applyLaplacian(Ap, p, n, m);
    double sigma = dot(p, Ap, n, m);
    double alpha = rho / sigma;
    saxpy(r, -alpha, Ap, n, m);
    mu = computeNullSpace(r, n, m);
    for (int j = 0; j < n * m; j++)
      r[i] -= mu;
    v = normLinf(r, n, m);
    double rho_new = normSqr(r, n, m);
//    printf("iter %d, residual %f\n", i, v);
    if (v <= tolerance || i == cg_option.max_iter - 1) {
      saxpy(x, alpha, p, n, m);
      for (int j = 0; j < n * m; j++)
        x[j] *= scale;
      return;
    }
    double beta = rho_new / rho;
    rho = rho_new;
    saxpy(x, alpha, p, n, m);
    scaleAndAdd(p, beta, r, n, m);
  }
}
}