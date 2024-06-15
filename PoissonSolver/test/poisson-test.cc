#include <PoissonSolver/poisson-solver.h>
#include <PoissonSolver/multigrid-solver.h>
#include <PoissonSolver/cg-solver.h>
#include <PoissonSolver/util.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
using namespace poisson;
// implement some test cases for PoissonSolver
int main() {
  int n = 256, m = 256;
  // generate a rhs vector of size n * m
  std::vector<double> rhs(n * m);
  for (int i = 0; i < n * m; i++)
    rhs[i] = rand() / (double)RAND_MAX;
  std::vector<double> sol(n * m, 0.0);
  std::vector<double> aux_cg_p(n * m, 0.0);
  std::vector<double> aux_cg_Ap(n * m, 0.0);
  std::vector<double> aux_mg(n * m, 0.0);
  std::vector<double> aux_mg_rhs(n * m, 0.0);
  std::vector<double> mg_residual(n * m, 0.0);
  std::vector<double> residual(n * m, 0.0);
  MultigridOption mg_option;
  mg_option.max_level = 4;
  mg_option.smooth_iter = 10;
  mg_option.bottom_iter = 60;
  mg_option.aux_vars = aux_mg.data();
  mg_option.aux_rhs = aux_mg_rhs.data();
  mg_option.residual = mg_residual.data();
  CgSolverOption cg_option;
  cg_option.max_iter = 10;
  cg_option.aux_var_step = aux_cg_p.data();
  cg_option.aux_var_Ap = aux_cg_Ap.data();
  PoissonSolverOption option;
  option.width = n;
  option.height = m;
  option.input_rhs = rhs.data();
  option.input_vars = sol.data();
  option.residual = residual.data();
  // multigridSolve(option, mg_option);
  mgpcgSolve(option, mg_option, cg_option);
  //cgSolve(option, cg_option);
  printf("Residual is %f\n", normLinf(option.residual, n, m));
}