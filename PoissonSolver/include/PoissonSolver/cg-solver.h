// add header protection
#ifndef CG_SOLVER_H
#define CG_SOLVER_H

#include <PoissonSolver/poisson-solver.h>

namespace poisson {
enum struct Preconditioner {
  None,
  Multigrid,
};
struct CgSolverOption {
  int max_iter = 100;
  double *aux_var_step = nullptr;
  double *aux_var_Ap = nullptr;
  Preconditioner preconditioner = Preconditioner::Multigrid;
};
void cgSolve(const PoissonSolverOption &option,
             const CgSolverOption &cg_option);
}

#endif