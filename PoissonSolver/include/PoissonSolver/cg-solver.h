// add header protection
#ifndef CG_SOLVER_H
#define CG_SOLVER_H

#include <PoissonSolver/poisson-solver.h>

namespace poisson {
    enum struct Preconditioner {
        NONE,
        MULTIGRID,
    };
    struct CgSolverOption {
        int max_iter = 100;
        float* aux_var_step = nullptr;
        float* aux_var_Ap = nullptr;
        Preconditioner preconditioner = Preconditioner::MULTIGRID;
    };
}

#endif