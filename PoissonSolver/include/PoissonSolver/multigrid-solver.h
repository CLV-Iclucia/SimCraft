// add header protection
#ifndef MULTIGRID_SOLVER_H
#define MULTIGRID_SOLVER_H
#include <PoissonSolver/poisson-solver.h>
namespace poisson {

enum struct RelaxationMethod {
    RbGaussSeidel,
};


struct MultigridOption {
    int max_level = -1;
    int nthread = 1;
    int smooth_iter = 2;
    int bottom_iter = 20;
    double tolerance = 1e-6;
    double* aux_vars = nullptr;
    double* aux_rhs = nullptr; // if nullptr, auxilary memory will be allocated
    double* residual = nullptr;
    RelaxationMethod method = RelaxationMethod::RbGaussSeidel;
};

void bottomSolve(double* u, double* f, int n, int m);
}

#endif