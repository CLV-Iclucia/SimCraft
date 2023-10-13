// add header protection
#ifndef MULTIGRID_SOLVER_H
#define MULTIGRID_SOLVER_H
#include <PoissonSolver/poisson-solver.h>
namespace poisson {

enum struct RelaxationMethod {
    RBGAUSS,
};


struct MultigridOption {
    int max_level = -1;
    int nthread = 1;
    int smooth_iter = 4;
    int bottom_iter = 20;
    float tolerance = 1e-6;
    float* aux_vars = nullptr;
    float* aux_rhs = nullptr; // if nullptr, auxilary memory will be allocated
    RelaxationMethod method = RelaxationMethod::RBGAUSS;
};

void bottomSolve(float* u, float* f, int n, int m);
}

#endif