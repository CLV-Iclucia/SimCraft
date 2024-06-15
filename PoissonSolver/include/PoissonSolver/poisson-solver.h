#ifndef POISSONSOLVER_H_
#define POISSONSOLVER_H_

// the design of this solver is that it does not store any internal state
// so that it is naturally thread-safe
namespace poisson {
enum struct Device {
    CPU,
    GPU
};

enum struct Method {
    JACOBI,
    GAUSS_SEIDEL,
    MULTIGRID,
    CG
};

enum struct Norm {
    L1,
    L2,
    LINF
};

struct PoissonSolverOption {
    int dim = 2;
    int width = -1, height = -1, depth = -1;
    Device input_device = Device::CPU;
    Device solving_device = Device::CPU;
    Device output_device = Device::CPU;
    double* input_vars;
    double* input_rhs;
    double* residual = nullptr; // if nullptr, auxilary memory will be allocated
    double* output_vars = nullptr; // if nullptr, input will be overwritten
    double tolerance = 1e-6;
    Method method = Method::CG;
};

struct MultigridOption;
struct CgSolverOption;

void poissonSolve(const PoissonSolverOption& option);
void multigridSolve(const PoissonSolverOption& option, const MultigridOption& mg_option);
void mgpcgSolve(const PoissonSolverOption& option, const MultigridOption& mg_option, const CgSolverOption& cg_option);
}

#endif