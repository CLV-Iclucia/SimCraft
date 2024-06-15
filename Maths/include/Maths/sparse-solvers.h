//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_SPARSE_SOLVERS_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_SPARSE_SOLVERS_H_
#include <Maths/types.h>
namespace maths {
template <typename T>
struct SparseSolver {
  virtual bool solve(const SparseMatrix<T>& A, const Vector<T, Dynamic>& b, Vector<T, Dynamic>& x) = 0;
  virtual ~SparseSolver() = default;
};
template <typename T>
struct SparseCholeskySolver : SparseSolver<T> {
  void solve(const SparseMatrix<T>& A, const Vector<T, Dynamic>& b, Vector<T, Dynamic>& x) override {
    Eigen::SimplicialLDLT<SparseMatrix<T>> solver;
    solver.compute(A);
    x = solver.solve(b);
    auto result = solver.info();
    return result == Eigen::Success;
  }
};
}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_SPARSE_SOLVERS_H_
