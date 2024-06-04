//
// Created by creeper on 5/23/24.
//

#ifndef SIMCRAFT_MATHS_INCLUDE_MATHS_AMG_PRECONDER_H_
#define SIMCRAFT_MATHS_INCLUDE_MATHS_AMG_PRECONDER_H_
#include <Maths/types.h>
#include <Eigen/IterativeLinearSolvers>
namespace maths {
template<typename T>
struct AlgebraicMultigridPreconditioner {
  using Scalar = T;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  typedef typename Vector::Index Index;

 public:
  // this typedef is only to export the scalar type and compile-time dimensions to solve_retval
  using MatrixType = SparseMatrix<Scalar>;

  AlgebraicMultigridPreconditioner() : m_isInitialized(false) {}

  template<typename MatType>
  AlgebraicMultigridPreconditioner(const MatType &mat) {
    compute(mat);
  }

  Index rows() const {  }
  Index cols() const {  }

  template<typename MatType>
  AlgebraicMultigridPreconditioner &analyzePattern(const MatType &) {
    return *this;
  }

  template<typename MatType>
  AlgebraicMultigridPreconditioner &factorize(const MatType &mat) {
    m_isInitialized = true;
    return *this;
  }

  template<typename MatType>
  AlgebraicMultigridPreconditioner &compute(const MatType &mat) {

  }

  template<typename Rhs, typename Dest>
  void _solve(const Rhs &b, Dest &x) const {

  }

  template<typename Rhs>
  inline const Eigen::internal::solve_retval<AlgebraicMultigridPreconditioner, Rhs>
  solve(const Eigen::MatrixBase<Rhs> &b) const {
    eigen_assert(m_isInitialized && "DiagonalPreconditioner is not initialized.");
    eigen_assert(m_invdiag.size() == b.rows()
                     && "DiagonalPreconditioner::solve(): invalid number of rows of the right hand side matrix b");
    return Eigen::internal::solve_retval<AlgebraicMultigridPreconditioner, Rhs>(*this, b.derived());
  }

 protected:
  bool m_isInitialized;

};
template<typename MatrixType, int UpLo>
using AmgPcgSolver = Eigen::ConjugateGradient<MatrixType,
                                              UpLo,
                                              AlgebraicMultigridPreconditioner<typename MatrixType::Scalar>>;
}
#endif //SIMCRAFT_MATHS_INCLUDE_MATHS_AMG_PRECONDER_H_
