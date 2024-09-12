//
// Created by creeper on 7/14/24.
//
#include <fem/ipc/distances.h>
namespace fem::ipc {

Real distancePointTriangle(const Vector<Real, 3> &p,
                           const Vector<Real, 3> &a,
                           const Vector<Real, 3> &b,
                           const Vector<Real, 3> &c) {
  auto type = decidePointTriangleDistanceType(p, a, b, c);
  switch (type) {
    case PointTriangleDistanceType::P_A:return distancePointPoint(p, a);
    case PointTriangleDistanceType::P_B:return distancePointPoint(p, b);
    case PointTriangleDistanceType::P_C:return distancePointPoint(p, c);
    case PointTriangleDistanceType::P_AB:return distancePointLine(p, a, b);
    case PointTriangleDistanceType::P_BC:return distancePointLine(p, b, c);
    case PointTriangleDistanceType::P_CA:return distancePointLine(p, c, a);
    case PointTriangleDistanceType::P_ABC:return distancePointPlane(p, a, b, c);
    default:throw std::runtime_error("Unknown distance type");
  }
}
Vector<Real, 12> localDistancePointTriangleGradient(const Vector<Real, 3> &p,
                                                    const Vector<Real, 3> &a,
                                                    const Vector<Real, 3> &b,
                                                    const Vector<Real, 3> &c) {
  auto type = decidePointTriangleDistanceType(p, a, b, c);
  Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      Vector<Real, 6> grad = localDistancePointPointGradient(p, a);
      local_grad.segment<6>(0) = grad;
      return local_grad;
    }
    case PointTriangleDistanceType::P_B: {
      Vector<Real, 6> grad = localDistancePointPointGradient(p, b);
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<3>(6) = grad.segment<3>(3);
      return local_grad;
    }
    case PointTriangleDistanceType::P_C: {
      Vector<Real, 6> grad = localDistancePointPointGradient(p, c);
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<3>(9) = grad.segment<3>(3);
      return local_grad;
    }
    case PointTriangleDistanceType::P_AB: {
      Vector<Real, 9> grad = localDistancePointLineGradient(p, a, b);
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<6>(3) = grad.segment<6>(3);
      return local_grad;
    }
    case PointTriangleDistanceType::P_BC: {
      Vector<Real, 9> grad = localDistancePointLineGradient(p, b, c);
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<6>(6) = grad.segment<6>(3);
      return local_grad;
    }
    case PointTriangleDistanceType::P_CA: {
      Vector<Real, 9> grad = localDistancePointLineGradient(p, a, c);
      local_grad.segment<6>(0) = grad.segment<6>(0);
      local_grad.segment<3>(9) = grad.segment<3>(6);
      return local_grad;
    }
    case PointTriangleDistanceType::P_ABC:return localDistancePointPlaneGradient(p, a, b, c);
    default:throw std::runtime_error("Unknown distance type");
  }
}
Matrix<Real, 12, 12> localDistancePointTriangleHessian(const Vector<Real, 3> &p,
                                                       const Vector<Real, 3> &a,
                                                       const Vector<Real, 3> &b,
                                                       const Vector<Real, 3> &c) {
  auto type = decidePointTriangleDistanceType(p, a, b, c);
  Matrix<Real, 12, 12> local_hessian = Matrix<Real, 12, 12>::Zero();
  switch (type) {
    case PointTriangleDistanceType::P_A: {
      Matrix<Real, 6, 6> hessian = localDistancePointPointHessian(p, a);
      local_hessian.block<6, 6>(0, 0) = hessian;
      return local_hessian;
    }
    case PointTriangleDistanceType::P_B: {
      Matrix<Real, 6, 6> hessian = localDistancePointPointHessian(p, b);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(0, 6) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(6, 0) = hessian.block<3, 3>(3, 0);
      local_hessian.block<3, 3>(6, 6) = hessian.block<3, 3>(3, 3);
      return local_hessian;
    }
    case PointTriangleDistanceType::P_C: {
      Matrix<Real, 6, 6> hessian = localDistancePointPointHessian(p, c);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(0, 9) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(9, 0) = hessian.block<3, 3>(3, 0);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(3, 3);
      return local_hessian;
    }
    case PointTriangleDistanceType::P_AB: {
      Matrix<Real, 9, 9> hessian = localDistancePointLineHessian(p, a, b);
      local_hessian.block<9, 9>(0, 0) = hessian;
      return local_hessian;
    }
    case PointTriangleDistanceType::P_BC: {
      Matrix<Real, 9, 9> hessian = localDistancePointLineHessian(p, b, c);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 6>(0, 6) = hessian.block<3, 6>(0, 3);
      local_hessian.block<6, 3>(6, 0) = hessian.block<6, 3>(3, 0);
      local_hessian.block<6, 6>(6, 6) = hessian.block<6, 6>(3, 3);
      return local_hessian;
    }
    case PointTriangleDistanceType::P_CA: {
      Matrix<Real, 9, 9> hessian = localDistancePointLineHessian(p, a, c);
      local_hessian.block<6, 6>(0, 0) = hessian.block<6, 6>(0, 0);
      local_hessian.block<3, 6>(9, 0) = hessian.block<3, 6>(6, 0);
      local_hessian.block<6, 3>(0, 9) = hessian.block<6, 3>(0, 6);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(6, 6);
    }
    case PointTriangleDistanceType::P_ABC:return localDistancePointPlaneHessian(p, a, b, c);
    default:throw std::runtime_error("Unknown distance type");
  }
}
Real distanceEdgeEdge(const Vector<Real, 3> &ea0,
                      const Vector<Real, 3> &ea1,
                      const Vector<Real, 3> &eb0,
                      const Vector<Real, 3> &eb1) {
  auto type = decideEdgeEdgeDistanceType(ea0, ea1, eb0, eb1);
  switch (type) {
    case EdgeEdgeDistanceType::A_C:return distancePointPoint(ea0, eb0);
    case EdgeEdgeDistanceType::A_D:return distancePointPoint(ea0, eb1);
    case EdgeEdgeDistanceType::B_C:return distancePointPoint(ea1, eb0);
    case EdgeEdgeDistanceType::B_D:return distancePointPoint(ea1, eb1);
    case EdgeEdgeDistanceType::A_CD:return distancePointLine(ea0, eb0, eb1);
    case EdgeEdgeDistanceType::B_CD:return distancePointLine(ea1, eb0, eb1);
    case EdgeEdgeDistanceType::AB_C:return distancePointLine(eb0, ea0, ea1);
    case EdgeEdgeDistanceType::AB_D:return distancePointLine(eb1, ea0, ea1);
    case EdgeEdgeDistanceType::AB_CD:return distanceLineLine(ea0, ea1, eb0, eb1);
    default:throw std::runtime_error("Unknown distance type");
  }
}
Vector<Real, 12> localDistanceEdgeEdgeGradient(const Vector<Real, 3> &a,
                                               const Vector<Real, 3> &b,
                                               const Vector<Real, 3> &c,
                                               const Vector<Real, 3> &d) {
  auto type = decideEdgeEdgeDistanceType(a, b, c, d);
  switch (type) {
    case EdgeEdgeDistanceType::A_C: {
      Vector<Real, 6> grad = localDistancePointPointGradient(a, c);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<3>(6) = grad.segment<3>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::A_D: {
      Vector<Real, 6> grad = localDistancePointPointGradient(a, d);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<3>(9) = grad.segment<3>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::B_C: {
      Vector<Real, 6> grad = localDistancePointPointGradient(b, c);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<6>(3) = grad.segment<6>(0);
      return local_grad;
    }
    case EdgeEdgeDistanceType::B_D: {
      Vector<Real, 6> grad = localDistancePointPointGradient(b, d);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(3) = grad.segment<3>(0);
      local_grad.segment<3>(9) = grad.segment<3>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::A_CD: {
      Vector<Real, 9> grad = localDistancePointLineGradient(a, c, d);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<6>(6) = grad.segment<6>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::B_CD: {
      Vector<Real, 9> grad = localDistancePointLineGradient(b, c, d);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(3) = grad.segment<3>(0);
      local_grad.segment<6>(6) = grad.segment<6>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::AB_C: {
      Vector<Real, 9> grad = localDistancePointLineGradient(c, a, b);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<6>(0) = grad.segment<6>(3);
      local_grad.segment<3>(6) = grad.segment<3>(0);
    }
    case EdgeEdgeDistanceType::AB_D: {
      Vector<Real, 9> grad = localDistancePointLineGradient(d, a, b);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<6>(0) = grad.segment<6>(3);
      local_grad.segment<3>(9) = grad.segment<3>(0);
    }
    case EdgeEdgeDistanceType::AB_CD:return localDistanceLineLineGradient(a, b, c, d);
    default:throw std::runtime_error("Unknown distance type");
  }
}
Matrix<Real, 12, 12> localDistanceEdgeEdgeHessian(const Vector<Real, 3> &a,
                                                  const Vector<Real, 3> &b,
                                                  const Vector<Real, 3> &c,
                                                  const Vector<Real, 3> &d) {
  auto type = decideEdgeEdgeDistanceType(a, b, c, d);
  Matrix<Real, 12, 12> local_hessian = Matrix<Real, 12, 12>::Zero();
  switch (type) {
    case EdgeEdgeDistanceType::A_C: {
      Matrix<Real, 6, 6> hessian = localDistancePointPointHessian(a, c);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(6, 6) = hessian.block<3, 3>(3, 3);
      local_hessian.block<3, 3>(0, 6) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(6, 0) = hessian.block<3, 3>(3, 0);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::A_D: {
      Matrix<Real, 6, 6> hessian = localDistancePointPointHessian(a, d);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(3, 3);
      local_hessian.block<3, 3>(0, 9) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(9, 0) = hessian.block<3, 3>(3, 0);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::B_C: {
      Matrix<Real, 6, 6> hessian = localDistancePointPointHessian(b, c);
      local_hessian.block<6, 6>(3, 3) = hessian;
      return local_hessian;
    }
    case EdgeEdgeDistanceType::B_D: {
      Matrix<Real, 6, 6> hessian = localDistancePointPointHessian(b, d);
      local_hessian.block<3, 3>(3, 3) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(3, 3);
      local_hessian.block<3, 3>(3, 9) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(9, 3) = hessian.block<3, 3>(3, 0);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::A_CD: {
      Matrix<Real, 9, 9> hessian = localDistancePointLineHessian(a, c, d);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 6>(0, 6) = hessian.block<3, 6>(0, 3);
      local_hessian.block<6, 3>(6, 0) = hessian.block<6, 3>(3, 0);
      local_hessian.block<6, 6>(6, 6) = hessian.block<6, 6>(3, 3);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::B_CD: {
      Matrix<Real, 9, 9> hessian = localDistancePointLineHessian(b, c, d);
      local_hessian.block<9, 9>(3, 3) = hessian;
      return local_hessian;
    }
    case EdgeEdgeDistanceType::AB_C: {
      Matrix<Real, 9, 9> hessian = localDistancePointLineHessian(c, a, b);
      local_hessian.block<9, 9>(0, 0) = hessian;
      return local_hessian;
    }
    case EdgeEdgeDistanceType::AB_D: {
      Matrix<Real, 9, 9> hessian = localDistancePointLineHessian(d, a, b);
      local_hessian.block<6, 6>(0, 0) = hessian.block<6, 6>(3, 3);
      local_hessian.block<3, 6>(9, 0) = hessian.block<3, 6>(6, 3);
      local_hessian.block<6, 3>(0, 9) = hessian.block<6, 3>(3, 6);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(6, 6);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::AB_CD:return localDistanceLineLineHessian(a, b, c, d);
    default:throw std::runtime_error("Unknown distance type");
  }
}

}