//
// Created by creeper on 7/14/24.
//
#include <fem/ipc/distances.h>
namespace sim::fem::ipc {

Real distanceSqrPointTriangle(const Vector<Real, 3> &p,
                              const Vector<Real, 3> &a,
                              const Vector<Real, 3> &b,
                              const Vector<Real, 3> &c) {
  auto type = decidePointTriangleDistanceType(p, a, b, c);
  switch (type) {
    case PointTriangleDistanceType::P_A:return distanceSqrPointPoint(p, a);
    case PointTriangleDistanceType::P_B:return distanceSqrPointPoint(p, b);
    case PointTriangleDistanceType::P_C:return distanceSqrPointPoint(p, c);
    case PointTriangleDistanceType::P_AB:return distanceSqrPointLine(p, a, b);
    case PointTriangleDistanceType::P_BC:return distanceSqrPointLine(p, b, c);
    case PointTriangleDistanceType::P_CA:return distanceSqrPointLine(p, c, a);
    case PointTriangleDistanceType::P_ABC:return distanceSqrPointPlane(p, a, b, c);
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
      Vector<Real, 6> grad = localDistanceSqrPointPointGradient(p, a);
      local_grad.segment<6>(0) = grad;
      return local_grad;
    }
    case PointTriangleDistanceType::P_B: {
      Vector<Real, 6> grad = localDistanceSqrPointPointGradient(p, b);
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<3>(6) = grad.segment<3>(3);
      return local_grad;
    }
    case PointTriangleDistanceType::P_C: {
      Vector<Real, 6> grad = localDistanceSqrPointPointGradient(p, c);
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<3>(9) = grad.segment<3>(3);
      return local_grad;
    }
    case PointTriangleDistanceType::P_AB: {
      Vector<Real, 9> grad = localDistanceSqrPointLineGradient(p, a, b);
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<6>(3) = grad.segment<6>(3);
      return local_grad;
    }
    case PointTriangleDistanceType::P_BC: {
      Vector<Real, 9> grad = localDistanceSqrPointLineGradient(p, b, c);
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<6>(6) = grad.segment<6>(3);
      return local_grad;
    }
    case PointTriangleDistanceType::P_CA: {
      Vector<Real, 9> grad = localDistanceSqrPointLineGradient(p, a, c);
      local_grad.segment<6>(0) = grad.segment<6>(0);
      local_grad.segment<3>(9) = grad.segment<3>(6);
      return local_grad;
    }
    case PointTriangleDistanceType::P_ABC:return localDistanceSqrPointPlaneGradient(p, a, b, c);
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
      Matrix<Real, 6, 6> hessian = localDistanceSqrPointPointHessian(p, a);
      local_hessian.block<6, 6>(0, 0) = hessian;
      return local_hessian;
    }
    case PointTriangleDistanceType::P_B: {
      Matrix<Real, 6, 6> hessian = localDistanceSqrPointPointHessian(p, b);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(0, 6) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(6, 0) = hessian.block<3, 3>(3, 0);
      local_hessian.block<3, 3>(6, 6) = hessian.block<3, 3>(3, 3);
      return local_hessian;
    }
    case PointTriangleDistanceType::P_C: {
      Matrix<Real, 6, 6> hessian = localDistanceSqrPointPointHessian(p, c);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(0, 9) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(9, 0) = hessian.block<3, 3>(3, 0);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(3, 3);
      return local_hessian;
    }
    case PointTriangleDistanceType::P_AB: {
      Matrix<Real, 9, 9> hessian = localDistanceSqrPointLineHessian(p, a, b);
      local_hessian.block<9, 9>(0, 0) = hessian;
      return local_hessian;
    }
    case PointTriangleDistanceType::P_BC: {
      Matrix<Real, 9, 9> hessian = localDistanceSqrPointLineHessian(p, b, c);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 6>(0, 6) = hessian.block<3, 6>(0, 3);
      local_hessian.block<6, 3>(6, 0) = hessian.block<6, 3>(3, 0);
      local_hessian.block<6, 6>(6, 6) = hessian.block<6, 6>(3, 3);
      return local_hessian;
    }
    case PointTriangleDistanceType::P_CA: {
      Matrix<Real, 9, 9> hessian = localDistanceSqrPointLineHessian(p, a, c);
      local_hessian.block<6, 6>(0, 0) = hessian.block<6, 6>(0, 0);
      local_hessian.block<3, 6>(9, 0) = hessian.block<3, 6>(6, 0);
      local_hessian.block<6, 3>(0, 9) = hessian.block<6, 3>(0, 6);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(6, 6);
    }
    case PointTriangleDistanceType::P_ABC:return localDistanceSqrPointPlaneHessian(p, a, b, c);
    default:throw std::runtime_error("Unknown distance type");
  }
}
Real distanceSqrEdgeEdge(const Vector<Real, 3> &ea0,
                         const Vector<Real, 3> &ea1,
                         const Vector<Real, 3> &eb0,
                         const Vector<Real, 3> &eb1) {
  auto type = decideEdgeEdgeDistanceType(ea0, ea1, eb0, eb1);
  switch (type) {
    case EdgeEdgeDistanceType::A_C:return distanceSqrPointPoint(ea0, eb0);
    case EdgeEdgeDistanceType::A_D:return distanceSqrPointPoint(ea0, eb1);
    case EdgeEdgeDistanceType::B_C:return distanceSqrPointPoint(ea1, eb0);
    case EdgeEdgeDistanceType::B_D:return distanceSqrPointPoint(ea1, eb1);
    case EdgeEdgeDistanceType::A_CD:return distanceSqrPointLine(ea0, eb0, eb1);
    case EdgeEdgeDistanceType::B_CD:return distanceSqrPointLine(ea1, eb0, eb1);
    case EdgeEdgeDistanceType::AB_C:return distanceSqrPointLine(eb0, ea0, ea1);
    case EdgeEdgeDistanceType::AB_D:return distanceSqrPointLine(eb1, ea0, ea1);
    case EdgeEdgeDistanceType::AB_CD:return distanceSqrLineLine(ea0, ea1, eb0, eb1);
    default:throw std::runtime_error("Unknown distance type");
  }
}
Vector<Real, 12> localDistanceSqrEdgeEdgeGradient(const Vector<Real, 3> &a,
                                                  const Vector<Real, 3> &b,
                                                  const Vector<Real, 3> &c,
                                                  const Vector<Real, 3> &d) {
  auto type = decideEdgeEdgeDistanceType(a, b, c, d);
  switch (type) {
    case EdgeEdgeDistanceType::A_C: {
      Vector<Real, 6> grad = localDistanceSqrPointPointGradient(a, c);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<3>(6) = grad.segment<3>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::A_D: {
      Vector<Real, 6> grad = localDistanceSqrPointPointGradient(a, d);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<3>(9) = grad.segment<3>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::B_C: {
      Vector<Real, 6> grad = localDistanceSqrPointPointGradient(b, c);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<6>(3) = grad.segment<6>(0);
      return local_grad;
    }
    case EdgeEdgeDistanceType::B_D: {
      Vector<Real, 6> grad = localDistanceSqrPointPointGradient(b, d);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(3) = grad.segment<3>(0);
      local_grad.segment<3>(9) = grad.segment<3>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::A_CD: {
      Vector<Real, 9> grad = localDistanceSqrPointLineGradient(a, c, d);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(0) = grad.segment<3>(0);
      local_grad.segment<6>(6) = grad.segment<6>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::B_CD: {
      Vector<Real, 9> grad = localDistanceSqrPointLineGradient(b, c, d);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<3>(3) = grad.segment<3>(0);
      local_grad.segment<6>(6) = grad.segment<6>(3);
      return local_grad;
    }
    case EdgeEdgeDistanceType::AB_C: {
      Vector<Real, 9> grad = localDistanceSqrPointLineGradient(c, a, b);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<6>(0) = grad.segment<6>(3);
      local_grad.segment<3>(6) = grad.segment<3>(0);
    }
    case EdgeEdgeDistanceType::AB_D: {
      Vector<Real, 9> grad = localDistanceSqrPointLineGradient(d, a, b);
      Vector<Real, 12> local_grad = Vector<Real, 12>::Zero();
      local_grad.segment<6>(0) = grad.segment<6>(3);
      local_grad.segment<3>(9) = grad.segment<3>(0);
    }
    case EdgeEdgeDistanceType::AB_CD:return localDistanceSqrLineLineGradient(a, b, c, d);
    default:throw std::runtime_error("Unknown distance type");
  }
}
Matrix<Real, 12, 12> localDistanceSqrEdgeEdgeHessian(const Vector<Real, 3> &a,
                                                     const Vector<Real, 3> &b,
                                                     const Vector<Real, 3> &c,
                                                     const Vector<Real, 3> &d) {
  auto type = decideEdgeEdgeDistanceType(a, b, c, d);
  Matrix<Real, 12, 12> local_hessian = Matrix<Real, 12, 12>::Zero();
  switch (type) {
    case EdgeEdgeDistanceType::A_C: {
      Matrix<Real, 6, 6> hessian = localDistanceSqrPointPointHessian(a, c);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(6, 6) = hessian.block<3, 3>(3, 3);
      local_hessian.block<3, 3>(0, 6) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(6, 0) = hessian.block<3, 3>(3, 0);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::A_D: {
      Matrix<Real, 6, 6> hessian = localDistanceSqrPointPointHessian(a, d);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(3, 3);
      local_hessian.block<3, 3>(0, 9) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(9, 0) = hessian.block<3, 3>(3, 0);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::B_C: {
      Matrix<Real, 6, 6> hessian = localDistanceSqrPointPointHessian(b, c);
      local_hessian.block<6, 6>(3, 3) = hessian;
      return local_hessian;
    }
    case EdgeEdgeDistanceType::B_D: {
      Matrix<Real, 6, 6> hessian = localDistanceSqrPointPointHessian(b, d);
      local_hessian.block<3, 3>(3, 3) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(3, 3);
      local_hessian.block<3, 3>(3, 9) = hessian.block<3, 3>(0, 3);
      local_hessian.block<3, 3>(9, 3) = hessian.block<3, 3>(3, 0);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::A_CD: {
      Matrix<Real, 9, 9> hessian = localDistanceSqrPointLineHessian(a, c, d);
      local_hessian.block<3, 3>(0, 0) = hessian.block<3, 3>(0, 0);
      local_hessian.block<3, 6>(0, 6) = hessian.block<3, 6>(0, 3);
      local_hessian.block<6, 3>(6, 0) = hessian.block<6, 3>(3, 0);
      local_hessian.block<6, 6>(6, 6) = hessian.block<6, 6>(3, 3);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::B_CD: {
      Matrix<Real, 9, 9> hessian = localDistanceSqrPointLineHessian(b, c, d);
      local_hessian.block<9, 9>(3, 3) = hessian;
      return local_hessian;
    }
    case EdgeEdgeDistanceType::AB_C: {
      Matrix<Real, 9, 9> hessian = localDistanceSqrPointLineHessian(c, a, b);
      local_hessian.block<9, 9>(0, 0) = hessian;
      return local_hessian;
    }
    case EdgeEdgeDistanceType::AB_D: {
      Matrix<Real, 9, 9> hessian = localDistanceSqrPointLineHessian(d, a, b);
      local_hessian.block<6, 6>(0, 0) = hessian.block<6, 6>(3, 3);
      local_hessian.block<3, 6>(9, 0) = hessian.block<3, 6>(6, 3);
      local_hessian.block<6, 3>(0, 9) = hessian.block<6, 3>(3, 6);
      local_hessian.block<3, 3>(9, 9) = hessian.block<3, 3>(6, 6);
      return local_hessian;
    }
    case EdgeEdgeDistanceType::AB_CD:return localDistanceSqrLineLineHessian(a, b, c, d);
    default:throw std::runtime_error("Unknown distance type");
  }
}

}