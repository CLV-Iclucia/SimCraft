//
// Created by creeper on 5/28/24.
//

#pragma once
#include <fem/ipc/distances.h>
#include <fem/ipc/barrier-functions.h>
#include <fem/simplex.h>
#include <Maths/block-sparse-matrix.h>
#include <Maths/block-types.h>

namespace sim {
namespace fem {
struct System;
namespace ipc {
struct LogBarrier;

using maths::LocalGrad;
using maths::LocalHessian;

struct VertexTriangleConstraint {
  Triangle triangle;
  const maths::BlockVector<3>& x;  // 全局 BlockVector 引用
  int globalVertex;                 // 全局顶点 block 索引
  int globalTriVerts[3];           // triangle 三顶点的全局 block 索引
  PointTriangleDistanceType type{};
  
  void updateDistanceType();
  [[nodiscard]] Real distanceSqr() const;
  // Block interface
  void assembleBarrierGradient(const LogBarrier &barrier,
                              maths::BlockVector<3> &globalGradient,
                              Real kappa) const;
  void assembleBarrierHessian(const LogBarrier &barrier,
                              maths::BlockSparseMatrix<3> &globalHessian,
                              Real kappa) const;
};

struct EdgeEdgeConstraint {
  const maths::BlockVector<3>& x;   // 全局当前配置
  const maths::BlockVector<3>& X;   // 全局参考配置(mollifier用)
  int globalEdgeA[2];               // edge A 两端点全局 block 索引
  int globalEdgeB[2];               // edge B 两端点全局 block 索引
  EdgeEdgeDistanceType type{};
  void updateDistanceType();
  [[nodiscard]] Real distanceSqr() const;
  // Block interface
  void assembleMollifiedBarrierGradient(const LogBarrier &barrier,
                                        maths::BlockVector<3> &globalGradient,
                                        Real kappa) const;
  void assembleMollifiedBarrierHessian(
      const LogBarrier &barrier,
      maths::BlockSparseMatrix<3> &globalHessian, Real kappa) const;
  [[nodiscard]] Real mollifier() const;
  [[nodiscard]] LocalGrad<4> mollifierGradient() const;
  [[nodiscard]] LocalHessian<4> mollifierHessian() const;
  [[nodiscard]] LocalGrad<4>
  mollifiedBarrierGradient(const LogBarrier &barrier) const;
  [[nodiscard]] LocalHessian<4>
  mollifiedBarrierHessian(const LogBarrier &barrier) const;

private:
  [[nodiscard]] Real epsCross() const;
  [[nodiscard]] LocalGrad<4> crossedNormGradient() const;
  [[nodiscard]] LocalHessian<4> crossedNormHessian() const;
  [[nodiscard]] Real crossSquaredNorm() const;

  static Real mollifier(Real c, Real e_x) {
    if (c < e_x)
      return -(c / e_x) * (c / e_x) + 2 * (c / e_x);
    return 1.0;
  }
  static Real mollifierDerivative(Real c, Real e_x) {
    if (c < e_x)
      return -2 * c / (e_x * e_x) + 2 / e_x;
    return 0.0;
  }
  static Real mollifierSecondDerivative(Real c, Real e_x) {
    if (c < e_x)
      return -2 / (e_x * e_x);
    return 0.0;
  }
};

/// 弹性顶点 vs 运动学三角形的 barrier 约束 (单侧)
/// 运动学体的位置不计入 DOF，仅弹性体顶点有梯度/Hessian 贡献
struct DeformableKinematicVTConstraint {
  const maths::BlockVector<3>& x;  // 全局弹性体位置
  int deformableVertex;             // 弹性体顶点全局 block 索引
  glm::dvec3 ka, kb, kc;       // kinematic 三角形当前位置 (非 DOF)
  PointTriangleDistanceType type{};
  
  void updateDistanceType() {
    auto p = x[deformableVertex];
    type = decidePointTriangleDistanceType(p, ka, kb, kc);
  }
  
  [[nodiscard]] Real distanceSqr() const {
    return distanceSqrPointTriangle(x[deformableVertex], ka, kb, kc);
  }
  
  /// 梯度只贡献到弹性顶点（1 个 block）
  void assembleBarrierGradient(const LogBarrier& barrier,
                               maths::BlockVector<3>& grad, Real kappa) const {
    // 计算局部梯度（类似 VertexTriangleConstraint 但只取弹性顶点的部分）
    auto p = x[deformableVertex];
    auto distSqr = distanceSqrPointTriangle(p, ka, kb, kc);
    auto gradSqr = localDistancePointTriangleGradient(p, ka, kb, kc);

    // 只取弹性顶点部分 (gradSqr[0] 即 point p 的梯度 block)
    glm::dvec3 localGrad = gradSqr[0];

    grad[deformableVertex] += localGrad * (kappa * barrier.distanceSqrGradient(distSqr));
  }
  
  /// Hessian 只有 1×1 block（弹性顶点对自己）
  void assembleBarrierHessian(const LogBarrier& barrier,
                              maths::BlockSparseMatrix<3>& H, Real kappa) const {
    auto p = x[deformableVertex];
    auto distSqr = distanceSqrPointTriangle(p, ka, kb, kc);
    auto hessSqr = localDistancePointTriangleHessian(p, ka, kb, kc);
    // Extract the top-left 3×3 block: ∂²d²/∂p∂p (deformable vertex self-Hessian)
    glm::dmat3 localHess = hessSqr[0][0];
    
    Real b1 = barrier.distanceSqrGradient(distSqr);
    Real b2 = barrier.distanceSqrHessian(distSqr);
    
    // H = kappa * (b2 * g * g^T + b1 * H_sqr)
    auto gradSqr = localDistancePointTriangleGradient(p, ka, kb, kc);
    glm::dvec3 g = gradSqr[0];
    
    glm::dmat3 H_contrib = kappa * (b2 * glm::outerProduct(g, g) + b1 * localHess);
    
    H.addBlock(deformableVertex, deformableVertex, H_contrib);
  }
};

} // namespace ipc
} // namespace fem
} // namespace sim
