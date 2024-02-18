//
// Created by creeper on 23-8-10.
//
#include "HairSim/grads.h"
#include <HairSim/forces.h>
#include <iostream>

namespace hairsim {
static void addStiffness(vector<Triplet<Real>> &J, int i, int j, Real v) {
  J.emplace_back(i, j, v);
}
void StretchingForce::computeElementForce(const Hair &hair, Index e,
                                          Vec4d &f) const {
  if (e > 0 && e < hair.NumVertices() - 1) {
    f << hair.E() * hair.area() *
             (((hair.edgeLength(e) / hair.referenceConfig().length[e]) - 1.0) *
                  hair.tangent(e) -
              ((hair.edgeLength(e - 1) / hair.referenceConfig().length[e - 1]) -
               1.0) *
                  hair.tangent(e - 1)),
        0.0;
  } else if (e == hair.NumVertices() - 1) {
    f << hair.E() * hair.area() *
             (((hair.edgeLength(e - 1) / hair.referenceConfig().length[e - 1]) -
               1.0) *
              hair.tangent(e - 1)),
        0.0;
  } else {
#ifndef NDEBUG
    std::cerr << "BendingForce::computeElementForce: "
              << "computing force on a fixed vertex" << std::endl;
#endif
  }
}
void StretchingForce::computeElementStiffness(const Hair &hair, Index e,
                                              Mat4d &H) const {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i < 3 && j < 3) {
        H(i, j) = hair.E() * hair.area() *
                  ((1.0 / hair.referenceConfig().length[e] -
                    1.0 / hair.edgeLength(e)) *
                       ((i == j) - hair.tangent(e)(i) * hair.tangent(e)(j)) +
                   hair.tangent(e)(i) * hair.tangent(e)(j) /
                       hair.referenceConfig().length[e]);
      } else
        H(i, j) = 0.0;
    }
  }
}
void StretchingForce::computeForce(const Hair &hair, VecXd &f) const {
  Vec4d f_i;
  for (int i = 0; i < hair.NumVertices(); i++) {
    computeElementForce(hair, i, f_i);
    hair.addForce(i, f_i.segment<3>(0));
  }
}
void StretchingForce::computeStiffness(const Hair &hair,
                                       vector<Triplet<Real>> &J) const {
  Mat4d H_i;
  for (int i = 0; i < hair.NumVertices() - 1; i++) {
    computeElementStiffness(hair, i, H_i);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        // add the stiffness to the corresponding position in the global
        addStiffness(J, 4 * i + j, 4 * i + k, H_i(j, k));
        addStiffness(J, 4 * i + j, 4 * (i + 1) + k, -H_i(j, k));
        addStiffness(J, 4 * (i + 1) + j, 4 * i + k, -H_i(j, k));
        addStiffness(J, 4 * (i + 1) + j, 4 * (i + 1) + k, H_i(j, k));
      }
    }
  }
}
void TwistingForce::computeElementForce(const Hair &hair, Index e,
                                        Vec4d &f) const {
  if (e >= 1 && e < hair.NumVertices() - 2) {
    f << pEt_pe(hair, e) - pEt_pe(hair, e - 1),
        -pEt_pm(hair, e) + pEt_pm(hair, e + 1);
  } else if (e == 0) {
    f << pEt_pe(hair, 0), -pEt_pm(hair, 0) + pEt_pm(hair, 1);
  } else if (e == hair.NumVertices() - 2) {
    f << pEt_pe(hair, e) - pEt_pe(hair, e - 1),
        -pEt_pm(hair, hair.NumVertices() - 2);
  } else {
#ifndef NDEBUG
    std::cerr << "TwistingForce::computeElementForce: "
              << "invalid index" << std::endl;
  }
#endif
}
void TwistingForce::computeElementStiffness(const Hair &hair, Index i, Index j,
                                            Mat4d &H) const {
  H.block<3, 3>(0, 0) =
      hair.G() * hair.area() / (4 * hair.edgeLength(i)) *
          (tensorProduct(pmpe(hair, i, i), pmpe(hair, i, j)) +
           hair.m(i) * p2m_pe2(hair, i, i, j)) +
      hair.G() * hair.area() / (4 * hair.edgeLength(i + 1)) *
          (tensorProduct(pmpe(hair, i + 1, i), pmpe(hair, i + 1, j)) +
           hair.m(i + 1) * p2m_pe2(hair, i + 1, i, j));
  H.block<3, 1>(0, 3) = Vec3d::Zero();
  H.block<1, 3>(3, 0) = Vec3d::Zero();
  H(3, 3) = 0.25 * hair.G() * hair.area() * hair.radius() * hair.radius() /
            hair.vertexReferenceLength(i);
}
void TwistingForce::computeForce(const Hair &hair, VecXd &f) const {
  Vec4d f_i;
  for (int i = 0; i < hair.NumVertices(); i++) {
    computeElementForce(hair, i, f_i);
    hair.addForce(i, f_i.segment<3>(0));
    hair.addTorsion(i, f_i(3));
  }
}
void TwistingForce::computeStiffness(const Hair &hair,
                                     vector<Triplet<Real>> &J) const {
  Mat4d H_i;
  for (int i = 0; i < hair.NumVertices() - 2; i++) {
    for (int j = std::max(i - 1, 0); j < i + 2; j++) {
      computeElementStiffness(hair, i, j, H_i);
      for (int k = 0; k < 3; k++)
        for (int l = 0; l < 3; l++)
          addStiffness(J, 4 * i + k, 4 * j + l, H_i(k, l));
    }
    addStiffness(J, 4 * i + 3, 4 * i + 3,
                 0.25 * hair.G() * hair.area() * hair.radius() * hair.radius() /
                     hair.vertexReferenceLength(i));
  }
}
void BendingForce::computeElementForce(const Hair &hair, Index e,
                                       Vec4d &f) const {}
void BendingForce::computeElementStiffness(const Hair &hair, Index i, Index j,
                                           Mat4d &H) const {}
void BendingForce::computeForce(const Hair &hair, VecXd &f) const {}
void BendingForce::computeStiffness(const Hair &hair,
                                    vector<Triplet<Real>> &J) const {}
} // namespace hairsim