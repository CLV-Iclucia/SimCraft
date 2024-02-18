//
// Created by creeper on 23-8-9.
//

#ifndef SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_SIM_H_
#define SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_SIM_H_

#include <Eigen/Core>
#include <glm/glm.hpp>
#include <Eigen/SparseCore>
namespace hairsim {
using Real = double;
using Eigen::VectorXd;
using Vec3d = Eigen::Vector3d;
using Vec4d = Eigen::Vector4d;
using VecXd = VectorXd;
using Mat3d = Eigen::Matrix3d;
using Mat4d = Eigen::Matrix4d;
using MatXd = Eigen::MatrixXd;
using Mat43d = Eigen::Matrix<Real, 4, 3>;
using Eigen::Triplet;
using Eigen::SparseMatrix;
using Index = int;
using std::vector;
}
#endif // SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_SIM_H_
