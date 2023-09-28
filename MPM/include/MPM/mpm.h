//
// Created by creeper on 23-8-13.
//

#ifndef SIMCRAFT_MPM_INCLUDE_MPM_H_
#define SIMCRAFT_MPM_INCLUDE_MPM_H_
#include <Core/core.h>
#include <Core/data-structures/grids.h>
#include <Core/tensor.h>
#include <Eigen/SparseCore>
namespace mpm {
using core::Real;
using core::Index;
using core::Vec2d;
using core::Vec3d;
using core::Vec4d;
using core::Vec2f;
using core::Vec3f;
using core::Vec4f;
using core::Vector;
using core::vector;
using core::Mat3d;
using core::Matrix;
using core::Grid;
using core::FourthOrderTensor;
using Eigen::Triplet;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
}
#endif // SIMCRAFT_MPM_INCLUDE_MPM_H_
