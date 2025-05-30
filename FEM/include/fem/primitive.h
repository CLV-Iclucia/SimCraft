//
// Created by creeper on 10/21/24.
//

#pragma once

#include <variant>
#include <fem/primitives/elastic-tet-mesh.h>
#include <Maths/sparse-matrix-builder.h>

namespace sim::fem {
struct Primitive {
  explicit Primitive(ElasticTetMesh &&data) : impl(std::move(data)) {}
  void init(const SubVector<Real>& x, const SubVector<Real>& xdot, const SubVector<Real>& X) {
    return std::visit([&](auto &&arg) {
      arg.init(x, xdot, X);
    }, impl);
  }
  [[nodiscard]] Real deformationEnergy() const {
    return std::visit([](auto &&arg) {
      return arg.deformationEnergy();
    }, impl);
  }
  void updateDeformationEnergyGradient(const SubVector<Real>& x) {
    return std::visit([&](auto &&arg) {
      arg.updateDeformationEnergyGradient(x);
    }, impl);
  }
  void assembleEnergyGradient(const SubVector<Real>& primitiveGrad) const {
    std::visit([&](auto &&arg) {
      arg.assembleEnergyGradient(primitiveGrad);
    }, impl);
  }
  [[nodiscard]] size_t dofDim() const {
    return std::visit([](auto &&arg) {
      return arg.dofDim();
    }, impl);
  }
  void assembleEnergyHessian(maths::SubMatrixBuilder<Real> &globalHessianSubView) const {
    std::visit([&](auto &&arg) {
      arg.assembleEnergyHessian(globalHessianSubView);
    }, impl);
  }
  void assembleMassMatrix(maths::SubMatrixBuilder<Real> &globalMassSubView) const {
    std::visit([&](auto &&arg) {
      arg.assembleMassMatrix(globalMassSubView);
    }, impl);
  }
  [[nodiscard]] std::span<const Triangle> getSurfaceView() const {
    return std::visit([&](auto &&arg) {
      return arg.getSurfaceView();
    }, impl);
  }
  [[nodiscard]] std::span<const Edge> getEdgesView() const {
    return std::visit([&](auto &&arg) {
      return arg.getEdgesView();
    }, impl);
  }
  
  [[nodiscard]] size_t getVertexCount() const {
    return std::visit([&](auto &&arg) {
      return arg.getVertexCount();
    }, impl);
  }
  
  SubVector<Real> view(VecXd& vec) const {
    return SubVector<Real>(vec.data() + dofStart, dofDim());
  }
  [[nodiscard]] CSubVector<Real> cview(const VecXd& vec) const {
    return CSubVector<Real>(vec.data() + dofStart, dofDim());
  }

  static Primitive static_deserialize(const core::JsonNode& node) {
    static std::unordered_map<std::string, std::function<fem::Primitive(const core::JsonNode &)>> factories {
          {"ElasticTetMesh", [](const core::JsonNode &node) {
            return fem::Primitive(deserialize<fem::ElasticTetMesh>(node));
          }},
        };
    if (!node.is<core::JsonDict>())
      throw std::runtime_error("Deserializing Primitive from non-dict node");
    const auto &dict = node.as<core::JsonDict>();
    if (!dict.contains("type"))
      throw std::runtime_error("Primitive missing type field");
    const auto &type = dict.at("type").as<std::string>();
    if (!factories.contains(type))
      throw std::runtime_error("Unknown primitive type");
    return factories.at(type)(node);
  }

private:
  friend struct System;
  void setDofStart(int start) {
    dofStart = start;
  }
  int dofStart;
  std::variant<ElasticTetMesh> impl;
};

}


