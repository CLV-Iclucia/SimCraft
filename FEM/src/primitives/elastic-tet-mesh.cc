//
// Created by creeper on 10/31/24.
//
#include <Maths/tensor.h>
#include <fem/primitives/elastic-tet-mesh.h>
namespace sim::fem {
using maths::vectorize;

void ElasticTetMesh::init(const SubVector<Real>& x, const SubVector<Real>& xdot, const SubVector<Real>& X) {
  tetRefVolumes.reserve(mesh.tets.size());
  tetDeformationGradients.reserve(mesh.tets.size());
  for (int i = 0; i < numTets(); i++) {
    auto &tet = mesh.tets[i];
    Matrix<Real, 3, 3> local_x;
    for (int j = 0; j < 3; j++)
      local_x.col(j) = mesh.getVertices()[tet[j + 1]] - mesh.getVertices()[tet[0]];
    Real volume = local_x.determinant() / 6.0;
    tetRefVolumes.emplace_back(volume);
    tetDeformationGradients.emplace_back(local_x);
    tetDeformationGradients.back().updateCurrentConfig(local_x);
  }
  mesh.commit(x, xdot, X);
}

void ElasticTetMesh::updateDeformationEnergyGradient(SubVector<Real> x) {
  for (int i = 0; i < numTets(); i++) {
    auto &dg = tetDeformationGradients[i];
    Matrix<Real, 3, 3> local_x;
    for (int j = 0; j < 3; j++)
      local_x.col(j) =
          x.segment<3>(mesh.tets[i][j + 1]) - x.segment<3>(mesh.tets[i][0]);
    dg.updateCurrentConfig(local_x);
  }
}
void ElasticTetMesh::assembleEnergyGradient(
    const SubVector<Real> &primitiveGradSubView) const {
  for (int i = 0; i < numTets(); i++) {
    auto &dg = tetDeformationGradients[i];
    auto grad = energy->computeEnergyGradient(dg);
    auto p_F_p_x = dg.gradient();
    Vector<Real, 12> grad_x =
        p_F_p_x.transpose() * vectorize(grad) * tetRefVolumes[i];
    tetAssembleGlobal(primitiveGradSubView, grad_x, mesh.tets[i]);
  }
}

Real ElasticTetMesh::deformationEnergy() const {
  Real sum = 0.0;
  for (int i = 0; i < numTets(); i++) {
    auto &dg = tetDeformationGradients[i];
    sum += energy->computeEnergyDensity(dg) * tetRefVolumes[i];
  }
  return sum;
}

void ElasticTetMesh::assembleEnergyHessian(
    maths::SubMatrixBuilder<Real> &globalHessianSubView) const {
  for (int i = 0; i < numTets(); i++) {
    auto &dg = tetDeformationGradients[i];
    auto hessian_F = energy->filteredEnergyHessian(dg);
    auto p_F_p_x = dg.gradient();
    auto hessian_x =
        p_F_p_x.transpose() * hessian_F * p_F_p_x * tetRefVolumes[i];
    std::array<int, 4> tetArray = {mesh.tets[i][0], mesh.tets[i][1],
                     mesh.tets[i][2], mesh.tets[i][3]};
    globalHessianSubView.assembleBlock<12, 4>(hessian_x, tetArray);
  }
}

void ElasticTetMesh::assembleMassMatrix(
    maths::SubMatrixBuilder<Real> &globalMassSubView) const {
  for (auto i = 0; i < numTets(); i++) {
    Real volume = tetRefVolumes[i];
    Matrix<Real, 12, 12> localMass;
    localMass.setZero();
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++)
        localMass.block<3, 3>(j * 3, k * 3) += density * volume *
                                               (j == k ? 0.1 : 0.05) *
                                               Matrix<Real, 3, 3>::Identity();
    std::array<int, 4> tetArray = {mesh.tets[i][0], mesh.tets[i][1],
                                   mesh.tets[i][2], mesh.tets[i][3]};
    globalMassSubView.assembleBlock<12, 4>(localMass, tetArray);
  }
}

ElasticTetMesh ElasticTetMesh::static_deserialize(const core::JsonNode &json) {
  if (!json.is<core::JsonDict>())
    throw std::runtime_error("Expected a dictionary for ElasticTetMesh");
  const auto &dict = json.as<core::JsonDict>();
  
  ElasticTetMesh result;
  
  if (dict.contains("mesh"))
    result.setMesh(TetMesh::static_deserialize(dict.at("mesh")));
  else
    throw std::runtime_error("ElasticTetMesh missing mesh field");

  if (dict.contains("density")) {
    result.density = dict.at("density").as<Real>();
  } else {
    throw std::runtime_error("ElasticTetMesh missing density field");
  }
  
  if (dict.contains("energy"))
    result.energy = deform::createStrainEnergyDensity<Real>(dict.at("energy"));
  else
    throw std::runtime_error("ElasticTetMesh missing energy field");

  return result;
}

} // namespace fem