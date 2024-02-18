//
// Created by creeper on 23-8-9.
//

#ifndef SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_SYSTEM_H_
#define SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_SYSTEM_H_
#include <HairSim/band-matrix.h>
#include <HairSim/hair-dof.h>
#include <HairSim/hair.h>
#include <Core/properties.h>

#include <utility>

namespace hairsim {
struct VertexIdentifier {
  Index hairIndex;
  Index vertexIndex;
};

struct SystemConfig {
  Index nHairs = 0;
  Index nVerticesPerHair = 0;
  std::vector<VertexIdentifier> controlVertices{};

  [[nodiscard]] Index numVertices() const { return nHairs * nVerticesPerHair; }
};

class System : core::NonCopyable {
  public:
    explicit System(SystemConfig config_,
                    HairParams params_) : config(std::move(config_)),
      params(std::move(params_)), q(config.numVertices()), qdot(config.numVertices()),
      rhs(config.numVertices()), matrix(config.numVertices()) {
    }
    [[nodiscard]] Index numHairs() const {
      return config.nHairs;
    }
    [[nodiscard]] Index numVerticesPerHair() const {
      return config.nVerticesPerHair;
    }
    [[nodiscard]] Index numVertices() const {
      return config.nHairs * config.nVerticesPerHair;
    }
    [[nodiscard]] Index dofIndex(Index hairIndex, Index vertexIndex) const {
      return hairIndex * config.nVerticesPerHair + vertexIndex;
    }
    // VertexIdentifier is trivially copyable
    [[nodiscard]] Index dofIndex(VertexIdentifier id) const {
      return dofIndex(id.hairIndex, id.vertexIndex);
    }

  private:
    SystemConfig config{};
    HairParams params{};
    HairDof q{};
    HairDof qdot{};
    HairDof rhs{};
    BandSquareMatrix<Real, 5> matrix;
};
}
#endif // SIMCRAFT_HAIRSIM_INCLUDE_HAIRSIM_HAIR_SYSTEM_H_
