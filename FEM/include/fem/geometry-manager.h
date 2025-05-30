#pragma once

#include <vector>
#include <unordered_map>
#include <fem/types.h>
#include <fem/primitive.h>
#include <fem/colliders.h>
#include <fem/simplex.h>

namespace sim::fem {

struct GeometryReference {
  int primitiveId;
  int localIndex;
};

class GeometryManager {
public:
  GeometryManager() = default;

  void collectGeometryReferences(const std::vector<Primitive>& primitives, const std::vector<Collider>& colliders);
  
  [[nodiscard]] int triangleCount() const { return static_cast<int>(triangleRefs.size()); }
  [[nodiscard]] int edgeCount() const { return static_cast<int>(edgeRefs.size()); }
  [[nodiscard]] int vertexCount() const { return static_cast<int>(vertexRefs.size()); }

  [[nodiscard]] Triangle getGlobalTriangle(int globalIndex) const;
  
  [[nodiscard]] Edge getGlobalEdge(int globalIndex) const;
  
  [[nodiscard]] bool triangleContainsVertex(int triangleIdx, int vertexIdx) const;
  
  [[nodiscard]] bool checkEdgeAdjacent(int edgeA, int edgeB) const;
  
  [[nodiscard]] int getVertexPrimitiveId(int globalVertexIdx) const {
    return vertexRefs[globalVertexIdx].primitiveId;
  }
  
  [[nodiscard]] int getVertexLocalIndex(int globalVertexIdx) const {
    return vertexRefs[globalVertexIdx].localIndex;
  }
  
  [[nodiscard]] GeometryReference globalToLocalVertex(int globalVertexIdx) const {
    return vertexRefs[globalVertexIdx];
  }
  
  [[nodiscard]] int localToGlobalVertex(int primitiveId, int localVertexIdx) const;
  
  class TriangleAccessor {
  public:
    using CoordType = Real;
    
    TriangleAccessor(const GeometryManager& manager, const VecXd& positions) 
      : manager(manager), positions(positions) {}
    
    [[nodiscard]] BBox<Real, 3> bbox(int idx) const;
    [[nodiscard]] int size() const { return manager.triangleCount(); }
    
  private:
    const GeometryManager& manager;
    const VecXd& positions;
  };
  
  class EdgeAccessor {
  public:
    using CoordType = Real;
    
    EdgeAccessor(const GeometryManager& manager, const VecXd& positions) 
      : manager(manager), positions(positions) {}
    
    [[nodiscard]] BBox<Real, 3> bbox(int idx) const;
    [[nodiscard]] int size() const { return manager.edgeCount(); }
    
  private:
    const GeometryManager& manager;
    const VecXd& positions;
  };
  
  class VertexAccessor {
  public:
    using CoordType = Real;
    
    VertexAccessor(const GeometryManager& manager, const VecXd& positions)
      : manager(manager), positions(positions) {}
    
    [[nodiscard]] BBox<Real, 3> bbox(int idx) const;
    [[nodiscard]] int size() const { return manager.vertexCount(); }
    
  private:
    const GeometryManager& manager;
    const VecXd& positions;
  };
  
  class TrajectoryAccessor {
  public:
    using CoordType = Real;
    
    TrajectoryAccessor(const GeometryManager& manager, const VecXd& positions, 
                      const VecXd& directions, Real toi) 
      : manager(manager), positions(positions), 
        directions(directions), toi(toi) {}
    
    [[nodiscard]] BBox<Real, 3> triangleBBox(int idx) const;
    [[nodiscard]] BBox<Real, 3> edgeBBox(int idx) const;
    [[nodiscard]] BBox<Real, 3> vertexBBox(int idx) const;
    
    [[nodiscard]] int triangleSize() const { return manager.triangleCount(); }
    [[nodiscard]] int edgeSize() const { return manager.edgeCount(); }
    [[nodiscard]] int vertexSize() const { return manager.vertexCount(); }
    
  private:
    const GeometryManager& manager;
    const VecXd& positions;
    const VecXd& directions;
    Real toi{1.0};
  };
  
  [[nodiscard]] TriangleAccessor getTriangleAccessor(const VecXd& positions) const { 
    return TriangleAccessor(*this, positions); 
  }
  
  [[nodiscard]] EdgeAccessor getEdgeAccessor(const VecXd& positions) const { 
    return EdgeAccessor(*this, positions); 
  }
  
  [[nodiscard]] VertexAccessor getVertexAccessor(const VecXd& positions) const {
    return VertexAccessor(*this, positions);
  }
  
  [[nodiscard]] TrajectoryAccessor getTrajectoryAccessor(const VecXd& positions, 
                                                        const VecXd& directions, 
                                                        Real toi) const {
    return TrajectoryAccessor(*this, positions, directions, toi);
  }
  
  [[nodiscard]] GeometryReference getTriangleRef(int globalIndex) const {
    return triangleRefs[globalIndex];
  }
  
  [[nodiscard]] GeometryReference getEdgeRef(int globalIndex) const {
    return edgeRefs[globalIndex];
  }
  
  [[nodiscard]] GeometryReference getVertexRef(int globalIndex) const {
    return vertexRefs[globalIndex];
  }
  
  [[nodiscard]] Triangle getTriangle(int globalIndex) const;
  [[nodiscard]] Edge getEdge(int globalIndex) const;

private:
  std::vector<GeometryReference> triangleRefs{};
  std::vector<GeometryReference> edgeRefs{};
  std::vector<GeometryReference> vertexRefs{};
  
  std::vector<std::pair<int, int>> primitiveTriangleRanges{};
  std::vector<std::pair<int, int>> primitiveEdgeRanges{};
  std::vector<std::pair<int, int>> primitiveVertexRanges{};
  
  const std::vector<Primitive>* primitives = nullptr;
};

} // namespace fem 