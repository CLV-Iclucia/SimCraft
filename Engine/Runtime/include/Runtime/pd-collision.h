#pragma once

// Projective Dynamics collision handling (post-processing).
//
// PD does NOT participate in IPC contact mechanics. Collision is resolved as a
// separate post-processing phase after the PD local/global iterations:
//
//   Stage 1: Detect    - DCD on the current geometry (reuses IPC broad/narrow
//                        phase distance functions, no CCD / barrier).
//   Stage 2: Interpret - classify penetration domains from mesh connectivity.
//   Stage 3: Solve     - apply boundary displacements + propagate to interior.
//
// The three stages are decoupled: each consumes the previous stage's output
// and produces its own. Stages can be tested and replaced independently.
//
// PD v1 limitations (accepted): no CCD, pure position corrections (no momentum
// conservation), no friction, rigid-rigid handled via simple projection or
// skipped, and domain classification relies on the pinned-vertex set.

#include <Runtime/contact-detection.h>
#include <Runtime/global-geometry-manager.h>
#include <Runtime/simulation-context.h>
#include <Runtime/types.h>

#include <array>
#include <glm/glm.hpp>
#include <span>
#include <unordered_set>
#include <vector>

namespace ksk::runtime {

// A single penetrating primitive pair resolved by the PD post-processing
// collision stage. Points are global geometry point indices.
struct PdPenetrationConstraint {
  EContactCandidate kind = EContactCandidate::PointTriangle;
  std::array<PointIdx, 4> points{};
  int pointCount = 0;  // 2/3/4 depending on degeneracy

  // World-space closest points.
  glm::dvec3 closestA{0.0};
  glm::dvec3 closestB{0.0};

  // Penetration info.
  double penetrationDepth = 0.0;  // > 0 when penetrating
  glm::dvec3 normal{0.0};         // A -> B

  // Barycentric coordinates of the closest points on each primitive.
  double baryA[2] = {0.0, 0.0};  // primitive A
  double baryB[2] = {0.0, 0.0};  // primitive B
};

// Stage 1 output: raw penetration information.
struct RawPenetrations {
  std::vector<PdPenetrationConstraint> constraints;
  std::unordered_set<PointIdx> penetratingVertices;     // V_pen
  std::unordered_set<TriangleIdx> penetratingTriangles; // T_pen

  [[nodiscard]] bool empty() const noexcept
  {
    return penetratingTriangles.empty();
  }
};

// Stage 2 output: a penetration domain (a connected component of the mesh
// graph whose vertices need positional corrections).
struct PenetrationDomain {
  enum class Type {
    // No interior propagation; only the directly penetrating vertices move.
    Shallow,
    // A closed component surrounded by the penetration boundary; interior
    // vertices receive propagated displacements.
    PenetrationDomain,
  };

  Type type = Type::Shallow;
  std::vector<PointIdx> allVertices;
  // Boundary vertices -> target displacement (from DCD push).
  std::vector<std::pair<PointIdx, glm::dvec3>> boundaryDisplacements;
  // Interior vertices (displacement computed by Stage 3 propagation).
  std::vector<PointIdx> interiorVertices;
  // Mesh subgraph (vertex + edge local indices) used by Stage 3.
  std::vector<std::pair<int, int>> edges;
};

// ---------------------------------------------------------------------------
// Stage 1: Detect -- raw penetration detection at the current geometry.
//
// Reuses the IPC narrow-phase squared-distance functions. A pure Euclidean
// distance below the summed primitive thickness counts as penetration (no
// signed-normal test -- required for two-dimensional manifolds such as cloth).
// The broad phase currently performs a simple pairwise scan; a BVH broad phase
// is a drop-in performance improvement and does not change the output contract.
// ---------------------------------------------------------------------------
[[nodiscard]] RawPenetrations detectPenetrations(
    const GlobalGeometryManager& geometry);

// ---------------------------------------------------------------------------
// Stage 2: Interpret -- identify penetration domains from connectivity.
//
// meshEdges are global mesh edges (G = (V, E)); pinnedVertices (V_pin) are
// excluded from correction; totalVertexCount bounds vertex indices.
// ---------------------------------------------------------------------------
[[nodiscard]] std::vector<PenetrationDomain> identifyPenetrationDomains(
    const RawPenetrations& raw,
    std::span<const GeometryEdge> meshEdges,
    const std::unordered_set<PointIdx>& pinnedVertices,
    int totalVertexCount);

// ---------------------------------------------------------------------------
// Stage 3: Solve -- apply boundary displacements and propagate to interior
// vertices with BFS layer averaging (per design: disp = avg(neighbor disp)).
// ---------------------------------------------------------------------------
void resolvePenetrationDomains(
    DofBuffer& q,
    const std::vector<PenetrationDomain>& domains,
    std::span<const GeometryEdge> meshEdges);

// Full post-processing loop: detect -> interpret -> resolve -> resync geometry.
void resolvePdCollisions(SimulationContext& simulation,
                         const std::unordered_set<PointIdx>& pinnedVertices,
                         int maxResolveIterations);

}  // namespace ksk::runtime
