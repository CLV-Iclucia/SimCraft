#include <fem/constraints.h>
#include <spdlog/spdlog.h>

namespace sim::fem {

void ConstraintManager::pinVertices(const std::vector<int>& indices,
                                    const maths::BlockVector<3>& positions) {
  for (int idx : indices) {
    if (idx < 0 || idx >= positions.numBlocks()) {
      spdlog::warn("pinVertices: invalid vertex index {}", idx);
      continue;
    }
    VertexConstraint c;
    c.globalBlockIdx = idx;
    c.mask = {true, true, true};
    c.data = FixedConstraint{positions[idx]};
    m_constraints.push_back(c);
  }
}

void ConstraintManager::pinComponent(int vertexIdx, int component, Real value) {
  if (component < 0 || component > 2) {
    spdlog::warn("pinComponent: invalid component {}", component);
    return;
  }
  
  // 检查是否已有该顶点的约束，有则更新 mask；无则新建
  for (auto& c : m_constraints) {
    if (c.globalBlockIdx == vertexIdx) {
      c.mask[component] = true;
      // 更新 target 对应分量
      if (auto* fc = std::get_if<FixedConstraint>(&c.data))
        fc->targetPosition[component] = value;
      return;
    }
  }
  // 新建
  VertexConstraint c;
  c.globalBlockIdx = vertexIdx;
  c.mask = {false, false, false};
  c.mask[component] = true;
  auto fc = FixedConstraint{};
  fc.targetPosition[component] = value;
  c.data = fc;
  m_constraints.push_back(c);
}

void ConstraintManager::prescribeMotion(int vertexIdx, 
                                       std::function<glm::dvec3(Real)> positionFunc,
                                       std::optional<std::function<glm::dvec3(Real)>> velocityFunc) {
  VertexConstraint c;
  c.globalBlockIdx = vertexIdx;
  c.mask = {true, true, true};
  TimeVaryingConstraint tvc;
  tvc.positionFunc = std::move(positionFunc);
  if (velocityFunc.has_value()) {
    tvc.velocityFunc = std::move(velocityFunc.value());
  }
  c.data = tvc;
  m_constraints.push_back(c);
}

void ConstraintManager::prescribeVelocity(int vertexIdx, std::function<glm::dvec3(Real)> velocity) {
  VertexConstraint c;
  c.globalBlockIdx = vertexIdx;
  c.mask = {true, true, true};
  c.data = VelocityConstraint{std::move(velocity)};
  m_constraints.push_back(c);
}

void ConstraintManager::build(int totalBlocks) {
  m_constraintMask.assign(totalBlocks, {false, false, false});
  for (const auto& c : m_constraints) {
    // 手动合并 mask (glm::bvec3 没有 operator|)
    auto& mask = m_constraintMask[c.globalBlockIdx];
    mask.x = mask.x || c.mask.x;
    mask.y = mask.y || c.mask.y;
    mask.z = mask.z || c.mask.z;
  }
  
  m_freeBlocks.clear();
  m_constrainedBlocks.clear();
  for (int i = 0; i < totalBlocks; i++) {
    auto m = m_constraintMask[i];
    bool anyConstrained = m.x || m.y || m.z;
    bool anyFree = !m.x || !m.y || !m.z;
    
    if (anyConstrained) {
      m_constrainedBlocks.push_back(i);
    }
    if (anyFree) {
      m_freeBlocks.push_back(i);
    }
  }
}

void ConstraintManager::enforcePosition(maths::BlockVector<3>& x, Real t) const {
  for (const auto& c : m_constraints) {
    if (!c.isPositionConstraint()) continue;
    auto target = c.targetAt(t);
    if (c.globalBlockIdx < 0 || c.globalBlockIdx >= x.numBlocks()) continue;
    auto& block = x[c.globalBlockIdx];
    if (c.mask.x) block.x = target.x;
    if (c.mask.y) block.y = target.y;
    if (c.mask.z) block.z = target.z;
  }
}

void ConstraintManager::enforceVelocity(maths::BlockVector<3>& xdot, Real t) const {
  for (const auto& c : m_constraints) {
    if (c.globalBlockIdx < 0 || c.globalBlockIdx >= xdot.numBlocks()) continue;
    auto& block = xdot[c.globalBlockIdx];
    
    if (c.isVelocityConstraint()) {
      auto v = c.targetVelocityAt(t);
      if (c.mask.x) block.x = v.x;
      if (c.mask.y) block.y = v.y;
      if (c.mask.z) block.z = v.z;
    } else if (c.isPositionConstraint()) {
      // 位置约束的顶点速度由约束类型决定
      if (std::holds_alternative<FixedConstraint>(c.data)) {
        // Fixed → 速度 = 0
        if (c.mask.x) block.x = 0.0;
        if (c.mask.y) block.y = 0.0;
        if (c.mask.z) block.z = 0.0;
      } else if (std::holds_alternative<TimeVaryingConstraint>(c.data)) {
        // TimeVarying → 使用提供的速度函数或数值差分
        auto v = c.targetVelocityAt(t);
        if (c.mask.x) block.x = v.x;
        if (c.mask.y) block.y = v.y;
        if (c.mask.z) block.z = v.z;
      }
    }
  }
}

void ConstraintManager::projectToFreeSpace(maths::BlockVector<3>& v) const {
  for (const auto& c : m_constraints) {
    if (c.globalBlockIdx < 0 || c.globalBlockIdx >= v.numBlocks()) continue;
    auto& block = v[c.globalBlockIdx];
    if (c.mask.x) block.x = 0.0;
    if (c.mask.y) block.y = 0.0;
    if (c.mask.z) block.z = 0.0;
  }
}

} // namespace sim::fem
