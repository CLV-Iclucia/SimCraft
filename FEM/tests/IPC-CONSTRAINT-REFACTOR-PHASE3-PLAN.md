# Phase 3: Barrier 层统一索引规范与重构

## 3.1 当前状态

已完成：
- Phase 1: 基础数据结构 (`Collision Pair`, `Constraint Pair`) ✓
- Phase 2: Activation Pass 实现 (`Collision Pair -> ConstraintPair` 筛选生成) ✓
- **Phase 2 关键改进**: 统一索引布局规范，不再丢失距离类型信息 ✓

现状：`ConstraintPair` 中的 `indices` 已按统一规范填充：
- **PP**: `indices[0..1]` = 两个点索引
- **PE**: `indices[0..2]` = 点 + 边两端点
- **PT**: `indices[0..3]` = 点 + 三角形三个顶点
- **EE**: `indices[0..3]` = 两条边各两个端点

## 3.2 Phase 3 的核心任务

让 barrier 层（energy/gradient/hessian）**只看 `ConstraintPair` 的统一索引**，不再需要原始的 distance type。

### 原问题（用户提出）

```
@integrator.cc:440-456

这里做得不好，比如 PP 约束，执行层怎么知道是 PA 还是 PB 还是 PC。
所以应该统一规范：PP 约束下，前两个 index 是点的 index，...
这样我们就不必存的更具体了
```

### 解决方案

已经完成的工作：
1. ✓ 更新 `constraint.h` 中的文档，明确规范索引布局
2. ✓ 更新 `collision-pair.cc` 中的 `appendConstraintPair()` 方法，按规范填充索引
3. ✓ 更新 `integrator.cc` 的第二遍循环，按规范构建 `ConstraintPair`

现在需要做的工作（Phase 3）：
- [ ] 重构 barrier 层，使其只依赖 `ConstraintPair.type + indices`
- [ ] 移除旧的 `ConstraintSet` 在 barrier 计算中的使用
- [ ] 验证 barrier energy/gradient/hessian 的正确性

## 3.3 Barrier 层重构的关键点

### 当前状况

`constraint.cc` 中的 barrier 计算混合了两种方式：
1. 直接使用旧的 `ConstraintSet.vtConstraints / eeConstraints` 中的具体类型
2. 每个约束方法 (PP/PE/PT/EE) 都是分开实现的

### 目标状况

- 所有 barrier 计算从 `ConstraintPairSet.pairs` 中取 `ConstraintPair`
- 根据 `ConstraintPair.type` 分发到对应的 barrier 计算函数
- 函数参数只需 `ConstraintPair + 几何引用`，不需具体的 VT/EE/Collider 类型

### 实施步骤

#### 第一步：新增统一的 barrier 接口

在 `constraint.h` 中添加针对 `ConstraintPair` 的 barrier 方法：

```cpp
// 针对普通 ConstraintPair
namespace sim::fem::ipc {

Real constraintPairBarrierEnergy(
    const ConstraintPair& pair,
    const std::vector<glm::dvec3>& x,
    const std::vector<glm::dvec3>& X,  // 仅 EE 需要
    const Barrier& barrier,
    Real kappa);

void constraintPairBarrierGradient(
    const ConstraintPair& pair,
    const std::vector<glm::dvec3>& x,
    const std::vector<glm::dvec3>& X,
    maths::BlockVector<3>& globalGradient,
    const Barrier& barrier,
    Real kappa);

// Hessian 类似...
}
```

#### 第二步：实现 ConstraintPair 的 barrier 函数

根据 `pair.type` 分发：
- `PP`: 调用 2-DOF 的 barrier 计算
- `PE`: 调用 3-DOF 的 barrier 计算
- `PT`: 调用 4-DOF 的 barrier 计算
- `EE`: 调用 4-DOF mollifier 路径

#### 第三步：在 integrator 中更新 barrier 装配

从遍历 `ConstraintSet.vtConstraints / eeConstraints` 改为遍历 `ConstraintPairSet.pairs`，按类型区间执行。

#### 第四步：移除旧的 ConstraintSet

在 Phase 3 完成后，可逐步移除：
- `vtConstraints`
- `eeConstraints`
- `kinematicVTConstraints`

## 3.4 预期代码改变量

### 文件修改

1. **constraint.h**
   - 添加新的统一 barrier 函数声明
   - 保留旧接口的向后兼容（可选，用于过渡）

2. **constraint.cc**
   - 实现统一的 barrier energy/gradient/hessian 函数
   - 在函数内按 `ConstraintPair.type` 分发

3. **integrator.cc**
   - 修改 `barrierEnergy()` 以遍历 `constraintPairs.pairs`
   - 修改 `assembleBarrierGradient()` 以遍历 `constraintPairs.pairs`
   - 修改 `spdProjectHessian()` 以遍历 `constraintPairs.pairs`

## 3.5 关键验证点

- [ ] 统一索引布局能否完整恢复所有距离类型信息
- [ ] barrier energy 数值是否保持一致
- [ ] gradient FD 验证
- [ ] Hessian SPD 投影是否正确
- [ ] EE 来源约束的 mollifier 路径是否能正确激活

## 3.6 测试计划

### 数值回归测试

验证：
1. 单个 PP/PE/PT/EE 约束的 energy 与之前一致
2. gradient 与 FD 偏差 < 1e-6
3. Hessian 经 SPD 投影后特征值非负

### 集成测试

验证：
1. 复杂场景下的 barrier energy 总和
2. 梯度装配正确
3. Hessian 稀疏性正确

## 3.7 里程碑

### 已完成

- [x] 定义统一索引规范（PP/PE/PT/EE）
- [x] 更新 `collision-pair.cc` 的 `appendConstraintPair()` 按规范填充
- [x] 在 `constraint.h` 中声明统一 barrier 接口
- [x] 实现 `constraintPairBarrierEnergy()` 
- [x] 实现 `constraintPairBarrierGradient()`
- [x] 实现 `constraintPairBarrierHessian()`
- [x] 实现 Collider 约束的统一 barrier 接口
- [x] 创建 Phase 3 测试文件 `test-phase3-unified-barrier.cc`

### 待完成

- [ ] 编译验证所有测试通过
- [ ] integrator 中集成新接口
- [ ] 数值回归测试（energy/gradient/Hessian）
- [ ] 移除旧的 ConstraintSet barrier 调用（迁移到新接口）

## 3.8 下一步工作

在 integrator.cc 中：

1. 在 `barrierEnergy()` 中：
   - 遍历 `constraintPairs.pairs` 按类型区间
   - 调用 `constraintPairBarrierEnergy()` 或旧接口求和
   
2. 在 `assembleBarrierGradient()` 中：
   - 遍历 `constraintPairs.pairs` 按类型区间
   - 调用 `constraintPairBarrierGradient()`

3. 在 `spdProjectHessian()` 中：
   - 遍历 `constraintPairs.pairs` 按类型区间
   - 调用 `constraintPairBarrierHessian()`

这会完全解耦 barrier 层对旧 `ConstraintSet` 的依赖。
