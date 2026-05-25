# SimCraft Python Bindings

## What This Is

将 SimCraft C++ 物理模拟框架的核心 API 暴露为 Python 模块 `simcraft`，使用户可以用 Python 脚本（而非 JSON）配置和运行模拟系统。

## Core Value

**用 Python 的动态逻辑替代静态 JSON 配置** — 用户获得条件分支、循环、变量、函数组合等编程能力来构建模拟场景，同时保持"配置完成后锁定系统"的安全模型。

## Context

SimCraft 是一个 C++20 物理模拟框架，包含 FEM、流体、头发等模块。当前通过 JSON 配置系统参数，灵活性极差。本里程碑为框架增加 Python 接口层。

### 现有架构

- **构建系统**: CMake 3.20+ / vcpkg / MSVC (Windows)
- **核心模块**: Core, Maths, Spatify, Deform, FEM, Renderer
- **类型系统**: GLM (dvec3/dmat3) + 自研 BlockVector<3>/BlockSparseMatrix<3>
- **目标平台**: Windows (开发), 未来可能 Linux

### 技术决策

- **绑定工具**: pybind11
- **Python 版本**: 3.12+
- **构建方式**: 纯 CMake (add_subdirectory)，输出 .pyd
- **模块名**: `simcraft`
- **NumPy 支持**: 是，按需转换（不自动缓存）

## Requirements

### Validated

- ✓ C++20 CMake 构建系统正常工作 — existing
- ✓ FEM/IPC 模拟器功能完整 — existing
- ✓ 约束系统（DOF约束/运动学体）已实现 — existing
- ✓ Block algebra 系统工作正常 — existing

### Active

- [ ] Python 模块 `simcraft` 可 import 并使用
- [ ] TetMesh 支持文件加载和 numpy array 构造
- [ ] NeoHookean 材料模型可从 Python 创建
- [ ] ElasticBody 组合 mesh + material + density
- [ ] System 类支持添加 body、设置重力
- [ ] ConstraintManager 支持 pin_vertices / prescribe_motion
- [ ] KinematicBody 支持平面/网格几何 + 运动规律
- [ ] IpcIntegrator 可配置 dHat/eps/kappa/stepSizeScale
- [ ] Simulation 封装 system + integrator，提供 run/step
- [ ] 配置阶段可自由修改，run() 后锁定不可变
- [ ] 结果读取：positions/velocities/energy 按需转换为 numpy
- [ ] 纯 CMake 构建，pip install 非必需

### Out of Scope

- 实时修改（模拟中改参数）— 未来里程碑
- 废弃/删除 JSON 配置代码 — 本次不动
- scikit-build-core / PyPI 分发 — 等稳定后
- 其他模块 (FluidSim/HairSim) 的 Python 绑定 — 未来
- GUI / 可视化集成 — 不在此范围

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| pybind11 而非 nanobind | 成熟稳定，社区大，文档全 | 确定 |
| 多类组合 API 风格 | 贴近 C++ 结构，灵活可扩展 | 确定 |
| "配置+锁定"模式 | 避免运行时修改的线程安全问题，简化设计 | 确定 |
| 纯 CMake 构建 | 项目已全 CMake，避免引入 scikit-build 复杂度 | 确定 |
| 不动 JSON | 降低本次范围，Python 和 JSON 并存 | 确定 |
| 按需结果转换 | 避免每帧自动拷贝开销 | 确定 |
| 仅 NeoHookean | 第一阶段最小范围 | 确定 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-25 after initialization*
