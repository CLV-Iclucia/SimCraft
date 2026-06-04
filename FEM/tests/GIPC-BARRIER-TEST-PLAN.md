# GIPC Barrier 测试计划

## 测试对象

| 模块 | 头文件 | 公开 API |
|------|--------|----------|
| Barrier 标量函数 | `barrier.h` | `Barrier::energy(dSqr)`, `gradCoeff(I5)`, `lambda0(I5)`, `clampedLambda0(I5)` |
| PFPx 构造器 | `pfpx.h` | `computePFPx_PP/PE/PT/EE/PEE` |
| Hessian 辅助 | `hessian.h` | `sandwichRank1`, `sandwichFull`, `makePD2x2` |
| Mollifier 分支 | `mollifier.h` | `computeMollifiedBarrier`, `computeMollifiedBarrierEnergy`, `needsMollifier` |

## 约定

- `dHat = 0.01`（几何距离阈值 d̂）
- `kappa = 1e5`（刚度系数）
- 所有端到端测试用有限差分（FD）做 ground truth
- FD gradient: 中心差分 `eps = 1e-7`
- FD hessian: 对 gradient 再做中心差分 `eps = 1e-5`
- 不测内部中间量（I1, I2, p1, p2, lambda10 等）
- 不测纯工具函数（makePD2x2, eigenToLocalHessian 等），它们被端到端测试间接覆盖

## 几何配置

| 类型 | 配置 | 距离 | 说明 |
|------|------|------|------|
| PP | `p0=(0,0,0)`, `p1=(0,0.007,0)` | 0.007 | < dHat, y 方向 |
| PE | `p=(0.5,0.006,0)`, `e0=(0,0,0)`, `e1=(1,0,0)` | ~0.006 | 点在边上方 |
| PT | `p=(0.3,0.007,0.3)`, `t0=(0,0,0)`, `t1=(1,0,0)`, `t2=(0,0,1)` | 0.007 | 点在三角形上方 |
| EE | `ea0=(0,0,0)`, `ea1=(1,0,0)`, `eb0=(0.5,0.005,-0.5)`, `eb1=(0.5,0.005,0.5)` | ~0.005 | 垂直交叉 |
| PEE | `ea0=(0,0,0)`, `ea1=(1,0,0)`, `eb0=(0.3,0.005,0.001)`, `eb1=(0.8,0.005,0.001)` | ~0.005 | 近平行，触发 mollifier |

---

## 第一组：Barrier 标量函数

### Test 1: `Barrier_EnergyBoundary`

验证 `Barrier::energy(dSqr)` 的基本行为。

```
energy(d²=d̂²) == 0
energy(d²=1.5*d̂²) == 0
energy(d²=0.5*d̂²) > 0
energy(0.8*d̂²) < energy(0.5*d̂²) < energy(0.1*d̂²)  // 距离越小能量越大
```

### Test 2: `Barrier_GradCoeffMatchesFD`

验证 `gradCoeff(I5)` 与 `energy` 的有限差分导数一致。

```
对 I5 ∈ {0.1, 0.3, 0.5, 0.7, 0.9}:
  energyOfI5(I5) = barrier.energy(I5 * dHatSqr)
  fd = (energyOfI5(I5+eps) - energyOfI5(I5-eps)) / (2*eps)
  // gradCoeff = 2 * dE/dI5，所以 gradCoeff(I5) ≈ 2 * fd
  EXPECT_NEAR(gradCoeff(I5), 2*fd, tol)
```

### Test 3: `Barrier_Lambda0MatchesFD`

验证 `lambda0(I5)` 与 `gradCoeff` 的有限差分导数一致。

```
对 I5 ∈ {0.1, 0.3, 0.5, 0.7, 0.9}:
  // lambda0 = 4*I5*b''(I5) + 2*b'(I5)
  // 用 FD 对 dBdI5 求导得 b''，然后验证
  bp_plus = dBdI5(I5+eps), bp_minus = dBdI5(I5-eps)
  bpp = (bp_plus - bp_minus) / (2*eps)
  expected = 4*I5*bpp + 2*dBdI5(I5)
  EXPECT_NEAR(lambda0(I5), expected, tol)
```

### Test 4: `Barrier_GaussClamp`

验证 `clampedLambda0` 在极小 I5 时数值稳定。

```
对 I5 ∈ {1e-5, 1e-6, 1e-8, 1e-10}:
  lam = clampedLambda0(I5)
  EXPECT_TRUE(isfinite(lam))
  EXPECT_GE(lam, 0)
  // 都应该被 clamp 到同一个阈值处的值
  EXPECT_DOUBLE_EQ(lam, clampedLambda0(GAUSS_THRESHOLD))
```

---

## 第二组：PFPx 正确性

PFPx 是 `∂vec(F)/∂x` 的线性矩阵。验证方式：在给定几何配置下，对 x 做 FD 扰动，
观察 `kappa * barrier.energy(I5(x) * dHatSqr)` 的 gradient 是否和
`kappa * PFPx^T * flatten_pk1` 一致。

这**同时验证**了 PFPx 和 flatten_pk1（= gradCoeff · q0 · √I5）的正确性。

### Test 5: `PP_GradientMatchesFD`

```
energyFunc(x_6d) = kappa * barrier.energy(||p0-p1||²)
pfpx = computePFPx_PP(p0, p1, dHat)
flatten_pk1 = q0 * gradCoeff(I5) * sqrt(I5)
grad_analytical = kappa * PFPx^T * flatten_pk1
grad_fd = finiteDiffGradient(energyFunc, x)
EXPECT_LT(maxErr, tol)
```

### Test 6: `PE_GradientMatchesFD`

同上结构，9 DOF。

### Test 7: `PT_GradientMatchesFD`

同上结构，12 DOF。

### Test 8: `EE_GradientMatchesFD`

同上结构，12 DOF。

---

## 第三组：内层 Hessian 投影正确性

**核心思想**：GIPC 的 Hessian 近似丢掉了 `∂²F/∂x²` 项，只保留了
`PFPx^T · H_inner · PFPx`（sandwich）。所以端到端 FD Hessian 和解析 Hessian
**不会**精确一致（差了 ∂²F/∂x² 项）。

但内层 Hessian（在 vec(F) 空间中）是**精确的**，因为 `b(I5)` 是 vec(F) 的
确定性函数，没有任何近似。所以正确的验证策略是：

1. **构造 vec(F) 空间的能量函数**: `E(vecF) = kappa * barrier.energy(||vecF 的特定分量||² * dHatSqr)`
2. **对 vecF 做 FD 得到真实 Hessian**
3. **对真实 Hessian 做 SPD 投影**（clamp 负特征值到 0）
4. **和我们的解析内层 Hessian 比较**: `lambda0 * q0 * q0^T` 应该等于 SPD 投影后的结果

这样分开验证后，如果端到端 gradient 正确（第二组已验证 PFPx），
加上内层 Hessian 正确（第三组），sandwich 结果自然正确。

### Test 9: `PP_InnerHessianMatchesSPDProjection`

```
// vec(F) 空间能量: E = kappa * barrier.energy(vecF(2)² * dHatSqr)
// 因为 PP NEWF 下 I5 = F₃₃² = vecF(2)²（3 维 vec(F)，q0=e₂）
energyOfVecF(vecF_3d) = kappa * sHat2 * (vecF(2)²-1)² * ln²(vecF(2)²)
                        // 或者直接: kappa * barrier.energy(vecF(2)² * dHatSqr)

vecF0 = PFPx_PP 在当前几何下的 F 值（即 (0, 0, d/d̂) 之类的）
H_fd = finiteDiffHessian(energyOfVecF, vecF0)          // 3×3
H_projected = SPD_projection(H_fd)                      // clamp 负特征值
H_analytical = clampedLambda0(I5) * kappa * q0 * q0^T  // rank-1
EXPECT_NEAR(H_analytical, H_projected, tol)
```

### Test 10: `PE_InnerHessianMatchesSPDProjection`

同上，4 维 vec(F) 空间。

### Test 11: `PT_InnerHessianMatchesSPDProjection`

同上，9 维 vec(F) 空间。

### Test 12: `EE_InnerHessianMatchesSPDProjection`

同上，9 维 vec(F) 空间。

---

## 第四组：端到端 Hessian 属性

即使不和 FD 精确比较，sandwich 出来的 Hessian 必须满足：

### Test 13: `PP_HessianIsSPD`

```
H = sandwichRank1(PFPx, q0, kappa * clampedLambda0(I5))
EXPECT 对称: max|H - H^T| < 1e-12
EXPECT 半正定: min eigenvalue >= -1e-10
```

### Test 14: `PE_HessianIsSPD`
### Test 15: `PT_HessianIsSPD`
### Test 16: `EE_HessianIsSPD`

---

## 第五组：多几何配置鲁棒性

### Test 17: `PP_MultiDirection`

对 6 个不同方向的点对（x/y/z 轴 + 3 个斜方向），分别验证梯度 vs FD 一致。

```
dirs = {(1,0,0), (0,1,0), (0,0,1), normalize(1,1,0), normalize(1,1,1), normalize(-2,3,1)}
对每个 dir:
  p0 = (0.5, 0.5, 0.5)
  p1 = p0 + 0.006 * dir
  验证 gradient vs FD
```

### Test 18: `EE_MultiConfig`

对 3 种不同交叉角的边对，分别验证梯度 vs FD 一致。

```
configs = {
  "perpendicular": ea沿x, eb沿z, y偏移0.005
  "parallel_offset": ea沿x, eb也沿x, y偏移0.004
  "skew_45deg": ea沿x, eb斜45度, y偏移0.006
}
对每个 config:
  验证 gradient vs FD
```

---

## 第六组：Mollifier 分支端到端

### Test 19: `Mollifier_EnergyMatchesFD`

验证 `computeMollifiedBarrierEnergy` 对顶点做 FD 扰动后连续。

```
energyFunc(x_12d):
  从 x 提取 ea0/ea1/eb0/eb1
  dSqr = _d_EE(...)
  return computeMollifiedBarrierEnergy(..., dSqr, barrier, kappa)
grad_fd = finiteDiffGradient(energyFunc, x)
// 只验证 FD gradient 有限且非零（说明能量是光滑的）
```

### Test 20: `Mollifier_GradientMatchesFD`

```
pfpx = computePFPx_PEE(ea0, ea1, eb0, eb1, dHat)
dSqr = _d_EE(...)
result = computeMollifiedBarrier(..., dSqr, pfpx.PFPx, barrier, kappa)
// result.gradient vs FD of computeMollifiedBarrierEnergy
EXPECT_LT(maxErr, 5% * scale)
```

### Test 21: `Mollifier_HessianIsSPD`

```
result = computeMollifiedBarrier(...)
H_12x12 = result.hessian 转 Eigen
EXPECT 对称
EXPECT 半正定
```

### Test 22: `Mollifier_HessianCurvatureAlongGradient`

```
g = result.gradient
H = result.hessian
gHg = g^T * H * g
EXPECT_GT(gHg, 0)  // Newton 方向是下降方向
```

---

## 不测什么

| 不测的东西 | 原因 |
|-----------|------|
| `mollifierP1/P2` | 内部系数，被 Test 20 间接覆盖 |
| `mollifierLambda10/11/20/G1G` | 内部系数，被 Test 21-22 间接覆盖 |
| `computeMollifierInnerHessian` | 内部诊断函数 |
| `makePD2x2` | 工具函数，被 Hessian SPD 测试间接覆盖 |
| `eigenToLocalHessian/Grad` | 格式转换，被端到端测试间接覆盖 |
| 端到端 Hessian vs FD 精确比较 | GIPC 故意丢掉 ∂²F/∂x² 项，不会精确一致 |
| 物理直觉（力的方向） | 过于脆弱，依赖几何配置 |
| 边界连续性 | 由标量函数的数学性质保证 |
