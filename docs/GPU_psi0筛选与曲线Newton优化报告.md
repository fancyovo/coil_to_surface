# GPU psi0 筛选与曲线 Newton 优化报告

## 1. 本次目标

本次检查的是候选 $\psi_0$ 筛选阶段是否适合继续 GPU 化，尤其是：

1. $\Phi=0$ 截面上提取 $\psi=\psi_0$ 曲线的一维 Newton 是否耗时。
2. 磁力线追踪部分用 `fp32`、`mixed64`、`fp64` 时，筛选结论是否一致。
3. 如果 Newton 耗时，应该迁移到 GPU，还是先做算法优化。

## 2. 原筛选流程

对每个候选 $\psi_0$：

1. 在 $\Phi=0$ 截面取 `n_alpha=256` 个角度 $\theta$。
2. 沿射线求解

$$
\psi(\Phi=0,\rho,\theta)=\psi_0
$$

得到 256 个起点。

3. 从这些起点追踪一个场周期，终点在 $\Phi=2\pi/nfp$。
4. 在终点计算

$$
\frac{|\psi_{\mathrm{end}}-\psi_0|}{|\nabla\psi|}
$$

作为空间偏离量。

5. 用 `p95(distance)` 和相对偏离量判断这个 $\psi_0$ 是否可行。

## 3. 做了什么

### 3.1 新增 benchmark 脚本

新增：

```text
gpu_backend/scripts/bench_psi0_screen_gpu.py
```

它读取已有 `psi_model.npz`，不重新拟合 $\psi$，直接测试候选 $\psi_0$ 筛选阶段。

它能分别记录：

```text
curve Newton 时间
GPU fieldline trace 时间
fp32/mixed64/fp64 筛选结果
fp32/mixed64 相对 fp64 的终点误差
```

### 3.2 测试三种追踪精度

使用全部 10 个默认 level：

```text
0.001, 0.002, 0.004, 0.008, 0.012, 0.02, 0.04, 0.08, 0.12, 0.16
```

每个 level 取 256 条线，总共 2560 条线。

测试精度：

```text
fp32
mixed64
fp64
```

### 3.3 实现并测试曲线 Newton 多项式化

在 [surface.py](../stellarator_eval/surface.py) 中优化了 `level_curve_phi0()`。

原实现每次 Newton 迭代都扫完整的 $\psi$ 模式，大约 1574 项。

现在利用 $\Phi=0$ 且 $\theta$ 固定时，$\psi$ 变成 $\rho$ 的一维多项式：

$$
\psi(\rho,\theta,0)
=
\sum_d c_d(\theta)\left(\frac{\rho}{a}\right)^d
$$

Newton 迭代时只需要评估这个低阶多项式和导数：

$$
\frac{\partial\psi}{\partial\rho}
=
\sum_d c_d(\theta)\frac{d}{a}\left(\frac{\rho}{a}\right)^{d-1}
$$

因此从每轮扫约 1574 个模式，变成预处理扫一次模式，之后每轮只扫多项式阶数。

## 4. 测试结果

### 4.1 曲线 Newton 时间

10 个 level，`n_alpha=256`：

| case | 原 serial Newton | batched Newton | 多项式化 Newton |
|---|---:|---:|---:|
| `01/raw` | 4.877 s | 4.717 s | 0.0146 s |
| `debug/raw` | 4.466 s | 3.681 s | 0.0142 s |

结论：直接 batch 化帮助不大，说明瓶颈不是 Python 外层 level 循环，而是每次 $\psi$ 评估重复扫描全部模式。多项式化后这个步骤已经不值得优先迁移 GPU。

### 4.2 GPU 追踪时间

2560 条线，追踪一个周期：

| case | fp32 | mixed64 | fp64 |
|---|---:|---:|---:|
| `01/raw` | 0.118 s | 0.244 s | 1.153 s |
| `debug/raw` | 0.140 s | 0.266 s | 1.397 s |

对可行性筛选来说，`fp32`、`mixed64`、`fp64` 的 ok/fail 结论在这两个例子上完全一致。

### 4.3 追踪精度误差

终点相对 `fp64` 的空间误差：

| case | fp32 p95 | mixed64 p95 |
|---|---:|---:|
| `01/raw` | `1.40e-06` | `2.10e-08` |
| `debug/raw` | `1.60e-06` | `1.56e-08` |

筛选阈值是 `drift_abs_tol=5e-4`，所以在当前测试中 `fp32` 误差远小于筛选容差。

## 5. 结论

1. 曲线 Newton 原本确实很耗时，但不应该先迁移 GPU。
2. 更合理的优化是利用固定 $\Phi=0$ 后的多项式结构。
3. 多项式化后，10 个 level 的曲线提取只需约 `0.014 s`，已经不是瓶颈。
4. 磁力线追踪部分可继续 GPU 化。
5. 对 $\psi_0$ 可行性筛选，`fp32` 很可能已经足够；但为了保守，默认仍建议用 `mixed64`，最后最大候选可用 `fp64` 复核。

推荐后续实现策略：

```text
curve Newton: CPU 多项式化
psi0 cheap screen trace: GPU mixed64 默认，可选 fp32
最终候选 drift 复核: GPU fp64
```

