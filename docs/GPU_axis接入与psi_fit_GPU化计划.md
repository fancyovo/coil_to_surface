# GPU axis 接入与 psi fit GPU 化计划

## 1. 本次目标

本次将 GPU axis search 接入主 pipeline，并在远端完整跑通：

```text
输入线圈 -> simsopt field -> GPU axis search -> axis curve trace
        -> psi fit -> GPU psi0 screen -> Boozer/Newton -> QS error
```

默认 CPU 路径仍保留；GPU axis 需要显式启用：

```bash
--axis-backend gpu
```

## 2. 代码改动

### 2.1 Axis 配置

`AxisGAConfig` 新增：

```python
backend = "cpu" | "gpu"
gpu_lib_path
gpu_segments_per_coil
gpu_device
gpu_trace_precision
gpu_verify_precision
gpu_threads_per_line
staged
```

默认仍是 CPU。

### 2.2 GPU axis search

`axis.py` 新增：

```python
search_axis_ga_gpu(...)
find_axis_gpu(...)
```

行为：

1. 用 `CoilFieldGpu` 构建 GPU 线圈场。
2. 用 `mixed64` 一周期追踪做 GA 找轴。
3. 对最终 best 点用 `fp64` 一周期追踪复核 residual。
4. 磁轴曲线 `phi/R/Z/R_phi/Z_phi` 仍用 simsopt + DOP853 生成，保证后续 psi fit 的轴曲线质量。

### 2.3 Pipeline 接入

`pipeline.py` 中：

```python
if config.axis.backend == "gpu":
    axis = find_axis_gpu(...)
else:
    axis = find_axis(...)
```

`summary.json` 现在记录：

```text
axis.backend
axis.search_time_s
axis.trace_time_s
```

## 3. 远端完整链路结果

运行环境：

```text
server: master
venv: ~/stellarator_gpu_eval/venv
GPU: CUDA_VISIBLE_DEVICES=0
OMP/BLAS threads: 1
simsopt: 1.10.6
```

### 3.1 01/raw

核心参数：

```bash
--axis-backend gpu
--axis-span 0.08
--axis-tol 2e-8
--axis-gpu-precision mixed64
--axis-gpu-verify-precision fp64
--screen-trace-backend gpu
--levels 0.02,0.12
--max-boozer-candidates 1
```

结果：

```text
axis residual: 1.737e-08
axis mixed search residual: 9.352e-09
has_axis: True
best surface psi: 0.12
iota: -2.9076808676
volume: 0.0061502611
G: 7.6311250731
```

QS error：

```text
QA(1,0): 0.00249747
QH(1,1): 0.00241966
QP(0,1): 0.00246122
```

细粒度时间：

| step | time |
|---|---:|
| field build | 0.004 s |
| axis total | 1.022 s |
| axis GPU search | 0.651 s |
| axis curve trace | 0.023 s |
| psi fit | 100.885 s |
| psi0 screen total | 0.718 s |
| psi0 curve Newton | 0.025 s |
| psi0 GPU trace | 0.073 s |
| surface extraction 1D Newton | 0.919 s |
| Boozer LS | 0.089 s |
| Boozer Newton | 0.009 s |
| QS metrics | 0.007 s |
| Boozer candidate total | 1.957 s |

注意：`axis-tol` 使用 `2e-8` 是因为 fp64 复核 residual 为 `1.737e-08`，略高于默认 `1e-8`；mixed64 搜索 residual 本身为 `9.352e-09`。这个差异来自 GPU 离散线圈 Biot-Savart 与精度复核路径，不影响当前闭环，但后续需要统一 residual 判据。

### 3.2 debug/raw

核心参数：

```bash
--axis-backend gpu
--axis-span 0.08
--axis-gpu-precision mixed64
--axis-gpu-verify-precision fp64
--screen-trace-backend gpu
--a 0.02
--levels 0.004,0.012
--max-boozer-candidates 1
```

结果：

```text
axis residual: 9.789e-09
axis mixed search residual: 2.007e-09
has_axis: True
best surface psi: 0.012
iota: -2.1967521732
volume: 8.629094e-05
G: 8.0609167100
```

QS error：

```text
QA(1,0): 5.008e-06
QH(1,1): 1.2499e-02
QP(0,1): 2.930e-06
```

细粒度时间：

| step | time |
|---|---:|
| field build | 0.003 s |
| axis total | 1.208 s |
| axis GPU search | 0.837 s |
| axis curve trace | 0.033 s |
| psi fit | 100.388 s |
| psi0 screen total | 0.796 s |
| psi0 curve Newton | 0.025 s |
| psi0 GPU trace | 0.080 s |
| surface extraction 1D Newton | 0.855 s |
| Boozer LS | 0.210 s |
| Boozer Newton | 0.010 s |
| QS metrics | 0.008 s |
| Boozer candidate total | 1.951 s |

## 4. 当前瓶颈分析

### 4.1 绝对主瓶颈：psi fit

两组结果都显示：

```text
psi_fit_s ≈ 100 s
```

而其它主要阶段已经降到：

```text
axis_s ≈ 1.0-1.2 s
screen_s ≈ 0.7-0.8 s
boozer_candidates_s ≈ 2.0 s
```

所以现在总耗时几乎完全由 `psi fit` 决定。

### 4.2 除 psi fit 外的瓶颈

如果暂时忽略 `psi fit`，剩余较明显的耗时是：

1. Boozer candidate total：约 `1.95 s`
2. axis total：约 `1.0-1.2 s`
3. surface extraction 1D Newton：约 `0.85-0.92 s`
4. psi0 screen total：约 `0.7-0.8 s`

其中：

- Boozer LS/Newton 本身很快，主要不是优化瓶颈。
- surface extraction 仍使用多截面逐点 Newton，后续也可以用类似 $\Phi=0$ 的多项式预处理优化。
- psi0 screen 的实际 GPU trace 只有 `0.07-0.08 s`，总时间中包含 GPU field 创建和 fp64 复核等固定成本。
- axis 搜索已经不是主瓶颈，但可以后续把 axis curve trace 也改成 GPU/固定步长，进一步减少 simsopt `.B()` 小批量调用。

## 5. psi fit GPU 化计划

### 5.1 当前 psi fit 在做什么

`psi.py::fit_psi()` 大致流程：

1. 在磁轴附近生成训练点：

```text
n_r x n_z x n_phi
默认 80 x 80 x 80
保留 rho_min <= rho <= a 的点
```

实际训练点约数十万。

2. 对每个训练点计算磁场：

```python
br, bphi, bz = b_components(field, R, Z, phi)
```

当前走 simsopt `.B()`。

3. 构造线性方程：

$$
\vec B\cdot\nabla\psi = 0
$$

固定项是 $x^2$，未知项是多项式-傅里叶基函数系数。

4. 分 batch 累加正规方程：

$$
A^T A,\quad A^T b
$$

5. 解一个约 `1574 x 1574` 的线性系统。

### 5.2 主要耗时来源

预计有三个大头：

1. simsopt `.B()` 对几十万点的磁场采样。
2. Python 循环逐 mode 构造 `mat`，当前模式数约 `1574`。
3. `mat.T @ mat` 的大矩阵乘法。

从当前 `psi_fit_s ≈ 100 s` 看，单纯替换 `.B()` 可能会大幅改善，但还不一定够；后续还要处理 basis/normal-equation assembly。

### 5.3 阶段 A：只替换 B 采样

目标：最小改动，先验证收益。

做法：

1. 新增 `fit_psi(..., field_sampler=...)` 或 `PsiFitConfig.field_backend`。
2. CPU 路径保持：

```python
b_components(simsopt_field, ...)
```

3. GPU 路径使用：

```python
CoilFieldGpu.eval_B(xyz)
```

然后转换成 `br,bphi,bz`。

4. 仍在 CPU 上构造 basis 和累加 `A^T A`。

预期：

- 实现风险低。
- 能直接量化 `.B()` 在 `psi_fit` 中占多少。
- 如果 `psi_fit` 从 100s 降到几十秒，说明 B 采样是最大头。

### 5.4 阶段 B：basis 构造向量化/预计算

目标：减少 Python mode 循环。

优化点：

1. 对每个 batch 预计算：

```text
X^a Z^b
cos(m nfp phi)
sin(m nfp phi)
axis R_phi, Z_phi
```

2. 用数组广播生成多个 mode 的导数项，减少 Python 层循环。
3. 或按 degree/m 分组，重用 monomial/trig。

这个阶段仍可在 CPU 完成，但应显著降低 basis assembly 时间。

### 5.5 阶段 C：GPU 正规方程装配

目标：把 `A^T A` 和 `A^T b` 累加放到 GPU。

推荐实现路线：

```text
for each point batch:
    1. GPU eval_B
    2. GPU kernel 计算 basis matrix A 和 rhs b
    3. cuBLAS: ATA += A^T A
    4. cuBLAS: ATb += A^T b
transfer ATA/ATb back to CPU
CPU solve 1574x1574 system
```

理由：

- `A` 的 batch 矩阵很大，适合 cuBLAS。
- `ATA` 只有约 `1574 x 1574`，约 20 MB，适合常驻 GPU。
- 初期可以把求解仍放 CPU，避免一开始引入 cuSOLVER 复杂度。

### 5.6 阶段 D：精度策略

建议分三档测试：

```text
fp64: B、basis、ATA 全 double，作为基准
mixed64: B fp32，basis/ATA fp64
fp32 coarse: B、basis、ATA 用 float，用于快速筛选
```

评价指标：

1. train RMS
2. validation angle mean/p95/l2
3. psi0 screen 结果是否一致
4. Boozer/Newton 是否仍能收敛

### 5.7 阶段 E：验证标准

每次 GPU 化都需要和 CPU 版比较：

```text
coeff relative difference
validation_angle_p95
psi0 screen p95
最终 best surface iota/G/QS_error
总时间
```

优先接受的版本：

```text
fit 结果不改变筛选和 Boozer 结论，
总耗时明显下降，
且失败时可回退 CPU fit。
```

## 6. 下一步建议

下一步先做阶段 A：

```text
psi fit 中只替换 B 采样为 GPU eval_B
```

这是风险最低、信息量最大的改造。完成后再决定是否继续做 basis/ATA 的 GPU 化。

