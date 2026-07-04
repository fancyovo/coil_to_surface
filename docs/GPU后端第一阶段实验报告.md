# GPU 后端第一阶段实验报告

## 1. 本阶段完成内容

本阶段目标是验证 C++/CUDA 后端替换 `.B()` 和固定步长 RK4 的可行性。已经完成：

1. 新增 CUDA 后端目录：

```text
gpu_backend/
```

2. 实现 C ABI 共享库：

```text
libstellarator_gpu.so
```

3. 实现 Python `ctypes` 包装：

```text
gpu_backend/python/stellarator_gpu.py
```

4. 实现两个测试脚本：

```text
gpu_backend/scripts/bench_gpu_backend.py
gpu_backend/scripts/gpu_axis_ga.py
```

5. 在服务器 `202.38.89.83` 上编译并测试：

```text
~/stellarator_gpu_eval/local_surface_evaluator/gpu_backend
```

编译环境：

```text
CUDA 12.8
GCC 11.5
CMake 3.26
GPU: NVIDIA GeForce RTX 5090
CUDA_VISIBLE_DEVICES=0
```

本阶段没有改动主 Python 评估器的算法路径，只做独立 GPU backend 原型和 benchmark。

## 2. CUDA 后端设计

### 2.1 线圈离散

输入仍然是线圈 Fourier 系数、电流和 `nfp`。每个坐标的 `CurveXYZFourier` 系数顺序已经确认是：

```text
[常数, sin(1), cos(1), sin(2), cos(2), ...]
```

每根基础线圈被离散为小电流元。每个电流元存储：

```text
segment position: x, y, z
current weighted dl: I dl_x, I dl_y, I dl_z
```

这样 GPU 求和时不需要再乘电流。

对称线圈生成方式和 `coils_via_symmetries(..., stellsym=True)` 对齐：

1. 对基础线圈做 `nfp` 次绕 $z$ 轴旋转；
2. 对 stellarator symmetry 线圈做：

$$
(x,y,z)\rightarrow(x,-y,-z),
$$

并翻转电流符号。

### 2.2 数据布局

GPU 侧采用 SoA：

```cpp
double* seg_x;
double* seg_y;
double* seg_z;
double* seg_wx;
double* seg_wy;
double* seg_wz;
```

这样 warp 内 thread 访问连续 segment index 时更容易合并访存。

### 2.3 tile shared memory

没有把全部电流元一次性放进 shared memory，而是采用 tile：

```text
SEG_TILE = 256
```

每个 tile 放：

```text
256 * 6 * sizeof(double) = 12 KiB
```

这比一次性缓存全部电流元稳健得多。对于当前 case：

| case | `segments_per_coil=256` 时总电流元 |
|---|---:|
| `01/raw` | 4096 |
| `debug/raw` | 5120 |

如果全部放 shared memory，分别需要约 `192 KiB` 和 `240 KiB`，不适合作为默认设计。

### 2.4 warp-per-line / warp-per-point

当前实现：

```text
WARPS_PER_BLOCK = 8
THREADS_PER_BLOCK = 256
```

`eval_B`：

- 一个 warp 负责一个空间点；
- 32 个 thread 分摊电流元；
- warp 内规约得到总 $B$。

`trace_period`：

- 一个 warp 负责一条磁力线；
- 一个 block 负责 8 条磁力线；
- 一个 kernel launch 完成整批线的一周期 RK4；
- 每个 RK4 step 调用 4 次 $B$。

磁力线方程：

$$
\frac{dR}{d\Phi}=\frac{R B_R}{B_\Phi},\qquad
\frac{dZ}{d\Phi}=\frac{R B_Z}{B_\Phi}.
$$

## 3. 正确性验证

### 3.1 GPU vs CPU segment reference

`bench_gpu_backend.py` 会用同一组离散电流元分别在 CPU 和 GPU 上算 $B$，验证 CUDA 求和是否正确。

结果：

| case | `segments_per_coil` | GPU vs CPU segment mean rel | p95 rel | max rel |
|---|---:|---:|---:|---:|
| `01/raw` | 256 | 2.47e-15 | 5.08e-15 | 7.77e-15 |
| `debug/raw` | 256 | 2.39e-15 | 4.95e-15 | 7.10e-15 |

这说明 CUDA kernel 的 Biot-Savart 求和与 CPU segment 参考一致，误差在浮点规约顺序导致的舍入范围内。

### 3.2 segment reference vs simsopt `.B()`

为了验证离散线圈本身是否接近 simsopt 连续线圈，在本地用 simsopt 做了轴附近区域对照。

采样区域：

- `01/raw`: $R_0=1.13107, a=0.05$
- `debug/raw`: $R_0=1.17312, a=0.02$

结果：

| case | `segments_per_coil` | mean rel | p95 rel | max rel |
|---|---:|---:|---:|---:|
| `01/raw` | 128 | 4.73e-06 | 6.22e-06 | 3.99e-04 |
| `01/raw` | 256 | 2.38e-10 | 7.01e-11 | 3.02e-08 |
| `01/raw` | 512 | 4.17e-10 | 6.98e-11 | 4.75e-08 |
| `debug/raw` | 128 | 3.54e-04 | 1.80e-03 | 1.78e-02 |
| `debug/raw` | 256 | 4.71e-06 | 1.20e-06 | 1.01e-03 |
| `debug/raw` | 512 | 1.12e-05 | 1.84e-06 | 2.54e-03 |

判断：

- `01/raw` 用 `S=256` 已经非常接近 simsopt。
- `debug/raw` 用 `S=256` 的 p95 也在 `1e-6` 量级，但 max 仍有 `1e-3` 量级点。
- `S=512` 没有稳定改善 debug 的 max，说明差异可能不只是 segment 数，还包括 simsopt 线圈积分规则、近线圈敏感点或对称实现细节的高阶差异。

当前建议默认：

```text
segments_per_coil = 256
```

并在最终接入主流程时保留 `segments_per_coil` 为可调参数。

## 4. 内核速度测试

测试命令在服务器上运行，使用 `CUDA_VISIBLE_DEVICES=0`。

### 4.1 `eval_B` 与一周期 RK4 时间

每组：

```text
points = 8192
lines = 256
steps = 800
repeat = 3
```

| case | `segments_per_coil` | 总电流元 | `eval_B(8192)` median | `trace_period(256 lines, 800 steps)` median |
|---|---:|---:|---:|---:|
| `01/raw` | 128 | 2048 | 0.000572 s | 0.257 s |
| `01/raw` | 256 | 4096 | 0.001082 s | 0.501 s |
| `01/raw` | 512 | 8192 | 0.002114 s | 0.988 s |
| `debug/raw` | 128 | 2560 | 0.000704 s | 0.318 s |
| `debug/raw` | 256 | 5120 | 0.001354 s | 0.622 s |
| `debug/raw` | 512 | 10240 | 0.002615 s | 1.231 s |

速度近似随总电流元数线性变化。这符合当前 kernel 的工作量模型：

$$
O(N_{\rm lines}\times N_{\rm steps}\times N_{\rm segments}).
$$

## 5. GPU GA 找磁轴实验

GPU GA 使用和 CPU 版相同的控制逻辑：

1. 初始 `16x16=256` 点；
2. 每代 GPU 批量追踪一周期；
3. CPU 排序 top16；
4. top16 两两取中点；
5. 重复到闭合残差低于 `1e-8`。

### 5.1 `segments_per_coil` 扫描

| case | `segments_per_coil` | 总电流元 | 收敛 | 代数 | best R | best Z | residual | GA time |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `01/raw` | 128 | 2048 | true | 15 | 1.131068875 | -4.03e-18 | 7.41e-09 | 4.13 s |
| `01/raw` | 256 | 4096 | true | 9 | 1.131068868 | -2.28e-18 | 3.72e-09 | 5.01 s |
| `01/raw` | 512 | 8192 | true | 19 | 1.131068868 | 3.12e-09 | 7.46e-09 | 19.76 s |
| `debug/raw` | 128 | 2560 | true | 14 | 1.214626991 | -3.39e-09 | 9.26e-09 | 4.78 s |
| `debug/raw` | 256 | 5120 | true | 17 | 1.214626994 | -9.87e-18 | 1.22e-09 | 11.22 s |
| `debug/raw` | 512 | 10240 | true | 19 | 1.214626994 | -1.20e-09 | 2.22e-09 | 24.62 s |

### 5.2 与当前 CPU/simsopt 版对比

当前封装的 CPU/simsopt 版细粒度计时：

| case | CPU/simsopt GA axis time | GPU GA time, S=256 | 加速比 |
|---|---:|---:|---:|
| `01/raw` | 72.22 s | 5.01 s | 14.4x |
| `debug/raw` | 104.02 s | 11.22 s | 9.3x |

轴位置对比：

| case | CPU/simsopt R | GPU S=256 R | 差值 |
|---|---:|---:|---:|
| `01/raw` | 1.131068870 | 1.131068868 | 1.81e-09 |
| `debug/raw` | 1.214626994 | 1.214626994 | 9.18e-10 |

说明 GPU segment 后端在轴搜索任务上已经能复现当前 CPU/simsopt 封装结果。

注意：`debug/raw` 的轴位置是当前封装支持两根基础线圈后的结果，和早期只取第一根基础线圈的旧实验不同。

## 6. 瓶颈变化

### 6.1 迁移前

迁移前，找磁轴几乎完全由 simsopt `.B()` 支配：

| case | axis total | `.B()` time |
|---|---:|---:|
| `01/raw` | 72.22 s | 70.45 s |
| `debug/raw` | 104.02 s | 101.80 s |

### 6.2 GPU 后端第一阶段

迁移到 GPU 后，GA 找轴的核心耗时变为：

| case | GPU GA total | 每代 trace median |
|---|---:|---:|
| `01/raw` | 5.01 s | 0.501 s |
| `debug/raw` | 11.22 s | 0.622 s |

新的瓶颈仍然是 Biot-Savart 求和，但已经从 CPU/OpenMP/Python `.B()` 调度变成 GPU kernel 内的 segment 求和。

### 6.3 后续主流程中的瓶颈预测

如果把 GPU 后端接入主评估器：

1. `axis_s` 会从 `72/104 s` 降到 `5/11 s` 量级；
2. `surface_screen_fieldline_trace_s` 也应显著降低；
3. $\psi$ fit 当前总耗时约 `14 s`，其中 `.B()` 只有 `0.5 s`，所以只替换 $B$ 后不会大幅加速；
4. 接入 GPU 后，新的主要瓶颈可能变成：
   - $\psi$ fit 的设计矩阵/正规方程组装；
   - surface 等值面提取；
   - Python 控制逻辑和数据搬运。

因此下一阶段如果要继续优化，应把 $\psi$ fit 的 `ATA/ATb` 组装也迁到 GPU。

## 7. 当前实现的限制

### 7.1 `eval_B` 每次调用都会分配/释放 device memory

当前 C ABI 简化实现中，`sgpu_eval_B` 和 `sgpu_trace_period` 每次都会：

1. `cudaMalloc`;
2. host-to-device copy;
3. kernel;
4. device-to-host copy;
5. `cudaFree`。

对于 GA 来说，每代一次调用，影响不大。但未来如果高频调用 `eval_B`，应该加入持久 buffer 或 Python 侧批量接口。

### 7.2 CPU 控制 GA

当前 top16 排序和下一代生成仍在 CPU。因为每代只有 256 点，数据量很小，当前不是瓶颈。未来可选地迁移到 GPU，但优先级低。

### 7.3 只有 double 实现

当前全部使用 double。RTX 5090 的 FP64 吞吐不如 FP32。后续可以加 mixed precision 版本：

- segment 和坐标用 double；
- 部分累加用 double；
- 或测试 float 累加对 axis residual 的影响。

这可能显著提速，但必须做物理误差验证。

### 7.4 尚未接入主评估器

当前 GPU 后端仍是独立原型。主评估器仍走 Python/simsopt。下一步需要加后端选择：

```python
backend = "simsopt" | "cuda"
```

并把 GA 找轴和 $\psi_0$ 筛选的 trace 调用切到 GPU。

## 8. 下一步建议

### 优先级 1：接入主评估器的 GA 找轴

把当前 `gpu_axis_ga.py` 的逻辑接入 `stellarator_eval.axis`，形成：

```python
AxisBackendSimsopt
AxisBackendCuda
```

验收标准：

- `01/raw`、`debug/raw` 轴位置与当前 verified 结果一致；
- `summary.json` 中记录 backend、segments、GPU trace 时间；
- 不改变后续 $\psi$/Boozer 输出格式。

### 优先级 2：接入 $\psi_0$ 筛选 trace

把筛选阶段所有 level 的初始点拼成一个大 batch，一次或少数几次调用 GPU `trace_period`。

预期收益：

- `01/raw` 的 screen trace 从 `6.31 s` 降到约 `0.5-1 s`；
- `debug/raw` 的 screen trace 从 `9.94 s` 降到约 `0.6-1.5 s`。

实际收益取决于 batch 大小和 segments 数。

### 优先级 3：优化 GPU 后端内存管理

增加 persistent buffers：

```cpp
reserve_points_capacity(n)
eval_B_no_alloc(...)
trace_period_no_alloc(...)
```

减少 `cudaMalloc/cudaFree` 开销。

### 优先级 4：$\psi$ fit 的 GPU `ATA/ATb`

把采样点上的 $B\cdot\nabla f_j$ 和正规方程组装迁移到 GPU。这个阶段比单纯 GPU `B` 更有意义，因为当前 $\psi$ fit 的 `.B()` 本身不是主要瓶颈。

## 9. 资源使用情况

所有实验均限制：

```text
CUDA_VISIBLE_DEVICES=0
```

没有使用 Slurm，原因是本阶段只是短时间开发 benchmark。每个命令运行时间在几十秒以内。

实验结束后检查：

```text
nvidia-smi --query-compute-apps
ps -u $USER
```

没有发现残留 GPU 进程或僵尸计算进程。

## 10. 本阶段结论

第一阶段验证成功。

关键结论：

1. C++/CUDA 自建线圈 segment 后端可行；
2. tile shared memory 是正确方向，避免了 shared memory 不足；
3. warp-per-line 的 RK4 设计有效；
4. `segments_per_coil=256` 是当前较好的默认折中；
5. GA 找轴已获得 `9x-14x` 的端到端加速；
6. 轴位置与当前 CPU/simsopt 封装结果一致；
7. 下一步应把 GPU 后端接入主评估器，优先替换 GA 找轴和 $\psi_0$ 筛选。

