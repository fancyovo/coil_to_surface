# GPU 迁移可行性报告

## 0. 目标和边界

当前 Python/simsopt 版评估器的输入输出已经基本确定：

- 输入：线圈 Fourier 系数、电流、`nfp`、评估半径 `a` 和若干可选超参数。
- 输出：磁轴闭合残差、磁轴曲线、$\psi$ 模数和系数、候选磁面筛选结果、最大可信磁面的 $\iota$、volume、$G$、QS error 等。

GPU 迁移的目标不是改变物理定义，而是替换最耗时的底层数值内核：

1. 批量磁场 $B(x)$ 计算；
2. 批量固定步长 RK4 磁力线追踪；
3. $\psi$ 拟合采样阶段的 $B$ 批量计算；
4. $\psi_0$ 筛选阶段的一周期多线追踪。

暂时不建议第一阶段重写 Boozer LS/Newton。当前 Boozer 部分在实测中不是瓶颈，且依赖 simsopt 的几何对象、残差和优化器，直接迁移成本高。

## 1. 当前瓶颈回顾

现有细粒度计时显示，核心瓶颈是磁轴搜索和 $\psi_0$ 筛选中的批量磁力线追踪。

| case | 阶段 | 总时间 | 其中 `.B()` 时间 |
|---|---|---:|---:|
| `01/raw` | GA 找磁轴 | 72.22 s | 70.45 s |
| `debug/raw` | GA 找磁轴 | 104.02 s | 101.80 s |
| `01/raw` | $\psi_0$ 筛选场线 | 6.31 s | 6.08 s |
| `debug/raw` | $\psi_0$ 筛选场线 | 9.94 s | 9.63 s |

因此优先迁移方向非常明确：先做 GPU 版线圈磁场和固定步长 RK4。

## 2. 自建 C++/CUDA 线圈磁场类是否可行

可行，而且是必要的。

当前 simsopt 的 `.B()` 是通用实现，对我们的工作流有两个不利点：

1. 每次 Python 调用有调度开销；
2. 小批量时 OpenMP 同步成本很高；
3. 数据布局不是为我们固定的“同一组线圈、海量点反复求 $B$”优化的；
4. GPU 侧无法直接复用 simsopt 内部数据结构。

建议建立 C++/CUDA 类：

```cpp
class CoilFieldGpu {
public:
    CoilFieldGpu(CoilFourierInput input, int nfp, int segments_per_base_coil);
    void eval_B_device(const double3* points, double3* B, int n_points);
    void trace_rk4_period(...);
};
```

实例化时完成：

1. 将 Fourier 线圈离散成小电流元；
2. 应用 `nfp` 和 stellarator symmetry，生成完整线圈系统；
3. 将每个电流元整理为 GPU 友好的结构；
4. 上传到 device memory；
5. 可选地复制常用块到 constant memory 或 texture/read-only cache。

每个电流元可存为：

```cpp
struct Segment {
    double3 x_mid;   // 电流元中点
    double3 dl;      // 方向长度向量
    double current;  // 电流
};
```

磁场公式：

$$
B(x)=\frac{\mu_0}{4\pi}\sum_j I_j\frac{dl_j\times(x-x_j)}{|x-x_j|^3}.
$$

### 数据布局

为了合并访存，建议不要用 AoS 作为主布局，而是 SoA：

```cpp
double* seg_x;
double* seg_y;
double* seg_z;
double* dl_x;
double* dl_y;
double* dl_z;
double* current;
```

原因是 warp 内 32 个 thread 会读取连续的 segment index，SoA 能让每个字段连续读取。

如果后续发现寄存器压力较低，也可以测试 `double4`：

```cpp
double4 pos_current; // x,y,z,I
double4 dl_pad;      // dlx,dly,dlz,pad
```

但第一版建议 SoA，便于控制访存模式。

## 3. 你提出的 warp-per-fieldline RK4 策略

你的策略核心是：

- 一条磁力线由一个 warp 负责；
- 一个 block 负责 8 条线，即 8 warps = 256 threads；
- 每个 warp 的 32 个 thread 分摊所有电流元；
- thread `i` 累加 index `% 32 == i` 的电流元贡献；
- warp 内规约得到该点总 $B$；
- 每个 RK4 stage 都这样求一次 $B$；
- 一个 kernel launch 完成一批线的一周期追踪。

这个设计总体可行，而且很适合当前问题。

### 优点

1. 每条线的 RK4 步数固定，没有复杂分支；
2. 所有线共享同一组线圈数据；
3. 单条线求一个点的 $B$ 时，电流元求和天然并行；
4. warp 内规约开销低；
5. 一个 kernel 内完成整条线，避免每一步 kernel launch；
6. 对 GA 轴搜索和 $\psi_0$ 筛选都适用。

### 主要风险

#### 风险 1：shared memory 放不下所有电流元

这个要具体估算。

当前例子的基础线圈 Fourier order 是 16，因为每个坐标有 33 个系数。若每根基础线圈切成 `S` 段，完整线圈数是：

$$
n_{\rm total}=n_{\rm base}\times 2n_{\rm fp}
$$

例如：

- `01/raw`: `n_base=1, nfp=8`，完整线圈数 `16`；
- `debug/raw`: `n_base=2, nfp=5`，完整线圈数 `20`。

若每根完整线圈切 `S=256` 段：

| case | 总电流元数 |
|---|---:|
| `01/raw` | 4096 |
| `debug/raw` | 5120 |

每个电流元若用 7 个 `double` 存储：

$$
7\times8=56\text{ bytes}.
$$

那么 shared memory 需求：

| 电流元数 | shared memory |
|---:|---:|
| 4096 | 224 KiB |
| 5120 | 280 KiB |

这通常超过单个 block 可用 shared memory。RTX 5090 具体上限需要实测，但不应假设能一次塞下全部电流元。

因此“首先直接把电流元的信息全塞进 shared memory”这一步对中等以上 segment 数可能不可行。

#### 推荐修正：分 tile 加载电流元

更稳的做法是 tile 化：

```text
for tile in segments:
    block 内协作加载 tile 到 shared memory
    每个 warp 对 tile 内电流元做部分求和
    累加到寄存器 partial_B
warp reduce partial_B
```

tile 大小可以从 `256` 或 `512` 个 segment 开始测试。这样 shared memory 需求大约：

```text
512 segments * 56 bytes = 28 KiB
```

对 8 warps/block 是合理的。

#### 风险 2：一个 block 8 条线是否最佳不确定

8 warps/block = 256 threads，通常合理。但最优值依赖：

- 每个 thread 的寄存器数量；
- tile shared memory 大小；
- occupancy；
- 线圈段数；
- double precision 吞吐。

建议作为可调模板参数：

```cpp
template<int WARPS_PER_BLOCK, int SEG_TILE>
__global__ void trace_kernel(...);
```

初始测试：

| 参数 | 候选 |
|---|---|
| `WARPS_PER_BLOCK` | 4, 8 |
| `SEG_TILE` | 256, 512 |

#### 风险 3：double precision 性能

RTX 5090 是消费级 GPU，FP64 吞吐通常远低于 FP32。磁场积分是否必须全 double 需要测试。

建议第一版支持两种精度：

1. `double`：作为基准，验证与 CPU/simsopt 一致性；
2. `float` 或 mixed precision：用于性能探索。

比较指标：

- 单点 $B$ 相对误差；
- 一周期闭合残差误差；
- GA 找到的轴位置误差；
- $\psi_0$ drift 排序是否稳定。

对于最终物理评估，至少磁轴和候选筛选阶段可以考虑 mixed precision；但 Boozer 前的最终 surface 生成最好保留 double 或做 double 复核。

## 4. GPU 版 RK4 追踪接口设计

磁力线方程仍以 $\Phi$ 为自变量：

$$
\frac{dR}{d\Phi}=\frac{R B_R}{B_\Phi},\qquad
\frac{dZ}{d\Phi}=\frac{R B_Z}{B_\Phi}.
$$

GPU kernel 输入：

```cpp
struct TraceConfig {
    int nfp;
    int steps;
    double phi0;
    double phi1;
};

trace_period(
    const double* R0,
    const double* Z0,
    double* R1,
    double* Z1,
    int n_lines,
    TraceConfig cfg
);
```

对 GA 找轴，只需要终点：

```cpp
R0,Z0 -> R_end,Z_end -> residual
```

对 $\psi_0$ 筛选，可能还需要沿途最大 drift 或末端 drift。第一版可以只返回终点，回到 CPU 计算：

$$
d_\psi=\frac{|\psi_{\rm end}-\psi_0|}{|\nabla\psi|}.
$$

如果后续想进一步减少 CPU/GPU 往返，可以把 $\psi$ 和 $\nabla\psi$ 的评估也搬到 GPU。

## 5. GA 找轴的 GPU 版本

GA 控制逻辑可先留在 CPU：

1. CPU 生成 256 个候选点；
2. GPU 批量追踪一周期；
3. GPU 或 CPU 计算残差；
4. CPU 排序 top16；
5. CPU 生成下一代；
6. 重复。

这个版本实现简单，但每代有一次 host-device 往返。因为每代追踪成本远大于 256 个点的数据传输，这个开销可以接受。

后续可把残差计算和 top16 选择也放 GPU，但不是第一阶段重点。

## 6. $\psi$ 拟合的 GPU 化

当前 $\psi$ 拟合的主要工作包括：

1. 构造采样点；
2. 计算 $B$；
3. 计算每个基函数的 $B\cdot\nabla f_j$；
4. 组装正规方程：

$$
A^T A,\qquad A^T b.
$$

你说“最小二乘应该有经典做法或库”，这个判断是对的。可选路线：

### 路线 A：GPU 只算 $B$，CPU 继续组装和求解

这是第一阶段最稳的方案。

优点：

- 改动小；
- 容易验证；
- 可先复用现有 Python/NumPy 最小二乘；
- 风险低。

缺点：

- 当前实测中 $\psi$ fit 的 `.B()` 只占 `0.5 s`，而总耗时约 `14 s`；
- 只搬 $B$ 对总加速有限。

### 路线 B：GPU 直接组装 $A^T A$ 和 $A^T b$

更有价值。

采样点数量约 `1e5`，模式数默认 `1574`。直接显式保存完整 $A$：

$$
10^5\times1574\approx1.57\times10^8
$$

若 double，大约 `1.26 GB`，可以放进 5090 显存，但没必要。

更好的方案是分块累计：

```text
for point_batch:
    GPU 计算 A_batch 和 b_batch
    cuBLAS: ATA += A_batch^T A_batch
    cuBLAS: ATb += A_batch^T b_batch
```

最终 `ATA` 大小：

$$
1574^2\times8\approx19.8\text{ MB}.
$$

这很小。线性求解可以用 cuSOLVER 或回 CPU 求解。

### 路线 C：直接 QR/SVD 最小二乘

数值更稳，但实现更复杂。第一阶段不建议直接上。

当前正规方程条件数在 `1e7` 到 `2e7`，还没有到完全不可用，但要保留 ridge 和 CPU 对照。

## 7. $\psi_0$ 筛选的 GPU 化

$\psi_0$ 筛选和 GA 找轴很像：

- 多个 $\psi_0$；
- 每个 $\psi_0$ 多个初始点；
- 每条线追踪一个 field period；
- 计算末端 $d_\psi$。

第一版设计：

1. CPU 负责对每个 $\psi_0$ 解 $\Phi=0$ 截面曲线；
2. 把所有 level 的初始点拼成一个大 batch；
3. GPU 一次或少数几次 kernel 完成追踪；
4. CPU 计算每个 level 的 p95 drift 并筛选。

这样比当前逐 level 调用更好，也更适合 GPU。

后续可以把一维 Newton 解 $\psi=\psi_0$ 也搬到 GPU，但它当前不是瓶颈。当前一维 Newton 总共不到 1 秒。

## 8. 是否要把 Poincare/G 共轭路线放入 GPU 第一版

不建议。

原因：

1. 当前最终封装没有把这条路线作为主路径；
2. 它在 `debug/raw` 上出现过 $h$ 非单调；
3. 主要瓶颈不是这条路线；
4. GPU 第一阶段应先替换确定正确且占时最高的内核：$B$ 和 RK4。

可以把它作为第二阶段增强：

```text
GPU fieldline traces -> Poincare map -> iota/G estimate -> h monotonicity check -> optional Boozer-like initialization
```

但必须保留硬失败条件：

$$
F'(\alpha)>0,\qquad h'(\alpha)>0.
$$

## 9. C++/CUDA 与 Python 的接口

建议用 `pybind11` 暴露最小接口：

```python
from stellarator_gpu import CoilFieldGpu

field = CoilFieldGpu(coeffs_x, coeffs_y, coeffs_z, currents, nfp, segments=256)
B = field.eval_B(points)
R1, Z1 = field.trace_period(R0, Z0, steps=800)
```

Python 层继续负责任务编排、JSON/NPZ 输出、Boozer 调用。

这样可以保持当前输入输出结构不变：

```text
Python evaluator
  -> GPU field/RK4 backend
  -> existing psi/surface/Boozer pipeline
```

后续如果 $\psi$ fit 也完全 GPU 化，可以继续添加：

```python
ATA, ATb = field.assemble_psi_normal_equations(...)
```

## 10. 远程开发目录建议

服务器 `/home` 空间只剩约 73 GB，虽然本项目不大，但编译缓存、conda 环境和中间文件可能增长。你提出在 `~/` 下新建目录可以接受，但建议把大文件和环境放 `/data`。

建议目录：

```text
~/stellarator_gpu_eval
```

如果后续需要较大的构建目录或缓存：

```text
/data/cyfan/stellarator_gpu_eval_build
```

资源使用约束：

- 默认只用 `CUDA_VISIBLE_DEVICES=0`；
- CPU 线程默认不超过 16；
- 开发时直接命令行跑轻量测试；
- 长时间 benchmark 或大规模 sweep 走 Slurm；
- 每次调试后检查 `nvidia-smi` 和 `ps -u $USER`，避免遗留进程。

## 11. 第一阶段实施计划

### 阶段 1：CPU/CUDA 磁场一致性

目标：自建线圈离散和 Biot-Savart 与 simsopt `.B()` 对齐。

任务：

1. C++ 生成完整对称线圈电流元；
2. CPU 版 segment Biot-Savart；
3. CUDA 版 batch `eval_B`;
4. 随机点对比 simsopt `.B()`。

验收：

```text
mean relative B error < 1e-5 到 1e-6
max relative B error 可解释
```

误差取决于 `segments_per_coil`。需要扫描：

```text
segments = 128, 256, 512, 1024
```

### 阶段 2：GPU RK4 一周期追踪

目标：替换 GA 找轴中的 RK4。

任务：

1. 实现 warp-per-line RK4 kernel；
2. 实现 tile shared memory segment 加载；
3. CPU 控制 GA；
4. 对比现有 Python 结果。

验收：

| case | 目标 |
|---|---|
| `01/raw` | 找到轴残差 $\lesssim10^{-8}$ |
| `debug/raw` | 找到轴残差 $\lesssim10^{-8}$ |

并记录加速比。

### 阶段 3：GPU $\psi_0$ 筛选

目标：把所有候选 $\psi_0$ 的 fieldline screening 合成一个 GPU batch。

任务：

1. CPU 解各 level 初始曲线；
2. GPU 批量追踪；
3. CPU 计算 $d_\psi$ p95；
4. 与当前筛选结果对齐。

验收：

`01/debug` 上通过/失败的 level 与 CPU 版一致或差异可解释。

### 阶段 4：GPU $\psi$ fit 组装

目标：加速 $\psi$ 拟合的设计矩阵/正规方程组装。

优先实现分块 `ATA/ATb` 累计，而不是显式保存完整 $A$。

验收：

1. 系数与 CPU 版接近；
2. 验证集 angle 指标接近；
3. 下游 surface screen 结果接近。

## 12. 总体可行性判断

可行，且优先级明确。

最值得先做的是：

```text
C++/CUDA 自建 Biot-Savart + warp-per-line RK4
```

你提出的 warp 级设计方向是对的，但“把所有电流元一次性塞进 shared memory”需要改成 tile 化，否则 shared memory 很可能不够。一个 block 8 条线是合理初值，但应保留为可调参数。

第一阶段不建议动 Boozer LS/Newton，也不建议把 Poincare/G 共轭路线作为主线。先用 GPU 替换当前最确定、最耗时、最容易验证的部分，保持最终输入输出不变。

