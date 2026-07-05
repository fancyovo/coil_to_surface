# psi fit GPU 优化记录

## 1. 当前问题规模

以 `01/raw`、`D=10`、`M=12`、`n_r=n_z=n_phi=80` 为例：

```text
training points N = 389440
unknowns p = 1574
```

正规方程中 $A^T A$ 的完整矩阵乘法约需要：

$$
2Np^2 \approx 1.93\times 10^{12}
$$

也就是约 `1.93 TFLOP`。如果只按对称矩阵的理论下三角计算，约 `0.965 TFLOP`，但 NumPy/OpenBLAS 的 `mat.T @ mat` 实际走完整 GEMM。

## 2. 已完成优化

### 2.1 GPU 批量 B 采样

`psi fit` 支持：

```bash
--psi-backend gpu
```

此时训练点和验证点的 $B$ 采样使用 `CoilFieldGpu.eval_B`。在 `01/raw` 上，训练集 $B$ 采样时间约：

```text
0.058 s
```

这说明当前 `psi fit` 的瓶颈已经不是 $B$ 采样。

### 2.2 基函数矩阵向量化

原实现逐 mode Python 循环填充设计矩阵，耗时约：

```text
89.7 s
```

现在按“二元单项式块 × toroidal Fourier 表”批量填充，耗时约：

```text
1.75 - 2.13 s
```

这一步是本轮最大收益。

### 2.3 cuBLAS normal equation

新增：

```bash
--psi-normal-eq-backend auto|cpu|gpu
--psi-normal-eq-precision fp64|fp32
```

`gpu/fp64` 使用 cuBLAS `Dgemm/Dgemv`。  
`gpu/fp32` 使用 cuBLAS `Sgemm/Sgemv`，然后把 $A^T A,A^Tb$ 转回 double 做后续求解。

注意：`fp32` 路径可能使用 GPU 的快速单精度/TF32 执行路径，速度更快，但由于这里解的是正规方程，数值误差会被条件数放大。

## 3. 远端 01/raw 实测

运行环境：

```text
GPU: 1 张 5090
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=16
```

### 3.1 完整链路耗时

| 配置 | total | psi fit | normal eq | basis | validation |
|---|---:|---:|---:|---:|---:|
| 优化前 GPU B + 旧 Python mode 循环 | `103.50 s` | `100.77 s` | `9.58 s` | `89.71 s` | `1.01 s` |
| 向量化 + CPU normal eq | `6.55 s` | `4.09 s` | `1.52 s` | `2.13 s` | `0.008 s` |
| 向量化 + GPU fp64 normal eq | `6.32 s` | `3.88 s` | `1.71 s` | `1.76 s` | `0.008 s` |
| 向量化 + GPU fp32 normal eq | `5.64 s` | `3.03 s` | `0.87 s` | `1.75 s` | `0.009 s` |

### 3.2 等效 TFLOP/s

按完整 GEMM 的 `1.93 TFLOP` 估算：

| 配置 | normal eq 时间 | 等效吞吐 |
|---|---:|---:|
| CPU OpenBLAS 16 线程 | `1.52 s` | `1.27 TFLOP/s` |
| GPU fp64 cuBLAS | `1.71 s` | `1.13 TFLOP/s` |
| GPU fp32 cuBLAS | `0.87 s` | `2.22 TFLOP/s` |

这里 GPU fp64 不快是合理的：消费级 GPU 的 fp64 峰值很低，而且当前实现每个 batch 都有 host/device 拷贝和 cuBLAS 调用开销。GPU fp32 有收益，但还远没吃满 5090 的单精度峰值，因为设计矩阵仍在 CPU 生成，并且每个 batch 仍要把矩阵传到 GPU。

## 4. fp32 精度影响

`01/raw` 上：

| 指标 | CPU/GPU fp64 | GPU fp32 |
|---|---:|---:|
| condition number | `1.56e7` | `7.69e7` |
| train RMS | `4.25e-4` | `8.91e-4` |
| validation angle mean | `1.79e-5` | `2.12e-5` |
| validation angle p95 | `5.53e-5` | `6.26e-5` |
| iota | `-2.90768087` | `-2.90700598` |
| volume | `6.150e-3` | `6.021e-3` |
| QS QA | `2.497e-3` | `2.446e-3` |

结论：`fp32` 可以作为快速筛选档，但不应作为默认精度输出最终评估结果。默认建议仍使用 fp64 normal equation；若需要快速预筛，可用 `--psi-normal-eq-precision fp32`。

## 5. 当前瓶颈

在 `GPU fp32 normal eq` 下，`psi fit` 内部主要耗时为：

```text
basis fill:      ~1.75 s
normal equation: ~0.87 s
solve:           ~0.30 s
B sampling:      ~0.06 s
```

也就是说下一步真正值得 GPU 化的是“在 GPU 上直接生成设计矩阵并累计 $A^TA,A^Tb$”，避免：

1. CPU 生成 `mat`。
2. `mat` 从 host 拷到 device。
3. 每个 batch 独立分配 GPU buffer。

更进一步可以不显式保存完整 `mat`，而是由 CUDA kernel 直接对 mode pair 分块累计正规方程。

## 6. 全 GPU 训练链路

本轮新增了 `fullgpu` 路径：

```text
CPU 仅传训练点与磁轴表
-> GPU 上采样 B
-> GPU 上组装完整设计矩阵 A 和 rhs
-> GPU 上做 A^T A / A^T b
-> GPU 上用 cuSOLVER 解线性方程
-> CPU 仅接收 coeff 向量与少量标量
```

当前命令入口：

```bash
--psi-backend fullgpu
--psi-normal-eq-precision fp64|fp32
```

### 6.1 01/raw 全尺寸直接 fit 对比

| 配置 | wall | train RMS | val angle mean | val angle p95 |
|---|---:|---:|---:|---:|
| `gpu + CPU normal eq` | `4.19 s` | `4.247e-4` | `1.787e-5` | `5.533e-5` |
| `fullgpu fp64` | `1.46 s` | `4.247e-4` | `1.787e-5` | `5.533e-5` |
| `fullgpu fp32` | `0.115 s` | `1.302e-1` | `9.581e-3` | `3.840e-2` |

结论：

- `fullgpu fp64` 已经可用，结果与原路径一致到数值误差范围。
- `fullgpu fp32` 目前不能用于实际拟合，虽然极快，但误差已经大到失真。

### 6.2 01/raw 全链路

`fullgpu fp64` 端到端：

```text
total wall                ~5.84 s
axis                      ~1.08 s
psi fit                   ~1.43 s
surface screen            ~0.40 s
surface extract 1D newton ~0.90 s
boozer candidate          ~1.92 s
```

其中 `psi fit` 内部：

```text
copy_in      0.00086 s
assemble     1.35864 s
  basis      0.07258 s
  A^T A      1.28605 s
solve        0.01609 s
residual     0.00328 s
copy_out     0.00001 s
```

### 6.3 当前解释

`fullgpu fp64` 已经说明主路径是对的，而且当前 `psi fit` 比先前的 `gpu + CPU normal eq` 再快约 `2.9x`。

`fullgpu fp32` 虽然速度极高，但它把：

```text
B 采样
矩阵组装
A^T A
残差评估
```

都压成了单精度路径。对于当前约 `1.56e7` 条件数的系统，这个误差太大，不足以直接用于最终拟合。后续如果要继续做混合精度，应优先尝试：

1. `A` 用 fp32 存储；
2. `A^T A` 用更高精度累计；
3. 求解仍保留 fp64。
