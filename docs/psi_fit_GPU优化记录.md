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
