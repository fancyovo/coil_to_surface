# GPU TFLOPs 与瓶颈判断

## 1. 目的

本报告根据第一阶段 CUDA 后端 benchmark，估算实际 TFLOPs 和访存带宽，判断当前瓶颈更接近：

1. global memory 访存；
2. shared memory/寄存器局部带宽；
3. FP64 浮点算力和特殊函数吞吐；
4. kernel 并行度不足。

结论先行：

- 单独 `eval_B` 已经达到约 `0.8-1.0 TFLOP/s` FP64 等效吞吐，更接近双精度算力/特殊函数瓶颈，不是 global memory 瓶颈。
- 当前 `trace_period(256 lines)` 只有约 `0.17-0.22 TFLOP/s`，主要不是访存瓶颈，而是 kernel 只有 32 个 block，GPU 没吃满。
- 所以当前最需要优化的是追踪 kernel 的并行粒度，而不是 global memory 带宽。

## 2. 每个电流元贡献的 FLOP 估计

每个 segment 对一个点的贡献大致是：

```cpp
rx = px - sx;
ry = py - sy;
rz = pz - sz;
r2 = rx*rx + ry*ry + rz*rz;
invr = rsqrt(r2);
invr3 = invr * invr * invr;
B += cross(wdl, r) * invr3;
```

常规加乘计数：

| 操作 | FLOP |
|---|---:|
| 3 个坐标差 | 3 |
| $r^2$ 的 3 乘 + 3 加 | 6 |
| `invr3` 的 2 乘 | 2 |
| cross 三个分量，每分量约 5 FLOP | 15 |
| 合计，不含 `rsqrt` | 26 |

`rsqrt(double)` 是特殊函数，不能简单等同于 1 FLOP。为了给出范围，本报告使用两个口径：

```text
保守口径：26 FLOP / segment contribution
偏实际口径：32 FLOP / segment contribution
```

后文表格同时给出两个口径对应范围。

## 3. benchmark 数据

来自服务器 RTX 5090，`CUDA_VISIBLE_DEVICES=0`。

测试参数：

```text
eval_B: points = 8192
trace_period: lines = 256, steps = 800
RK4: 每 step 4 次 B
WARPS_PER_BLOCK = 8
```

## 4. `eval_B` 的 TFLOPs

交互次数：

$$
N_{\rm interact}=N_{\rm points}N_{\rm seg}.
$$

| case | S | 总 segment | 时间 | TFLOP/s, 26 FLOP | TFLOP/s, 32 FLOP |
|---|---:|---:|---:|---:|---:|
| `01/raw` | 128 | 2048 | 0.000572 s | 0.763 | 0.939 |
| `01/raw` | 256 | 4096 | 0.001082 s | 0.806 | 0.992 |
| `01/raw` | 512 | 8192 | 0.002114 s | 0.825 | 1.016 |
| `debug/raw` | 128 | 2560 | 0.000704 s | 0.774 | 0.953 |
| `debug/raw` | 256 | 5120 | 0.001354 s | 0.806 | 0.992 |
| `debug/raw` | 512 | 10240 | 0.002615 s | 0.834 | 1.026 |

结论：

- `eval_B` 稳定在 `0.8-1.0 TFLOP/s`。
- 随 segment 数增大，TFLOP/s 略升，说明小规模时还有固定开销或 occupancy 不足。
- 对 RTX 5090 这类消费级 GPU，FP64 峰值远低于 FP32。按消费级 NVIDIA 常见的 FP64 限速估计，`0.8-1.0 TFLOP/s` 已经不是很低。

## 5. `trace_period` 的 TFLOPs

交互次数：

$$
N_{\rm interact}=N_{\rm lines}\times N_{\rm steps}\times4\times N_{\rm seg}.
$$

| case | S | 总 segment | 时间 | TFLOP/s, 26 FLOP | TFLOP/s, 32 FLOP |
|---|---:|---:|---:|---:|---:|
| `01/raw` | 128 | 2048 | 0.257 s | 0.169 | 0.209 |
| `01/raw` | 256 | 4096 | 0.501 s | 0.174 | 0.214 |
| `01/raw` | 512 | 8192 | 0.988 s | 0.177 | 0.217 |
| `debug/raw` | 128 | 2560 | 0.318 s | 0.171 | 0.211 |
| `debug/raw` | 256 | 5120 | 0.622 s | 0.175 | 0.216 |
| `debug/raw` | 512 | 10240 | 1.231 s | 0.177 | 0.218 |

结论：

- `trace_period` 只有 `0.17-0.22 TFLOP/s`，明显低于单独 `eval_B`。
- 原因不是每次 $B$ 计算变慢，而是当前 kernel 的并行度太低。

当前设计：

```text
1 warp = 1 fieldline
1 block = 8 fieldlines
256 lines => 32 blocks
```

RTX 5090 的 SM 数远多于 32 个 block 能充分覆盖的范围，因此 kernel occupancy 明显不足。单个 block 要在内部跑完整 800 step RK4，block 生命周期很长，但全 GPU 同时活跃 block 数太少。

## 6. global memory 带宽估算

当前 tile shared memory 设计中，每个 block 对每个 segment tile 从 global memory 读取一次。一个 segment 读取：

```text
seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz
6 doubles = 48 bytes
```

但一个 block 有 8 个 warp，对应 8 个点或 8 条线。因此 global memory 摊到每个 point-segment contribution 上约为：

$$
\frac{48}{8}=6\text{ bytes}.
$$

### 6.1 `eval_B` 的 global bandwidth

| case | S | global bandwidth |
|---|---:|---:|
| `01/raw` | 128 | 176 GB/s |
| `01/raw` | 256 | 186 GB/s |
| `01/raw` | 512 | 190 GB/s |
| `debug/raw` | 128 | 179 GB/s |
| `debug/raw` | 256 | 186 GB/s |
| `debug/raw` | 512 | 192 GB/s |

RTX 5090 的显存带宽是 TB/s 量级。因此 `180-190 GB/s` 远未触顶。

### 6.2 `trace_period` 的 global bandwidth

| case | S | global bandwidth |
|---|---:|---:|
| `01/raw` | 128 | 39 GB/s |
| `01/raw` | 256 | 40 GB/s |
| `01/raw` | 512 | 41 GB/s |
| `debug/raw` | 128 | 40 GB/s |
| `debug/raw` | 256 | 40 GB/s |
| `debug/raw` | 512 | 41 GB/s |

这更不可能是 global memory 瓶颈。

## 7. 算术强度判断

按 global memory 摊销后：

$$
I\approx\frac{26\sim32}{6}=4.3\sim5.3\text{ FLOP/byte}.
$$

若显存带宽按 `1.8 TB/s` 量级估算，memory roofline 大约是：

$$
1.8\times(4.3\sim5.3)=7.7\sim9.5\text{ TFLOP/s}.
$$

而实际 `eval_B` 只有 `0.8-1.0 TFLOP/s`。这说明：

```text
eval_B 不是 global memory bound。
```

它更可能受限于：

- FP64 add/mul/FMA 吞吐；
- FP64 `rsqrt`/sqrt 特殊函数；
- shared memory 到寄存器的数据供给；
- warp 规约和指令调度。

但主要不是 global memory。

## 8. 当前真正瓶颈

### 8.1 `eval_B`

单独 `eval_B`：

```text
主要瓶颈：FP64 算力 / 特殊函数吞吐
次要瓶颈：shared memory 读带宽和 warp reduction
不是瓶颈：global memory bandwidth
```

### 8.2 `trace_period`

`trace_period(256 lines)`：

```text
主要瓶颈：并行度不足，只有 32 blocks
次要瓶颈：每个 B 的 FP64 算力 / rsqrt
不是瓶颈：global memory bandwidth
```

这解释了为什么 `eval_B` 有 `0.8-1.0 TFLOP/s`，但 `trace_period` 只有 `0.17-0.22 TFLOP/s`。

## 9. 优化方向

### 9.1 增大批量

最简单的办法是让一次 `trace_period` 包含更多线。

对于 GA 找轴，可以考虑：

```text
16x16 = 256 点 -> 32x32 = 1024 点
top16 -> top32
top32 两两中点 = 1024 点
```

这样 blocks 数从：

```text
256 / 8 = 32 blocks
```

提高到：

```text
1024 / 8 = 128 blocks
```

更接近充分占用 GPU。

代价是每代计算量增加 4 倍，但如果原来严重 under-occupied，实际时间可能不会增加 4 倍，甚至可能单位线效率明显提升。

### 9.2 调整 block/warp 映射

当前：

```text
8 warps/block
1 warp/line
```

候选方案：

| 方案 | 目的 | 风险 |
|---|---|---|
| `4 warps/block` | blocks 数翻倍，occupancy 提高 | global segment tile 复用从 8 降到 4 |
| `1 warp/block` | blocks 数提高到 256 | global load 重复增加 8 倍 |
| `1 block/line, 多 warp 协作一条线` | blocks 数提高，且一条线的 segment 求和更并行 | 需要跨 warp/block 内规约，global load 增加 |

当前最值得先测试：

```text
WARPS_PER_BLOCK = 4
```

因为改动小，可能在 `trace_period` 中提高 occupancy，同时仍保留一定 tile 复用。

### 9.3 多 warp 协作一条线

另一条更激进路线是：

```text
1 block = 1 line
block 内 4 或 8 warps 分摊 segment
```

优点：

- 256 lines 就有 256 blocks；
- 每条线的每次 $B$ 计算也更并行；
- 更适合小线数批量。

缺点：

- 同一个 segment tile 不再被 8 条线复用，global memory 压力增大；
- 需要 block 内跨 warp 规约；
- 对 `eval_B` 和 `trace` 可能需要不同 kernel。

考虑到当前 global bandwidth 很低，这个方案可能反而更快，值得作为第二个 kernel 变体实现。

### 9.4 mixed precision

当前全 double。由于 RTX 5090 FP64 吞吐有限，mixed precision 可能带来大幅提升。

可测试：

1. segment 坐标 double，计算 double；
2. segment 坐标 float，计算 float；
3. segment 坐标 float，局部累加 double；
4. fast inverse sqrt 近似。

必须用轴残差、轴位置、$\psi_0$ 筛选结果验证。不能只看速度。

## 10. 结论

当前 CUDA 原型的瓶颈判断如下：

1. `eval_B` 已经不是 global memory 瓶颈，而是 FP64 算力/特殊函数吞吐瓶颈。
2. `trace_period` 的主要瓶颈是并行度不足，当前 256 条线只产生 32 个 block，远远不够喂满 RTX 5090。
3. global memory 带宽利用率只有几十到一两百 GB/s，远低于硬件带宽上限。
4. 下一步优化优先级：
   - 增大 GA/筛选批量到 1024 条线；
   - 测试 `WARPS_PER_BLOCK=4`;
   - 实现 `multi-warp-per-line` kernel；
   - 再考虑 mixed precision。

