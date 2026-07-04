# GPU 追踪 kernel 变体实验报告

## 1. 问题

本轮实验回答三个问题：

1. 特殊函数瓶颈具体体现在哪里。
2. 减少一个 block 内的 warp 数、改成一个 block 负责一条磁力线、把批量扩大到 1024 条线，这几种方案哪个更有效。
3. tile 化 shared memory 是否会失去 shared memory 的意义，是否会让访存成为瓶颈。

实验在服务器 `master` 的单张 RTX 5090 上进行，运行时限制：

```text
CUDA_VISIBLE_DEVICES=0
segments_per_coil = 256
steps = 800
```

## 2. 特殊函数瓶颈

每个电流元贡献需要计算

$$
\frac{1}{|r|^3}.
$$

当前 CUDA 代码写法是：

```cpp
double invr = rsqrt(r2);
double invr3 = invr * invr * invr;
```

检查编译后的 SASS/PTX 后，可以看到以下指令：

```text
MUFU.RSQ64H
MUFU.RCP64H
DFMA
DMUL
DADD
```

这说明 double `rsqrt` / 倒数不是普通 FMA 指令，而是走了特殊函数/倒数近似再加 double 精度修正。它的具体影响是：

1. 每个 segment contribution 不只是普通加乘。
2. `rsqrt` 带来高延迟特殊函数指令。
3. 编译器会插入多条 `DFMA/DMUL` 做 double 精度修正。
4. 因此即使 global memory 带宽远未打满，kernel 仍会被 FP64 与特殊函数吞吐限制。

这个判断和 TFLOPs/带宽估算一致：

| kernel | 估计吞吐 | 估计 global bandwidth | 判断 |
|---|---:|---:|---|
| `eval_B` | 约 `0.8-1.0 TFLOP/s` | 约 `180-190 GB/s` | 更接近 FP64/特殊函数瓶颈 |
| `trace_period` | 约 `0.17-0.22 TFLOP/s` | 约 `39-41 GB/s` | 小批量下主要受 occupancy 限制 |

RTX 5090 的显存带宽上限远高于这里的 `180-190 GB/s`，所以目前不是 global memory bound。

## 3. tile 化 shared memory 的意义

tile 化没有失去 shared memory 的意义。

如果不用 shared memory，一个 block 内多个 warp 各自负责一条线时，每个 warp 都会从 global memory 读取同一批 segment。也就是说，同一个 segment 会被重复读取多次。

tile shared memory 的作用是：

```text
一个 block 先把一段 segment tile 从 global memory 读到 shared memory，
block 内多个 warp 或多个 thread 复用这段 tile，
然后再处理下一段 tile。
```

以 `WARPS_PER_BLOCK=8` 为例：

```text
每个 segment 的 global load = 6 doubles = 48 bytes
被 8 条线复用
摊到每个 line-segment contribution = 6 bytes
```

如果不用 shared memory：

```text
每个 line-segment contribution 约 48 bytes global load
```

所以 shared memory tile 理论上减少了约 8 倍 global load。

之所以必须 tile 化，是因为全部 segment 放不进 shared memory。对 `segments_per_coil=256`：

| case | 总 segment | 若一次性放入 shared memory |
|---|---:|---:|
| `01/raw` | 4096 | 约 192 KiB |
| `debug/raw` | 5120 | 约 240 KiB |

这对普通 block 不稳妥。tile 化后：

```text
SEG_TILE = 256
shared = 256 * 6 doubles = 12 KiB
```

这既能保持 segment 复用，又能控制 shared memory 占用。

不过，当前实测也说明：虽然 shared memory tile 有意义，但现在主要瓶颈不是 global memory。证据是把 `WARPS_PER_BLOCK` 从 8 降到 2 后，segment tile 在 block 内的复用变少，但小批量 GA 反而更快。这说明当前更缺的是足够多的 active blocks，而不是 global memory 带宽。

## 4. 方案 A：减少每个 block 的 warp 数

原始 warp-per-line kernel 是：

```text
一个 warp 负责一条磁力线
一个 block 负责 WARPS_PER_BLOCK 条磁力线
```

测试了 `WARPS_PER_BLOCK=2,4,8`。

### 256 条线批量

| case | W=2 | W=4 | W=8 |
|---|---:|---:|---:|
| `01/raw` 总时间 | 2.808 s | 2.818 s | 5.012 s |
| `01/raw` 每代中位数 | 0.280 s | 0.281 s | 0.501 s |
| `debug/raw` 总时间 | 6.278 s | 6.299 s | 11.211 s |
| `debug/raw` 每代中位数 | 0.348 s | 0.349 s | 0.622 s |

结论：对 256 条线这种小批量，`W=2` 或 `W=4` 明显优于 `W=8`，接近 2 倍加速。

原因是 `W=8` 时 block 数太少：

```text
256 lines / 8 warps per block = 32 blocks
```

这不足以喂满 GPU。改成 `W=2` 后：

```text
256 lines / 2 warps per block = 128 blocks
```

active blocks 更多，能更好地隐藏 `rsqrt` 和 FP64 指令延迟。

### 1024 条线批量

| case | W=2 | W=4 | W=8 |
|---|---:|---:|---:|
| `01/raw` 总时间 | 10.431 s | 10.147 s | 10.037 s |
| `01/raw` 每代中位数 | 0.521 s | 0.507 s | 0.501 s |
| `debug/raw` 总时间 | 8.436 s | 8.200 s | 8.107 s |
| `debug/raw` 每代中位数 | 0.648 s | 0.630 s | 0.622 s |

结论：批量扩大到 1024 后，`W=8` 不再吃亏，甚至略好。这是因为 block 数已经足够：

```text
1024 lines / 8 warps per block = 128 blocks
```

同时 `W=8` 的 shared tile 复用更强。

## 5. 方案 B：一个 block 负责一条磁力线

新增 `trace_period_blockline`：

```text
一个 block 负责一条磁力线
一个 block 内 256 或 512 个 thread 并行累加所有 segment contribution
每一步 RK4 的每次 B 评估都做 block 内规约
```

### 256 条线批量

| case | block 256 threads | block 512 threads |
|---|---:|---:|
| `01/raw` 总时间 | 2.715 s | 3.730 s |
| `01/raw` 每代中位数 | 0.150 s | 0.207 s |
| `debug/raw` 总时间 | 3.455 s | 4.661 s |
| `debug/raw` 每代中位数 | 0.181 s | 0.245 s |

结论：256 threads/block 最好，是目前 256 条线 GA 任务的推荐方案。

它比 `W=2` warp-per-line 更快，主要因为一条线内部的 segment 求和也被并行化了。对 4096/5120 个 segment，这个并行度是有价值的。

512 threads/block 反而更慢，说明额外 thread、规约和调度成本超过了进一步拆分 segment 的收益。

### 1024 条线批量

| case | block 256 threads | block 512 threads |
|---|---:|---:|
| `01/raw` 总时间 | 5.123 s | 7.226 s |
| `debug/raw` 总时间 | 6.193 s | 8.559 s |

对 1024 条线，block-per-line 仍然能跑通，但不一定比 256 条线更划算，因为遗传算法本身每代候选更多，可能需要更少代也可能不需要。实际终止时间取决于收敛代数。

## 6. 方案 C：增大批量到 1024

增大批量的好处是 occupancy 变好，尤其对 `W=8` 这种每 block 多 warp 的设计明显。

但对当前遗传算法，批量从 256 增到 1024 不一定减少总时间，因为：

1. 每代计算量约增加 4 倍。
2. 收敛代数不一定下降 4 倍。
3. 当前 top-k 交叉策略本来就是 `top16 x top16 = 256`，扩到 1024 需要改变遗传策略，否则只是额外采样。

本轮结果中，1024 批量对 `debug/raw` 有一定好处：

```text
debug, W=8:
256 batch: 11.211 s
1024 batch: 8.107 s
```

但对 `01/raw`：

```text
01, W=8:
256 batch: 5.012 s
1024 batch: 10.037 s
```

所以 1024 批量适合作为可选策略：当候选点很多、需要大范围筛选、或者要并行追踪很多 psi0/fieldline 时有价值；对当前 256 候选 GA，默认不应强制扩大到 1024。

## 7. 推荐配置

当前推荐：

| 场景 | 推荐 |
|---|---|
| 256 条线 GA 找磁轴 | `trace_period_blockline`, 256 threads/block |
| warp-per-line 小批量备用 | `WARPS_PER_BLOCK=2` 或 `4` |
| 大批量追踪，约 1024 条线以上 | `WARPS_PER_BLOCK=8` 可以重新变得合理 |
| 512 threads/block | 暂不推荐 |

用推荐的 block-per-line 256 threads 配置，GA 找轴时间：

| case | 收敛代数 | 最小残差 | 总时间 |
|---|---:|---:|---:|
| `01/raw` | 17 | `8.60e-10` | 2.715 s |
| `debug/raw` | 18 | `5.24e-09` | 3.455 s |

它和 CPU/simsopt 找到的磁轴位置一致到约 $10^{-9}$ 量级，同时显著快于 CPU/simsopt：

| case | CPU/simsopt | GPU blockline |
|---|---:|---:|
| `01/raw` | 72.22 s | 2.715 s |
| `debug/raw` | 104.02 s | 3.455 s |

## 8. 后续优化方向

1. 对 `trace_period_blockline` 做 Nsight Compute 细分，确认 occupancy、register pressure、shared memory bank conflict 和 FP64 指令占比。
2. 尝试近似/混合精度：例如先用 float 或近似 rsqrt 做粗筛，再用 double 精修。
3. 对 segment 数据做 coil/tile 层级组织，让缓存和 shared memory 复用更可控。
4. 对 GA 策略本身优化：避免 1024 批量只是增加无效候选，改成分阶段大批量粗筛、小批量精修。
5. 如果后续追踪很多 psi0 或很多磁面点，优先用大批量追踪摊薄 kernel launch 和 occupancy 问题。

