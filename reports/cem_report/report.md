# CEM1 / CEM2 磁面评估报告

生成日期：2026-07-08

本报告是整理后的交付入口。原始计算目录仍保留用于溯源，但主要结论、关键指标和需要查看的图片都集中在这里。交互式 3D 图以 HTML 链接给出。

## 1. 固化后的评估流程

已将流程固化为 Codex skill：

- `C:\Users\FanCY\.codex\skills\stellarator-surface-eval\SKILL.md`

流程包括：

1. 按默认 score 跑完整评估，输出 axis、$\psi$、screen、Boozer、QS、工程项和细粒度耗时。
2. 扫描不同 $a$ 和 $\psi$，以 Boozer LS/Newton 是否成功作为最大磁面存在性的最终判据。
3. 对找到的最大磁面生成直接的 $|B|$ 图和全装置线圈+磁面 HTML。
4. 把最大磁面转成 DESC 输入，做零压、零电流平衡求解和 DESC 图件。
5. 对分支跳变、DESC 非嵌套警告、DESC force residual 异常等情况做显式标注。

## 2. CEM1 结果

### 2.1 最大磁面

| 指标 | 数值 |
| --- | ---: |
| 采用半径参数 $a$ | `0.25` |
| 最大接受 $\psi$ level | `0.048` |
| volume | `0.2009300495` |
| $\iota$ | `-0.4876927678` |
| $G$ | `2.3559579919` |
| QS error QA | `0.3171422963` |
| QS error QH | `0.3085128291` |
| QS error QP | `0.0011661789` |
| $|B|_\min$ | `0.1644154899 T` |
| $|B|_\mathrm{mean}$ | `0.4988087972 T` |
| $|B|_\max$ | `0.8473320606 T` |

交互式图：

- [CEM1 全装置线圈+磁面 HTML](assets/cem1_coils_surface.html)
- [CEM1 Boozer 面 $|B|$ HTML](assets/cem1_boozer_b.html)

### 2.2 $|B|$ 分布

![CEM1 Boozer 面 |B|](assets/cem1_boozer_b.png)

### 2.3 DESC 复核

DESC 输入的 `PHIEDGE = -9.3483351503e-03`。DESC 求解返回成功，但该例出现过非嵌套/坐标修正警告，且 force residual 偏大，因此这里把 DESC 结果视为诊断图，不把它作为比 Simsopt/Boozer 更可信的最终物理量。

![CEM1 DESC boundary](assets/cem1_desc_boundary.png)

![CEM1 DESC Boozer B modes](assets/cem1_desc_b_modes.png)

![CEM1 DESC Boozer B contours](assets/cem1_desc_b_contours.png)

![CEM1 DESC QS error QA](assets/cem1_desc_qs_qa.png)

![CEM1 DESC QS error QH](assets/cem1_desc_qs_qh.png)

![CEM1 DESC QS error QP](assets/cem1_desc_qs_qp.png)

![CEM1 DESC iota](assets/cem1_desc_iota.png)

## 3. CEM2 结果

### 3.1 默认 score 评估

默认流程先按当前 evaluator 默认参数运行，结果如下：

| 指标 | 数值 |
| --- | ---: |
| quality score | `91.2696890764` |
| status | `surface` |
| axis residual | `1.4212876125e-08` |
| 默认最大 $\psi$ level | `0.16` |
| 默认 volume | `0.0137636937` |
| 默认 $\iota$ | `0.0362285694` |
| 默认 $G$ | `2.1757930772` |
| 默认 QS error QA | `0.2634040835` |
| 默认 QS error QH | `0.2571305475` |
| 默认 QS error QP | `6.0742958156e-05` |
| 默认总耗时 | `2.3217832204 s` |

### 3.2 扩展 $a$ 和 $\psi$ 后的最大磁面

扩展搜索中发现，直接增大 $a$ 会让 cheap screen 的绝对漂移阈值过早卡住；放宽 screen 后，Boozer 能继续求解更大面。边界细扫显示最大几何体积出现在 $a=0.111,\ \psi=0.18$。继续增大 $a$ 后可接受的 $\psi$ level 下降，体积反而变小。

| 指标 | 数值 |
| --- | ---: |
| 采用半径参数 $a$ | `0.111` |
| 最大接受 $\psi$ level | `0.18` |
| quality score | `81.4147832850` |
| volume | `0.0826110657` |
| $\iota$ | `0.2110181488` |
| $G$ | `2.1757950735` |
| QS error QA | `0.2691050477` |
| QS error QH | `0.2581974026` |
| QS error QP | `0.0021870195` |
| $|B|_\min$ | `0.1417607435 T` |
| $|B|_\mathrm{mean}$ | `0.4043352124 T` |
| $|B|_\max$ | `0.6792719434 T` |

交互式图：

- [CEM2 全装置线圈+磁面 HTML](assets/cem2_coils_surface.html)
- [CEM2 Boozer 面 $|B|$ HTML](assets/cem2_boozer_b.html)

### 3.3 $|B|$ 分布

![CEM2 Boozer 面 |B|](assets/cem2_boozer_b.png)

### 3.4 DESC 复核

DESC 输入的 `PHIEDGE = -3.0865581843e-03`。DESC 求解成功，收敛情况明显好于 CEM1：

| DESC 指标 | 数值 |
| --- | ---: |
| setup time | `14.2067656713 s` |
| solve time | `27.8728458257 s` |
| solve status | `success` |
| 终止条件 | `ftol condition satisfied` |
| 最大归一化 force error | 约 `5.993e-05` |
| 平均归一化 force error | 约 `2.585e-06` |

![CEM2 DESC boundary](assets/cem2_desc_boundary.png)

![CEM2 DESC Boozer B modes](assets/cem2_desc_b_modes.png)

![CEM2 DESC Boozer B contours](assets/cem2_desc_b_contours.png)

![CEM2 DESC QS error QA](assets/cem2_desc_qs_qa.png)

![CEM2 DESC QS error QH](assets/cem2_desc_qs_qh.png)

![CEM2 DESC QS error QP](assets/cem2_desc_qs_qp.png)

![CEM2 DESC iota](assets/cem2_desc_iota.png)

## 4. 简要结论

- CEM1 可以找到很大的 Boozer 可解面，但 DESC 对这个边界的平衡复核不够干净；CEM1 的 DESC 图主要用于诊断。
- CEM2 默认 score 很高，默认面较小但 QP QS error 很低。
- CEM2 扩展后可得到显著更大磁面：volume 从 `0.01376` 增至 `0.08261`，但 QP QS error 从 `6.07e-05` 增至 `2.19e-03`。
- CEM2 的最大面 DESC 复核较干净，适合作为目前更可信的“线圈到最大磁面再到 DESC”的完整示例。
