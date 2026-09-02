# Compact-flexible v3 解析先验 Adam200 可优化性报告

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-02 |
| 实验协议 | `qh-axis-surface-contour-compact-flexible-random-ok-adam200-64d-abi11-v1` |
| 实验代码 | `08a3c3fd15e61232eb0eed7cc359679c8d471810` |
| 报告绘图代码 | `e4590aebb7d06d6ad9e24f3e1d3dee2b722f3939` |

## 2026-09-02 手性审计更正

本报告的轨迹数量、ABI-11 分数、分量、运行时间和完整物理评估数值均保留。其
QH 解释需要增加符号限定：ABI-11 在本实验中固定评估 `(M,N)=(1,+nfp)`，完整
评估表中的 `QH` 也只表示 `QH_(1,+1)`。四个已完整评估的 balanced-v2 与
compact-flexible-v3 高分端点都具有负 `iota`，其 Boozer `|B|` 主导模对应
`theta+zeta`，与旧 QUASR/Flow 高分样本的 `theta-zeta` 分支手性相反。

因此，本报告中把达到 50 的比例直接解释为与旧 QUASR/Flow 同一手性 QH 收敛域
丰度的表述已被撤回。该比例目前仅是冻结的固定正手性 ABI-11 目标下的分数阈值
统计。嵌套曲面、体积、Poincare、线圈工程和 DESC 数值仍然有效；先验的 QH 丰度
以及约 70 分平台的成因，等待保存端点在 `N=+nfp`、`N=-nfp` 下的成对重评。
详见 `memory/CORRECTIONS.md` 的 `CORR-20260902-66`。

## 结论

Compact-flexible v3 解析先验提供了高条件可优化率的 QH 起点。120 条完整 Adam200
轨迹中，97 条的历史最佳 ABI-11 分数达到 50，合法起点内的条件成功率为 80.83%，
Wilson 95% 区间为 72.88%--86.87%。将该比例乘以 3600 个源样本测得的 15.81%
初始合法率，得到全先验 Adam200 可达丰度估计 12.78%。

Adam50 已经发现 65/120 条最终可优化起点，合法起点内的条件成功率为 54.17%，
Wilson 95% 区间为 45.26%--62.81%。折算到整个 compact-flexible v3 先验后，
`best-so-far@50 >= 50` 的估计丰度为 8.56%。按三分类口径估计，C0 初始不合法占
84.19%，C1 初始合法且 Adam50 未达到 50 占 7.24%，C2 初始合法且 Adam50 达到
50 占 8.56%。

优化结果形成明显的双群结构。97 条成功轨迹集中在约 60--69 分平台；18 条轨迹的
最佳分数低于 20，另有 5 条位于 `[20,50)`。全批次最佳分数为 68.8804，没有样本
达到 70。该结果证明 compact-flexible v3 中的合法起点普遍易于进入已知高分平台，
同时没有给出更高分收敛域的证据。QUASR 聚类归属、标准磁面和 MHD 平衡仍待后续
专项评估。

与 balanced-v2 的同口径实验相比，compact-flexible v3 的初始合法率从 37.50%
下降到 15.81%，合法起点的 Adam50 成功率从 15.66% 上升到 54.17%。两项因素合并
后，全先验 Adam50 可达丰度从 5.87% 上升到 8.56%。Adam200 的全先验估计从
14.01% 变为 12.78%。Compact-flexible v3 更适合 50 步快速奖励标签；200 步总可达
丰度与 balanced-v2 接近。

## 实验设计

源协议 `qh-axis-surface-contour-compact-flexible-score-abi11-v3` 从随机磁轴、紧凑绕组
面和随机标量场等值线构造线圈。绕组面小半径集中在 0.20 m，采样范围为
`[0.18,0.22] m`；基础线圈数限制为 `nc<=4`。源评分阶段在 26 个 `(nfp,nc)` 条件
上生成 3600 个样本，每个条件含 138 或 139 个样本。

原生 ABI-11 评分器认定 569 个初始状态合法，合法率为 15.8056%。Adam200 实验使用
固定种子 `20260904`，从这 569 个状态中无放回均匀抽取 120 个样本。入选样本按
`nc=1,2,3,4` 的数量分别为 `48,31,23,18`。运行时间权重用于六卡负载均衡，每卡
分配 20 条轨迹。

每个起点使用精确、无裁剪的标准化线圈系数进入 direct-data Adam。标准化尺度来自
冻结 Flow checkpoint，优化过程中不调用 Flow 模型。每条轨迹使用下列设置：

| 参数 | 值 |
|---|---:|
| 参数空间 | exact-unclipped standardized coil coefficients |
| Adam 更新数 | 200 |
| 每步方向数 | 64 个新鲜随机正交方向 |
| 差分规则 | centered |
| 差分扰动 `h` | 0.0025 |
| 学习率 | 0.01 |
| beta | (0.7, 0.999) |
| 评分器 | native ABI-11 |
| 成功阈值 | 历史最佳分数达到 50 |

评分库 SHA-256 为
`1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed`；checkpoint
SHA-256 为
`39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f`。

## 源先验评分

六卡评分数组 `52244/52245` 完成 3600/3600 条 ABI-11 评分，运行错误为 0；分析
作业 `52246` 生成冻结汇总。状态分布为：

| ABI-11 状态 | 数量 | 比例 |
|---|---:|---:|
| `ok` | 569 | 15.81% |
| `no_axis` | 1982 | 55.06% |
| `no_surface` | 760 | 21.11% |
| `drift_rejected` | 267 | 7.42% |
| `flux_rejected` | 22 | 0.61% |

源样本总分中位数为 0.1247，p90 为 5.5715，最大值为 46.6289。coil 工程分量中位数
为 72.0143，p10/p90 为 64.0936/78.9310，最大值为 84.5682。该分布显示紧凑柔性
先验保持了较高的初始线圈工程分量，同时大量样本停在磁轴和可用曲面建立阶段。

![Compact-flexible v3 的源 QH 分数和线圈工程分量](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/source_scoring/score_and_coil_distributions.png)

图 1：协议 `qh-axis-surface-contour-compact-flexible-score-abi11-v3` 的 3600 个源样本。
左列给出 ABI-11 总分分布和生存曲线，右列给出 coil 工程分量及 coil 分量与总分的关系。

## 作业验收

准备作业 `52313` 冻结 120 条起点、源行哈希、起点哈希和六卡分配。三步 smoke
作业 `52314` 的起点 roundtrip 相对误差为 0；冻结源分数与优化器独立重评估相差
0.0956，处于 0.1 验收容差内；三步最佳分数为 5.4215。

正式计算使用 P107 四卡数组 `52315` 和 Students 两卡数组 `52316`。六个 worker
均通过 GPU 空闲 preflight，每个 worker 完成分配的 20 条轨迹，停止原因为
`all_assigned_cases_complete`。分析作业 `52317` 生成逐轨迹 CSV、汇总 JSON 和原始
轨迹图。

| 验收项 | 结果 |
|---|---:|
| 计划抽样 | 120 |
| 完整 Adam200 | 120 |
| 完成率 | 100% |
| `status=ok, completed_iterations=200` | 120 |
| 运行失败 | 0 |
| incomplete 目录 | 0 |
| 未启动样本 | 0 |
| 六个 worker 墙钟范围 | 5.31--5.47 h |

## 总体结果

| 指标 | 初始分数 | Adam200 历史最佳 |
|---|---:|---:|
| p10 | 2.9259 | 10.8136 |
| 中位数 | 6.7697 | 65.3774 |
| p90 | 20.3150 | 66.6107 |
| 最大值 | 38.8484 | 68.8804 |

最佳分数低于 20 的轨迹为 18/120，位于 `[20,50)` 的轨迹为 5/120，达到 50 的
轨迹为 97/120。第 200 步当下有 96 条轨迹保持在 50 以上；1 条轨迹曾经跨过 50，
随后在第 200 步回落。

![Compact-flexible v3 的初始分数、Adam200 最佳分数和生存曲线](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/analysis/score_distribution.png)

图 2：120 条完整轨迹的初始 ABI-11 分数和 Adam200 历史最佳分数。绿色虚线为
50 分成功阈值；背景区间对应低于 20、`[20,50)` 和至少 50 三个分段。

## 收敛时间结构

| 更新上限 | 累计达到 50 | 占 120 条完整轨迹 | 占最终 97 条成功轨迹 |
|---:|---:|---:|---:|
| 25 | 44 | 36.67% | 45.36% |
| 50 | 65 | 54.17% | 67.01% |
| 100 | 89 | 74.17% | 91.75% |
| 150 | 95 | 79.17% | 97.94% |
| 200 | 97 | 80.83% | 100% |

Adam100 已覆盖 89/97 个最终成功起点。8 条成功轨迹在第 100 步后首次跨过 50，
其中 2 条发生在第 150 步后。第 50 步当下有 64 条轨迹保持在 50 以上，历史最佳
口径为 65 条。

![Compact-flexible v3 的 120 条 Adam200 轨迹与首次跨过 50 分的时间](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/analysis/trajectory_curves.png)

图 3：左图显示全部 120 条 ABI-11 轨迹，颜色标识基础线圈数 `nc`；右图显示各
`nc` 组首次达到 50 分的累计比例。曲线来自每条轨迹的冻结 200 步 history。

## `nc` 和 `nfp` 分层

| nc | 样本 | 初始合法率 | Adam50 达到 50 | Adam200 达到 50 | Adam200 条件成功率 | 全先验 Adam200 估计 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 30.94% | 8 | 31 | 64.58% | 19.98% |
| 2 | 31 | 17.06% | 21 | 26 | 83.87% | 14.31% |
| 3 | 23 | 10.66% | 19 | 23 | 100.00% | 10.66% |
| 4 | 18 | 8.80% | 17 | 17 | 94.44% | 8.31% |

`nc=3` 和 `nc=4` 的合法起点通常在前 50 步进入高分段；`nc=1` 的收敛更慢，前
50 步成功 8/48，200 步成功 31/48。`nc=1` 较高的初始合法率使其全先验 Adam200
估计仍为四组最高。

| nfp | 样本 | Adam50 达到 50 | Adam200 达到 50 | Adam200 最佳分数中位数 |
|---:|---:|---:|---:|---:|
| 4 | 2 | 0 | 0 | 1.5740 |
| 5 | 9 | 0 | 6 | 65.4958 |
| 6 | 18 | 5 | 13 | 64.8752 |
| 7 | 43 | 25 | 35 | 65.5315 |
| 8 | 48 | 35 | 43 | 65.4327 |

`nfp=4` 只有 2 个入选合法样本，0/2 结果的统计不确定性很大。`nc` 和 `nfp` 在
120 条样本中具有联合分布，分层差异用于确定后续采样重点，不能单独解释为某一参数
的因果效应。

![按 nc 和 nfp 分层的 Adam200 条件成功率](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/analysis/success_rates.png)

图 4：点为各组 `best-so-far@200 >= 50` 的条件成功率，误差条为 Wilson 95%
区间，文字给出成功数/完整轨迹数。

## 计算成本和最佳点位置

单条 Adam200 墙钟中位数为 916.3 秒，p90 为 1567.9 秒，最大值为 1773.7 秒，
对应 15.27、26.13 和 29.56 分钟。按 `nc=1,2,3,4` 分组的单条中位墙钟分别为
10.26、15.40、20.66 和 27.81 分钟。六卡成本均衡使 worker 墙钟差保持在约
9.7 分钟内，120 条轨迹在 7 小时预算中全部完成。

![按 nc 的运行成本和最佳分数出现位置](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/analysis/runtime_and_convergence.png)

图 5：左图给出各 `nc` 组单条 Adam200 墙钟分布；右图给出历史最佳分数出现的
更新步，颜色标识 `nc`。成功轨迹的最佳点广泛分布在整个 200 步区间。

## 高分端

最高分的五条轨迹为：

| 样本 | nfp | nc | 初始分数 | 最佳分数 | 最佳步 | 第 200 步分数 |
|---|---:|---:|---:|---:|---:|---:|
| `axisv3_case_01341` | 5 | 3 | 2.5825 | 68.8804 | 198 | 68.0263 |
| `axisv3_case_02537` | 5 | 3 | 2.5810 | 68.2373 | 151 | 67.7988 |
| `axisv3_case_02591` | 7 | 3 | 15.0478 | 67.6465 | 174 | 67.4182 |
| `axisv3_case_02832` | 7 | 4 | 20.2231 | 67.2446 | 191 | 67.2185 |
| `axisv3_case_01186` | 6 | 3 | 6.5171 | 67.0585 | 198 | 67.0324 |

97 条成功轨迹的初始总分中位数为 7.3694，23 条未达到 50 的轨迹为 4.7992。
成功组的初始 coil 分量中位数为 67.7640，失败组为 76.0960。较高的初始工程分量
没有单独预测 QH 可达性；`nc`、`nfp` 和几何结构共同影响优化结果。

本批次的最佳分数低于 balanced-v2 的 69.6456，两个先验都形成约 60--70 分平台。
高分端的 QUASR 聚类新颖性仍待分析；本报告末尾已补充两个代表性样本的完整物理评估。

## 与 balanced-v2 的同口径比较

两个实验都使用 ABI-11、exact-unclipped direct-data Adam、200 步、64 个新鲜正交
方向、`h=0.0025`、学习率 0.01 和 beta `(0.7,0.999)`。下表比较完整配方的观测
结果；compact-flexible v3 同时改变了绕组面半径和多个形状扰动范围，因此该比较不
分离单一生成参数的贡献。

| 指标 | Balanced-v2 | Compact-flexible v3 |
|---|---:|---:|
| 源样本 | 6000 | 3600 |
| 初始合法率 | 37.50% | 15.81% |
| 完整 Adam200 | 83 | 120 |
| 合法起点 Adam50 成功率 | 15.66% | 54.17% |
| 全先验 Adam50 估计丰度 | 5.87% | 8.56% |
| 合法起点 Adam200 成功率 | 37.35% | 80.83% |
| 全先验 Adam200 估计丰度 | 14.01% | 12.78% |
| Adam200 最佳分数中位数 | 16.9956 | 65.3774 |
| Adam200 最大分数 | 69.6456 | 68.8804 |

Compact-flexible v3 把合法起点内的收敛时间显著前移，并提高 Adam50 的全先验
丰度。较低的初始合法率抵消了部分条件成功率增益，使 Adam200 全先验丰度保持在与
balanced-v2 接近的水平。后续先验改进应优先提升磁轴和初始曲面建立成功率，同时
保持 compact-flexible v3 的快速可优化特征。

## 证据边界和后续用途

1. 97 个高分状态中的 95 个仍只有 ABI-11 筛选证据。`axisv3_case_01341` 和
   `axisv3_case_02832` 已完成标准 Simsopt 曲面、独立稠密残差、Poincare、Boozer
   `|B|` 和 DESC 评估，结果见报告末尾的补充评估。
2. `best-so-far@50 >= 50` 的 8.56% 全先验估计可作为后续 RL 起步阶段的 C2 丰度。
   同口径 C0/C1/C2 估计为 84.19%/7.24%/8.56%。
3. Adam100 已覆盖 91.75% 的最终成功轨迹，可作为精度和计算成本之间的主要审计点。
4. `nc=3/4` 适合快速获得 C2，`nc=1` 提供较高初始合法率和更慢的收敛轨迹。训练与
   评估应固定或显式报告 `nc/nfp` 组成，防止条件分布变化被解释为策略改进。
5. `axisv3_case_01341` 是本批次总分最高样本；`axisv3_case_02832` 是 `nc=4` 的
   高分代表。两者已完成完整物理评估，后续仍可进入 QUASR 聚类新颖性分析。

该实验属于解析先验探索。项目默认协议保持
`qh-flow-screen32-adam200-64d-abi11-v1`。

## 冻结证据

- [实验协议](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/protocol.json)
- [运行清单脱敏副本](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/runtime_manifest.json)
- [抽样、哈希和六卡分配清单脱敏副本](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/selection_manifest.json)
- [清单脱敏记录与原始 SHA-256](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/manifest_sanitization.json)
- [Smoke 结果](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/smoke_result.json)
- [Adam200 汇总](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/analysis/summary.json)
- [逐轨迹表](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/analysis/trajectories.csv)
- [首次跨阈值统计](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/analysis/first_passage_summary.json)
- [原始自动分析轨迹图](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/analysis/adam200_trajectories.png)
- [源评分汇总](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/source_scoring/summary.json)
- [六个 worker 完成记录](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/workers/)

本报告中的源评分数值来自代码提交 `3f901d5b5431fce50823749613a48bef368eb7fc`；
Adam200 轨迹来自代码提交 `08a3c3fd15e61232eb0eed7cc359679c8d471810`。报告绘图
脚本从 120 条冻结 history 重新计算首次跨阈值统计，生成代码提交为
`e4590aebb7d06d6ad9e24f3e1d3dee2b722f3939`。

## 补充完整物理评估（2026-09-02）

完整评估使用两条 Adam200 轨迹各自的历史最佳状态：本批次总分最高的
`axisv3_case_01341`（`nfp=5, nc=3`）和最高分 `nc=4` 样本
`axisv3_case_02832`（`nfp=7`）。两例先在 `a=0.04,0.05,0.06,0.08` 上独立拟合
psi，再并行测试 `s=0.12,0.24,0.36,0.49,0.64,0.81`。两例均选择覆盖物理半径
最大的 `a=0.08`，最大曲面由标准 Simsopt LS/Newton、97×97 独立稠密网格和
外侧失败边界共同确定。

| 样本 | Adam200 最佳分数 | `a` | 最大接受 `s` | 最近外侧失败 `s` | `iota` | `G` | 体积 (m³) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `axisv3_case_01341` | 68.8804 | 0.08 | 0.24 | 0.36 | -1.02289 | 6.28319 | 0.026633 |
| `axisv3_case_02832` | 67.2446 | 0.08 | 0.36 | 0.49 | -1.17462 | 6.28319 | 0.046051 |

`axisv3_case_01341` 的 `s=0.36` 候选完成 Newton，但独立稠密相对 L2 和法向场
p95 未通过；`axisv3_case_02832` 的 `s=0.49` 候选完成 Newton且稠密相对 L2
通过，法向场 p95 未通过。两例的最大接受曲面均有紧邻的外侧失败点。

| 样本 | 稠密相对 L2 | 法向场比 p95 | 直接 QA | 直接 QH `(1,+1)` | 直接 QP | `|B|` 范围 (T) | Poincare hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| `axisv3_case_01341` | 8.27e-5 | 8.03e-5 | 0.00434 | 0.00919 | 0.00458 | 0.7425--0.9665 | 8×55 |
| `axisv3_case_02832` | 4.29e-5 | 6.20e-5 | 0.00953 | 0.00973 | 0.00994 | 0.7102--1.0021 | 8×103 |

### `axisv3_case_01341`

| 线圈与最大接受曲面 | Poincare 截面 | 直接 Boozer `|B|` |
|---|---|---|
| ![axisv3_case_01341 的线圈与 s=0.24 最大接受曲面](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/assets/coils_surface.png) | ![axisv3_case_01341 的四个 Poincare 截面](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/assets/poincare.png) | ![axisv3_case_01341 最大接受曲面的 Boozer 磁场强度](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/assets/boozer_b.png) |

图 6：`axisv3_case_01341` 的完整装置几何、四个 Poincare 截面和 `s=0.24` 标准
Boozer 曲面上的磁场强度。Poincare 点在四个截面上保持径向有序；`iota` 接近 -1，
有限跟踪长度形成弧段而非稠密闭合点列。交互文件：
[线圈与曲面 HTML](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/assets/coils_surface.html)、
[Boozer `|B|` HTML](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/assets/boozer_b.html)。

| DESC 初始边界 | DESC 最终边界 |
|---|---|
| ![axisv3_case_01341 的 DESC 初始边界](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/desc/boundary_initial.png) | ![axisv3_case_01341 的 DESC 最终边界](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/desc/boundary.png) |

| DESC Boozer 模谱 | DESC Boozer `|B|` |
|---|---|
| ![axisv3_case_01341 的 DESC Boozer 模谱](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/desc/boozer_modes.png) | ![axisv3_case_01341 的 DESC Boozer 磁场强度](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/desc/boozer_B.png) |

| DESC QA 误差 | DESC QH 误差 | DESC QP 误差 |
|---|---|---|
| ![axisv3_case_01341 的 DESC QA 误差](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/desc/qs_QA.png) | ![axisv3_case_01341 的 DESC QH 误差](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/desc/qs_QH.png) | ![axisv3_case_01341 的 DESC QP 误差](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/desc/qs_QP.png) |

![axisv3_case_01341 的 DESC iota 剖面](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/desc/iota.png)

图 7：`axisv3_case_01341` 的 DESC 诊断。初始与最终边界均通过嵌套检查；平均
归一化力残差由 0.8988 降至 0.002701，p95 由 2.2521 降至 0.006225，分别降低
333 倍和 362 倍。优化器在第 50 次迭代达到上限，最终 cost 为 4.79e-4，
`optimizer_success=false` 保留为收敛边界。

### `axisv3_case_02832`

| 线圈与最大接受曲面 | Poincare 截面 | 直接 Boozer `|B|` |
|---|---|---|
| ![axisv3_case_02832 的线圈与 s=0.36 最大接受曲面](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/assets/coils_surface.png) | ![axisv3_case_02832 的四个 Poincare 截面](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/assets/poincare.png) | ![axisv3_case_02832 最大接受曲面的 Boozer 磁场强度](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/assets/boozer_b.png) |

图 8：`axisv3_case_02832` 的完整装置几何、四个 Poincare 截面和 `s=0.36` 标准
Boozer 曲面上的磁场强度。Poincare 点列呈现明显的多叶/岛链式结构。该图支持已选
外边界的标准曲面解，同时把物理结论限定在该外边界；内部真空场应按含显著共振结构
解释。交互文件：
[线圈与曲面 HTML](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/assets/coils_surface.html)、
[Boozer `|B|` HTML](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/assets/boozer_b.html)。

| DESC 初始边界 | DESC 最终边界 |
|---|---|
| ![axisv3_case_02832 的 DESC 初始边界](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/desc/boundary_initial.png) | ![axisv3_case_02832 的 DESC 最终边界](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/desc/boundary.png) |

| DESC Boozer 模谱 | DESC Boozer `|B|` |
|---|---|
| ![axisv3_case_02832 的 DESC Boozer 模谱](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/desc/boozer_modes.png) | ![axisv3_case_02832 的 DESC Boozer 磁场强度](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/desc/boozer_B.png) |

| DESC QA 误差 | DESC QH 误差 | DESC QP 误差 |
|---|---|---|
| ![axisv3_case_02832 的 DESC QA 误差](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/desc/qs_QA.png) | ![axisv3_case_02832 的 DESC QH 误差](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/desc/qs_QH.png) | ![axisv3_case_02832 的 DESC QP 误差](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/desc/qs_QP.png) |

![axisv3_case_02832 的 DESC iota 剖面](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/desc/iota.png)

图 9：`axisv3_case_02832` 的 DESC 诊断。初始与最终边界均通过嵌套检查；平均
归一化力残差由 1.2965 降至 0.001456，p95 由 2.6131 降至 0.003457，分别降低
891 倍和 756 倍。优化器在第 50 次迭代达到上限，最终 cost 为 4.04e-6，
`optimizer_success=false` 保留为收敛边界。DESC 的边界嵌套检查针对构造的平衡态；
真空场 Poincare 图显示的内部共振结构仍是该样本的物理限制。

两例都证明 compact-flexible v3 的 Adam200 高分状态可以包含标准 Boozer 可解外边界，
直接曲面 QH 误差均低于 0.01。两例最大接受体积为 0.0266 和 0.0461 m³，线圈仍有
高复杂度；`axisv3_case_02832` 还显示内部共振结构。完整评估因此支持该先验进入后续
形态与聚类研究，同时给出线圈工程和内部磁面质量的明确改进方向。

评估流程提交 `e4590aebb7d06d6ad9e24f3e1d3dee2b722f3939`，GPU 库由提交
`89206f4278155bc1c3b06db8bbe59aaea0e3ae9e` 构建，SHA-256 为
`23158593e57cd82300aa8d2efb2ee3023662d7f9d1765d1cfa22c84f26434af0`。两个输入
SHA-256 分别为
`399d3d197cb1e527043ea6d0798d28d342dedb0498519fd103f85fb9f2a19606` 和
`9e412f20b037490e264c65bf369a8f28857ca4fba023225347e004511c161b54`。
source-psi 与 `s` 候选跨六张 GPU 并行；下游首先同时提交到 P107，实际 16 CPU
QOS 上限只允许一例启动，未启动的一例改投 Students CPU 后端，两例随后并行完成。

- [完整评估冻结清单](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/manifest.json)
- [`axisv3_case_01341` 曲面选择](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/selection.json)
- [`axisv3_case_01341` 完整评估汇总](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_01341/full/full_summary.json)
- [`axisv3_case_02832` 曲面选择](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/selection.json)
- [`axisv3_case_02832` 完整评估汇总](assets/axis_surface_prior_compact_flexible_v3_adam200_20260902/full_eval/axisv3_case_02832/full/full_summary.json)
