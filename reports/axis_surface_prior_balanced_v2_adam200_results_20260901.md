# Balanced-v2 解析先验 Adam200 可优化性报告

日期：2026-09-01  
实验协议：`qh-axis-surface-contour-balanced-random-ok-adam200-64d-abi11-v1`  
运行提交：`11f703f2a3727ed709167e60cc1bb923c46b36a6`

## 结论

Balanced-v2 解析先验包含可观测且数量可用的 QH 可优化起点。83 条完整
Adam200 轨迹中，31 条的历史最佳 ABI-11 分数达到 50，合法起点内的条件成功率为
37.35%。将该比例乘以先验的初始合法率 37.5%，得到全先验 Adam200 可达丰度估计
14.01%。

Adam50 的口径明显更严格。前 50 步内共有 13/83 条轨迹达到 50，第 50 步当下也有
13/83 条保持在 50 以上。对应的合法起点内丰度为 15.66%，折算到全先验为
5.87%。该值可作为后续以 `best-so-far@50 >= 50` 定义高奖励样本时的初始 C2 丰度
估计。

高分端形成了约 60--70 分的平台，最高分为 69.6456，83 条轨迹中没有样本达到
70。达到 50 的 31 个最佳状态，其 `volume_qs` 分量中位数为 29.68，coil 分量中位数
为 70.73。Balanced-v2 解析先验已经提供了可达的 QH 收敛域，体 QH 质量和线圈工程仍限制
其进入更高分段。

两例代表性高分样本完成了标准 Simsopt 曲面求解、独立稠密残差、Poincare、直接
Boozer `|B|` 和 DESC 评估。`axisv2_case_02986` 接受到 `s=0.64`，体积为
0.0740 m³；`axisv2_case_04428` 接受到 `s=0.81`，体积为 0.0925 m³。两例的直接
QH 误差分别为 0.0170 和 0.0172，Poincare 截面均显示嵌套点列。DESC 在 50 次迭代
上限处停止；归一化力残差均值分别降低约 360 倍和 834 倍。两例结果提供了有力的
物理候选证据，同时保留了 DESC 尚未满足优化器收敛判据的边界。

## 实验设计

源数据来自 `qh-axis-surface-contour-balanced-score-abi11-v2` 的 6000 个独立样本。
原生 ABI-11 评分器认定其中 2250 个初始状态合法，合法率为 37.5%。本实验使用固定
随机种子 `20260902`，从这 2250 个合法状态中无放回均匀抽取 84 个样本。

每个起点使用精确、无裁剪的标准化线圈系数进入 direct-data Adam。标准化尺度取自
冻结 Flow checkpoint，优化过程中不调用 Flow 模型。每条轨迹执行 200 次 Adam
更新，每次使用 64 个新鲜随机正交方向和中心差分；主要参数为：

| 参数 | 值 |
|---|---:|
| 参数空间 | exact-unclipped standardized coil coefficients |
| Adam 更新数 | 200 |
| 每步方向数 | 64 |
| 差分扰动 `h` | 0.0025 |
| 学习率 | 0.01 |
| beta | (0.7, 0.999) |
| 评分器 | native ABI-11 |
| 完整物理评估 | 两例代表性高分样本，见“完整物理评估” |

评分库 SHA-256 为
`1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed`；checkpoint
SHA-256 为
`39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f`。

## 作业验收

准备作业 `51677` 和三步烟雾测试 `51678` 完成。烟雾测试的初始分数为 1.2541，
与冻结源记录相差 0.0313；参数 roundtrip 相对误差为 0；三步最佳分数为 6.7794。

正式计算使用 P107 四卡数组 `51679` 和 Students 两卡数组 `51680`。六个 worker
均通过 GPU preflight。五个 worker 完成各自的 14 条轨迹；worker 1 在 4.42 小时
后触发剩余预算保护，完成 13 条并停止开启最后一条。最终状态为：

| 项目 | 结果 |
|---|---:|
| 计划抽样 | 84 |
| 完整 Adam200 | 83 |
| 完成率 | 98.81% |
| 运行失败 | 0 |
| 不完整轨迹目录 | 0 |
| 未启动样本 | 1 |

未启动样本为 `axisv2_case_01510`，条件为 `nfp=6, nc=1`，初始总分 3.5518，
初始 coil 分量 80.0324。若该样本最终失败或成功，84 个入选样本的条件成功率分别为
36.90% 和 38.10%；折算到全先验后的范围为 13.84%--14.29%。单个未启动样本对总体
结论的影响较小。

83 条正式轨迹都记录 `completed_iterations=200`、`stop_reason=completed_iterations`
和 `status=ok`。依赖分析作业 `51681` 成功生成冻结汇总和逐轨迹表。

## 总体结果

| 指标 | 初始分数 | Adam200 历史最佳 |
|---|---:|---:|
| p10 | 1.2116 | 2.6182 |
| 中位数 | 2.7263 | 16.9956 |
| p90 | 6.6568 | 67.1651 |
| 最大值 | 10.7973 | 69.6456 |

Adam200 最佳分数低于 20 的样本为 44/83，位于 `[20, 50)` 的样本为 8/83，达到
50 的样本为 31/83。31/83 的条件成功率为 37.35%，Wilson 95% 区间为
27.72%--48.10%。该区间描述合法起点内的有限样本不确定性；14.01% 的全先验估计
还使用了 6000 个源样本测得的 37.5% 初始合法率。

![Balanced-v2 Adam200 分数分布和最佳分数生存曲线](assets/axis_surface_prior_balanced_v2_adam200_20260901/score_distribution.png)

图 1：协议 `qh-axis-surface-contour-balanced-random-ok-adam200-64d-abi11-v1` 的
83 条完整轨迹。左图比较初始 ABI-11 分数和 200 步内的历史最佳分数；右图给出最佳
分数的生存曲线。绿色虚线为 50 分成功阈值。

## 收敛时间结构

达到 50 的轨迹并不都在早期进入高分段。按历史首次跨过 50 的时间统计：

| 更新上限 | 累计达到 50 | 占 83 条完整轨迹 | 占最终 31 条成功轨迹 |
|---:|---:|---:|---:|
| 25 | 7 | 8.43% | 22.58% |
| 50 | 13 | 15.66% | 41.94% |
| 100 | 26 | 31.33% | 83.87% |
| 150 | 27 | 32.53% | 87.10% |
| 200 | 31 | 37.35% | 100% |

第 200 步当下有 30 条轨迹保持在 50 以上，另有 1 条轨迹的历史最佳超过 50 后回落。
五条成功轨迹在第 100 步后才首次跨过 50，其中四条发生在第 150 步后。Adam50
适合作为低成本富集标签，Adam200 额外发现了一批收敛较慢的可达起点。

![Balanced-v2 的 83 条 Adam200 轨迹与首次跨 50 分的时间](assets/axis_surface_prior_balanced_v2_adam200_20260901/trajectory_curves.png)

图 2：左图显示 83 条完整 ABI-11 轨迹，颜色标识基础线圈数 `nc`；右图显示每个
`nc` 组首次达到 50 分的累计比例。所有曲线均来自冻结的 200 步 history 文件。

## `nc` 和 `nfp` 分层

| nc | 完整样本 | 最佳分数中位数 | 达到 50 | 条件成功率 |
|---:|---:|---:|---:|---:|
| 1 | 41 | 15.0550 | 8 | 19.51% |
| 2 | 20 | 35.4855 | 10 | 50.00% |
| 3 | 10 | 28.6262 | 4 | 40.00% |
| 4 | 12 | 66.8154 | 9 | 75.00% |

未启动样本属于 `nc=1`。将该样本分别计为失败和成功后，`nc=1` 的成功率范围为
19.05%--21.43%。本次抽样中 `nc=4` 的可达率最高，`nc=1` 最低。

| nfp | 完整样本 | 最佳分数中位数 | 达到 50 | 条件成功率 |
|---:|---:|---:|---:|---:|
| 3 | 3 | 3.6400 | 0 | 0.00% |
| 4 | 16 | 7.2771 | 1 | 6.25% |
| 5 | 9 | 54.2828 | 6 | 66.67% |
| 6 | 19 | 14.5145 | 7 | 36.84% |
| 7 | 14 | 16.8238 | 4 | 28.57% |
| 8 | 22 | 63.1047 | 13 | 59.09% |

![按 nc 和 nfp 分层的 Adam200 条件成功率](assets/axis_surface_prior_balanced_v2_adam200_20260901/success_rates.png)

图 3：点为各组 `best-so-far@200 >= 50` 的观测比例，误差条为 Wilson 95% 区间，
文字给出成功数/完整样本数。`nc` 与 `nfp` 在本次随机样本中存在联合分布，分组差异
用于提出后续假设，尚不能解释为单一变量的因果效应。

## 计算成本和最佳点位置

单条 Adam200 的墙钟中位数为 823.4 秒，p90 为 1574.3 秒，最大值为 4281.4 秒，
对应 13.72、26.24 和 71.36 分钟。`nc` 增加时运行成本明显上升。六个 worker 的
总墙钟范围为 3.08--4.42 小时，均处于 5 小时预算内。

![按 nc 的运行成本和最佳分数出现位置](assets/axis_surface_prior_balanced_v2_adam200_20260901/runtime_and_convergence.png)

图 4：左图给出各 `nc` 组单条 Adam200 的墙钟分布；右图给出历史最佳分数出现的
更新步，颜色标识 `nc`。多个高分样本在 150--200 步达到最佳值。

## 高分端组成

最高分的五个样本为：

| 样本 | nfp | nc | 初始分数 | 最佳分数 | 最佳步 | 第 200 步分数 |
|---|---:|---:|---:|---:|---:|---:|
| `axisv2_case_02986` | 5 | 4 | 7.3602 | 69.6456 | 170 | 69.4161 |
| `axisv2_case_01081` | 5 | 3 | 1.2182 | 69.5408 | 118 | 56.5847 |
| `axisv2_case_04428` | 5 | 2 | 0.9754 | 69.1175 | 182 | 69.0002 |
| `axisv2_case_04155` | 4 | 4 | 6.6930 | 68.6537 | 200 | 68.6537 |
| `axisv2_case_02155` | 6 | 4 | 2.3490 | 68.2056 | 200 | 68.2056 |

前三个最佳状态的 `volume_qs` 分量为 38.54、36.89、35.94，coil 分量为 70.50、
69.94、69.49。31 个达到 50 的最佳状态中，`volume_qs` 分量范围为 24.25--42.59，
中位数 29.68；coil 分量范围为 62.17--80.27，中位数 70.73。高分平台同时受有限
体 QH 质量和中等线圈工程分量约束。

源起点的初始 coil 分量在成功组中的中位数为 69.91，在未达到 50 的组中为 77.04。
初始线圈工程分量较高没有单独预测 Adam200 的 QH 可达性；该关系还受到 `nc`、
`nfp` 和起点几何的共同影响。

## 完整物理评估

完整评估选择同为 `nfp=5` 的两个样本：总分最高的 `axisv2_case_02986`
（`nc=4`）和总分接近且基础线圈数较少的 `axisv2_case_04428`（`nc=2`）。该选择
同时覆盖本批次的最高 screening 状态和不同线圈数量。两例输入分别是各自 Adam200
轨迹的历史最佳状态。

每例先在 `a=0.04, 0.05, 0.06, 0.08` 上拟合 psi，均选择 `a=0.08`；随后由内向外
提交 `s` 候选，使用标准 Simsopt LS/Newton、97×97 独立稠密网格和体积单调性选择
最大接受曲面。下表中的外侧失败级别是已测试网格上的最近失败点，因此最大接受
曲面是离散搜索结果。

| 样本 | nc | Adam200 最佳分数 | `a` | 最大接受 `s` | 最近外侧失败 `s` | `iota` | `G` | 体积 (m³) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `axisv2_case_02986` | 4 | 69.6456 | 0.08 | 0.64 | 0.81 | -1.16467 | 6.28319 | 0.074017 |
| `axisv2_case_04428` | 2 | 69.1175 | 0.08 | 0.81 | 1.00 | -1.07142 | 6.28319 | 0.092547 |

两例标准曲面均通过 Newton 收敛、稠密相对 L2、法向场 p95、环向绕数和非零法向量
检查。`axisv2_case_02986` 的外侧 `s=0.81` 在进入标准求解前只有 161978 个有效
采样点，低于 180000 的要求；`axisv2_case_04428` 的 `s=1.00` 和 `s=1.21` 分别
只有 115616 和 41349 个有效采样点。

| 样本 | 稠密相对 L2 | 法向场比 p95 | QA | QH | QP | `|B|` 范围 (T) | Poincare |
|---|---:|---:|---:|---:|---:|---:|---:|
| `axisv2_case_02986` | 6.43e-5 | 7.47e-5 | 0.01124 | 0.01701 | 0.01267 | 0.6426--1.0018 | 8×67 hits |
| `axisv2_case_04428` | 4.38e-5 | 7.33e-5 | 0.01221 | 0.01720 | 0.01310 | 0.6054--1.0456 | 8×35 hits |

### `axisv2_case_02986`

| 线圈与最大接受曲面 | Poincare 截面 | 直接 Boozer `|B|` |
|---|---|---|
| ![axisv2_case_02986 的线圈与 s=0.64 最大接受曲面](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/assets/coils_surface.png) | ![axisv2_case_02986 的四个 Poincare 截面](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/assets/poincare.png) | ![axisv2_case_02986 最大接受曲面的 Boozer 磁场强度](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/assets/boozer_b.png) |

图 5：`axisv2_case_02986` 的完整装置几何、四个 Poincare 截面和 `s=0.64` 标准
Boozer 曲面上的磁场强度。黑线为全部物理线圈。交互文件：
[线圈与曲面 HTML](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/assets/coils_surface.html)、
[Boozer `|B|` HTML](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/assets/boozer_b.html)。

| DESC 初始边界 | DESC 最终边界 |
|---|---|
| ![axisv2_case_02986 的 DESC 初始边界](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/desc/boundary_initial.png) | ![axisv2_case_02986 的 DESC 最终边界](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/desc/boundary.png) |

| DESC Boozer 模谱 | DESC Boozer `|B|` |
|---|---|
| ![axisv2_case_02986 的 DESC Boozer 模谱](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/desc/boozer_modes.png) | ![axisv2_case_02986 的 DESC Boozer 磁场强度](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/desc/boozer_B.png) |

| DESC QA 误差 | DESC QH 误差 | DESC QP 误差 |
|---|---|---|
| ![axisv2_case_02986 的 DESC QA 误差](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/desc/qs_QA.png) | ![axisv2_case_02986 的 DESC QH 误差](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/desc/qs_QH.png) | ![axisv2_case_02986 的 DESC QP 误差](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/desc/qs_QP.png) |

![axisv2_case_02986 的 DESC iota 剖面](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/desc/iota.png)

图 6：`axisv2_case_02986` 的 DESC 诊断。边界在求解前后保持嵌套；归一化力残差
均值由 0.9003 降至 0.002504，p95 由 1.7583 降至 0.006311。求解器在第 50 次
迭代达到上限，最终 cost 为 1.55e-4。

### `axisv2_case_04428`

| 线圈与最大接受曲面 | Poincare 截面 | 直接 Boozer `|B|` |
|---|---|---|
| ![axisv2_case_04428 的线圈与 s=0.81 最大接受曲面](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/assets/coils_surface.png) | ![axisv2_case_04428 的四个 Poincare 截面](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/assets/poincare.png) | ![axisv2_case_04428 最大接受曲面的 Boozer 磁场强度](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/assets/boozer_b.png) |

图 7：`axisv2_case_04428` 的完整装置几何、四个 Poincare 截面和 `s=0.81` 标准
Boozer 曲面上的磁场强度。黑线为全部物理线圈。交互文件：
[线圈与曲面 HTML](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/assets/coils_surface.html)、
[Boozer `|B|` HTML](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/assets/boozer_b.html)。

| DESC 初始边界 | DESC 最终边界 |
|---|---|
| ![axisv2_case_04428 的 DESC 初始边界](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/desc/boundary_initial.png) | ![axisv2_case_04428 的 DESC 最终边界](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/desc/boundary.png) |

| DESC Boozer 模谱 | DESC Boozer `|B|` |
|---|---|
| ![axisv2_case_04428 的 DESC Boozer 模谱](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/desc/boozer_modes.png) | ![axisv2_case_04428 的 DESC Boozer 磁场强度](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/desc/boozer_B.png) |

| DESC QA 误差 | DESC QH 误差 | DESC QP 误差 |
|---|---|---|
| ![axisv2_case_04428 的 DESC QA 误差](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/desc/qs_QA.png) | ![axisv2_case_04428 的 DESC QH 误差](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/desc/qs_QH.png) | ![axisv2_case_04428 的 DESC QP 误差](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/desc/qs_QP.png) |

![axisv2_case_04428 的 DESC iota 剖面](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/desc/iota.png)

图 8：`axisv2_case_04428` 的 DESC 诊断。边界在求解前后保持嵌套；归一化力残差
均值由 1.4306 降至 0.001715，p95 由 3.8908 降至 0.003839。求解器在第 50 次
迭代达到上限，最终 cost 为 3.12e-5。

两例线圈几何均保持 balanced-v2 优化后出现的高复杂度。完整评估证明所选磁场含有
可求解且嵌套的 QH 候选曲面，同时提示后续先验和优化仍需直接约束线圈曲率、挠率、
间距及形状复杂度。

## 解释和后续使用

1. Balanced-v2 先验为 QH 优化提供了数量充足的可达起点。按 Adam50 历史最佳口径，
   全先验约 5.87% 的样本可达到 50；该丰度足以支持后续奖励富集实验。
2. 原始分数筛选会遗漏大量可优化起点。84 个入选样本的初始分数均低于 20，
   Adam200 仍产生 31 个 50 分以上样本。
3. Adam100 已发现 26/31 个最终成功起点。Adam50、Adam100 和 Adam200 可分别作为
   低成本标签、主要训练标签和延迟收敛审计标签。
4. `nc=4` 同时表现出较高可达率和较高计算成本。后续训练需要显式记录并控制
   `nc/nfp` 分布，避免富集结果仅来自条件构成变化。
5. 两例约 69 分样本完成了完整物理评估并获得嵌套标准曲面；其余 81 条完整轨迹仍
   只有 ABI-11 screening 证据。两例 DESC 求解均达到 50 次迭代上限，报告保留该
   收敛边界。

该实验是解析先验探索，不改变项目默认协议
`qh-flow-screen32-adam200-64d-abi11-v1`。

## 冻结证据

- [实验协议](assets/axis_surface_prior_balanced_v2_adam200_20260901/protocol.json)
- [运行清单](assets/axis_surface_prior_balanced_v2_adam200_20260901/runtime_manifest.json)
- [抽样与哈希清单](assets/axis_surface_prior_balanced_v2_adam200_20260901/selection_manifest.json)
- [汇总统计](assets/axis_surface_prior_balanced_v2_adam200_20260901/summary.json)
- [逐轨迹表](assets/axis_surface_prior_balanced_v2_adam200_20260901/trajectories.csv)
- [首次跨阈值统计](assets/axis_surface_prior_balanced_v2_adam200_20260901/first_passage_summary.json)
- [烟雾测试结果](assets/axis_surface_prior_balanced_v2_adam200_20260901/smoke_result.json)
- [原始分析轨迹图](assets/axis_surface_prior_balanced_v2_adam200_20260901/adam200_trajectories.png)
- [`axisv2_case_02986` 曲面选择](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/selection.json)
- [`axisv2_case_02986` 完整评估汇总](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_02986/full/full_summary.json)
- [`axisv2_case_04428` 曲面选择](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/selection.json)
- [`axisv2_case_04428` 完整评估汇总](assets/axis_surface_prior_balanced_v2_adam200_20260901/full_eval/axisv2_case_04428/full/full_summary.json)

完整评估使用提交 `89206f4278155bc1c3b06db8bbe59aaea0e3ae9e` 的固定流程，GPU
库由该提交构建，SHA-256 为
`23158593e57cd82300aa8d2efb2ee3023662d7f9d1765d1cfa22c84f26434af0`。两例输入
SHA-256 分别为
`c26c6aa0dc8d9aa230fe514440d3eaa3db0e0b162e5bc2fcc1c7382de2ab13c3` 和
`ad4560d213d218cb4c065cc57703aa3371d256a5213fe0dee0932d3ba3c38628`。正式重跑前的
共享混合库缺少 `sgpu_trace_axis_samples`，四个失败作业在评估器构造阶段停止；该输出
根目录仅保留为运行故障证据，没有进入任何统计或物理结论。
