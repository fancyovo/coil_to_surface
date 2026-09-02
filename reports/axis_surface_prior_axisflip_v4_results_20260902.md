# 构造参考轴手性翻转 v4：正手性 Adam200 实验报告

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-02 |
| 分支 | `codex/axis-surface-prior` |
| 协议 | `qh-axis-surface-contour-compact-flexible-axisflip-stream-adam200-64d-abi11-v4` |
| 运行代码 | `d8de349b7d0d457f10402c8517e5930ddd260c7c` |
| 报告统计与绘图代码 | `2a9e66c` |
| 原生评分器 | ABI-11，SHA-256 `1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed` |
| 完整评估运行库 | 从 `d8de349` 构建，SHA-256 `93bc6f00d1ab278f3454275e73958a04105ac110706cbfbb3cc7bc90a539361a` |
| 目标手性 | `(M,N)=(1,+nfp)` |

## 结论

构造参考轴的手性翻转成功把 compact-flexible 解析先验移到正 iota 分支。v4 共筛选
110 个随机样本，56 个通过正式 ABI-11 配置，初始合法率为 50.91%；56 个合法样本的
iota 全为正。50 条完整 Adam200 轨迹的初始点、最佳点和终点也全部保持正 iota。

50 条完整轨迹中，39 条达到 50 分，37 条达到 70 分，9 条达到 80 分。轨迹最佳分数
中位数为 `78.7478`，最高分为 `81.8258`。37/50 的 70 分达成率与 compact-flexible
v3 固定正手性目标下 0/120 达到 70 的历史结果形成明显的观测差异。该完整配方证据支持
“旧批次的负手性与正目标失配造成约 70 分平台”这一解释。

体 QS 分量是本批次进入高分段的主导区分量。50 条轨迹的体 QS 分量中位数从
`19.4311` 升到 `59.1199`，线圈工程分量中位数从 `69.3725` 变为 `69.1066`。
Adam200 在群体层面大幅提高体 QS，同时基本保持线圈工程分量的中位水平。单条轨迹
仍存在工程权衡：24 条轨迹表现为体 QS 提高、线圈工程分量下降，23 条同时提高两个
分量。

本实验还完成了两个代表性高分样本的标准磁面、Poincare、Boozer 和 DESC 评估。
`axisflip_case_0000018` 与 `axisflip_case_0000023` 在测试层级中均接受 `s=0.49`
磁面，直接面上 `QH_(1,+1)` 误差分别为 `0.001626` 和 `0.002174`。两例的 DESC
初始与最终边界均保持嵌套；DESC 求解达到 50 次迭代上限，尚未满足优化器收敛判据。
这两例提供样本级物理验证；群体筛选结论继续由 50 条 Adam200 轨迹支持。v4 属于
解析先验探索，项目 QH 默认协议继续使用 `qh-flow-screen32-adam200-64d-abi11-v1`。

## 实验方法

v4 从 compact-flexible v3 的随机构造流程生成参考轴、绕组面和标量场等值线。每个
样本仅把构造参考轴的全部垂直 Fourier 系数乘以 `-1`，随后沿翻转后的参考轴重新生成
绕组面和线圈。实验覆盖注册的 26 个 `(nfp,nc)` 条件，并限制 `nc<=4`。

每个解析样本先映射到优化器可精确表示的未裁剪 FP32 标准化数据坐标，再重建物理
线圈。筛选和 Adam step 0 使用同一组重建线圈、同一套正式评分配置和同一条磁轴分支。
筛选阶段执行全局磁轴搜索；优化器 step 0 严格延续保存的 `axis_R/axis_Z`。更新前
要求两次总分差的绝对值不超过 `0.1`。

| Adam200 参数 | 值 |
|---|---:|
| 参数空间 | 精确、未裁剪的标准化线圈数据坐标 |
| Flow 调用 | 0 |
| 更新数 | 200 |
| 每步方向数 | 64 个新鲜随机正交方向 |
| 差分 | 中心差分 |
| 扰动 `h` | 0.0025 |
| 学习率 | 0.01 |
| beta | `(0.7, 0.999)` |
| 正式曲面角向点数 | 128 |
| 正式曲面跟踪步数 | 400 |

### 一致性问题与修复链

| 版本 | 状态 | 发现的问题 | 有效范围 |
|---|---|---|---|
| v1，`599dd31` | 作废 | 紧凑筛选记录遗漏 `axis_R/axis_Z`；一致性检查位于 Adam200 之后 | 24 个合法筛选点的正 iota 仅作初步手性证据 |
| v2，`57c3a06` | 作废 | 筛选使用生成器 FP64 token，优化器使用 FP32 标准化重建 token | 证明更新前一致性门能够及时阻断错误运行 |
| v3，`80d3d78` | 作废 | 筛选沿用 ABI-11 库默认曲面配置，优化器使用正式中心配置 | case 25 的 `1.84996` 分差定位了配置不一致 |
| v4，`d8de349` | 本报告协议 | 统一可表示 token、磁轴分支和正式评分配置 | 50 条完整 Adam200 与 110 条筛选进入正式统计 |

固定困难样本 case 25 的 v4 smoke 作业 `52758` 给出筛选分 `68.705151`、优化器
step-0 分数 `68.691824`，差值为 `0.013327`。六个正式 worker 的首条一致性差值
范围为 `[0.000711,0.013327]`，保存轴与延续轴完全一致。v1-v3 的 Adam 结果保持
隔离状态，未进入本报告的任何统计量或图。

## 作业闭环与排空

P107 数组 `52759` 使用四张 RTX 5090，Students 数组 `52760` 使用两张 RTX 5090，
六个独立 worker 并行执行。分析依赖作业 `52761` 在数组排空后完成汇总。

用户在约 2.3 小时时要求结束发现阶段。两秒间隔的排空监视器逐个等待当时活跃的
`.partial` 目录原子迁入完整轨迹目录，再取消对应数组元素。六条当时活跃的 Adam200
全部完成。运行器在取消到达前领取了六个后继合法起点，保存了 0--7 个更新；这些
后继项保留在 `incomplete/`，并从完成轨迹统计中排除。

| 账目 | 数量 |
|---|---:|
| 已筛选 | 110 |
| 初始合法 | 56 |
| 完整 Adam200 | 50 |
| 运行失败 | 0 |
| 用户排空的后继 partial | 6 |
| 未归档合法 case | 0 |

六个后继 partial 分别为 case `102/121/134/99/82/101`，对应保存更新数
`2/0/3/4/7/3`。它们不参与阈值、分量、手性端点和 `nc/nfp` 优化统计。机器可读
排空记录见 [`drain_manifest.json`](assets/axis_surface_prior_axisflip_v4_20260902/drain_manifest.json)。

## 初始筛选与手性

| ABI-11 状态 | 数量 | 比例 |
|---|---:|---:|
| `ok` | 56 | 50.91% |
| `no_axis` | 33 | 30.00% |
| `no_surface` | 18 | 16.36% |
| `flux_rejected` | 3 | 2.73% |

56/110 的合法率 Wilson 95% 区间为 41.70%--60.06%。所有 56 个合法起点的 iota
为正；按 `nc=1,2,3,4` 分组后，非空组仍保持 100% 正 iota。构造参考轴翻转已经
消除 compact-flexible v3 的全负 iota 群体偏置。

![v4 按 nc 的正式筛选合法率与 Adam200 条件成功率](assets/axis_surface_prior_axisflip_v4_20260902/group_rates.png)

图 1：左图为 110 个 v4 样本按基础线圈数分组的正式配置合法率；右图为 50 条完整
轨迹中 Adam200 轨迹最佳达到 50 的条件比例，误差条为 Wilson 95% 区间。

## Adam200 总分结果

| 指标 | 初始分数 | Adam200 轨迹最佳 | 增益 |
|---|---:|---:|---:|
| 中位数 | 28.2525 | 78.7478 | 22.8570 |
| p90 | 66.2848 | 80.6126 | 65.4478 |
| 最大值 | 75.8373 | 81.8258 | 75.0032 |

| 最佳分阈值 | 达到数量 | 条件比例 | Wilson 95% 区间 |
|---:|---:|---:|---:|
| 50 | 39/50 | 78.00% | 64.76%--87.25% |
| 60 | 39/50 | 78.00% | 64.76%--87.25% |
| 70 | 37/50 | 74.00% | 60.45%--84.13% |
| 80 | 9/50 | 18.00% | 9.77%--30.80% |

36 条轨迹在第 25 步前达到 50，38 条在第 50 步前达到 50，最终 39 条成功轨迹在
第 85 步前全部达到 50。前 50 步覆盖了 38/39 个最终成功起点，说明正手性先验中
的大多数可达高分样本能够很快进入高分段。

![v4 的 50 条 Adam200 当步分数与截至当步最佳轨迹](assets/axis_surface_prior_axisflip_v4_20260902/adam200_trajectories.png)

图 2：左图显示 50 条完整轨迹的当步 ABI-11 总分；右图显示截至每一步的最佳分数。
颜色表示 `nc`，红色虚线为 50 分阈值。低分轨迹主要来自 `nc=1`，高分轨迹集中在
约 75--82 分。

![v4 初始分数与 Adam200 最佳分数分布](assets/axis_surface_prior_axisflip_v4_20260902/score_outcomes.png)

图 3：左图对比初始分与轨迹最佳分的分布；右图逐点展示初始分和最终可达最佳分。
低至 3--20 分的多个起点仍进入 75 分以上，初始总分对可达上限只有有限筛选能力。

## 体 QS 与线圈工程分量

`volume_qs` 和 `coil` 均为 ABI-11 的 0--100 分分量，数值越高表示该项越好。初始
分量读取自正式筛选记录，最佳分量读取自每条轨迹的 `best.json`。二者共享 v4 的
正式评分配置，更新前一致性门把磁轴分支差异限制在 0.1 总分以内。

| 分量 | 初始中位数 | 最佳点中位数 | 初始 p90 | 最佳点 p90 | 最佳点范围 |
|---|---:|---:|---:|---:|---:|
| 体 QS | 19.4311 | 59.1199 | 35.8700 | 65.4105 | 1.2193--66.9787 |
| 线圈工程 | 69.3725 | 69.1066 | 76.8721 | 76.6774 | 61.2277--79.4451 |

体 QS 的配对变化中位数为 `+31.4269`，范围为 `-0.6333` 到 `+54.1381`。线圈工程
分量的配对变化中位数为 `-0.0405`，范围为 `-11.4451` 到 `+8.3579`。联合变化分为：

| 联合变化 | 轨迹数 |
|---|---:|
| 体 QS 与线圈工程同时提高 | 23 |
| 体 QS 提高、线圈工程下降 | 24 |
| 体 QS 下降、线圈工程提高 | 1 |
| 两个分量同时下降 | 2 |

![v4 体 QS 与线圈工程分量的分布、配对移动和联合变化](assets/axis_surface_prior_axisflip_v4_20260902/component_tradeoff.png)

图 4：左图给出两个分量在初始点和轨迹最佳点的分布；中图用线段连接每条轨迹的
初始点 `x` 与最佳点 `o`；右图给出两个分量的配对变化。Adam200 的主要移动方向
沿体 QS 轴展开，线圈工程变化围绕零分布，并包含少数明显工程代价。

按最佳总分分组后，分量结构更加清楚：

| 最佳总分组 | 轨迹数 | 最佳体 QS 中位数 | 最佳体 QS 范围 | 最佳线圈工程中位数 | 最佳线圈工程范围 |
|---|---:|---:|---:|---:|---:|
| `<50` | 11 | 11.2361 | 1.2193--18.6523 | 76.5679 | 71.9588--79.4451 |
| `[50,70)` | 2 | 27.2832 | 26.4383--28.1281 | 72.7839 | 71.9970--73.5707 |
| `>=70` | 37 | 61.0771 | 42.8972--66.9787 | 68.3801 | 61.2277--74.1016 |

最佳总分与最佳体 QS 的 Pearson 相关系数为 `+0.963`，与最佳线圈工程分量的相关
系数为 `-0.728`。初始点对应相关系数为 `+0.930` 和 `-0.724`。这些系数描述本批次
的群体结构：低分组已经具有较高工程分量，体 QS 停留在低值；高分组通过显著提高
体 QS 进入 70 分以上，同时接受一定线圈工程权衡。相关系数按观测分布解释，单独
改变某个分量的因果效应需要受控配对实验。

## `nc` 与 `nfp` 分层

| nc | 筛选数 | 初始合法 | 完整 Adam200 | 最佳 >=50 | 最佳 >=70 | 最佳总分中位数 | 最佳体 QS 中位数 | 最佳 coil 中位数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22 | 10 (45.45%) | 8 | 2 | 0 | 12.9104 | 11.0410 | 77.1153 |
| 2 | 30 | 12 (40.00%) | 12 | 9 | 9 | 75.8590 | 52.3388 | 70.1382 |
| 3 | 30 | 18 (60.00%) | 17 | 15 | 15 | 79.1040 | 59.4242 | 68.6069 |
| 4 | 28 | 16 (57.14%) | 13 | 13 | 13 | 79.9008 | 64.4711 | 63.7604 |

`nc=1` 起点保持最高的线圈工程分量，同时多数轨迹无法提高体 QS。`nc=3/4` 的体 QS
可达性明显更高；13 条完整 `nc=4` 轨迹全部达到 70。排空后继项含两个 `nc=1`、
一个 `nc=3` 和三个 `nc=4`，因此表中的优化比例对应 50 条完整 stream-prefix 轨迹。
样本量和运行成本共同影响分层组成，组间差异用于后续采样设计。

| nfp | 完整轨迹 | 最佳 >=50 | 最佳 >=70 | 最佳总分中位数 | 最大值 |
|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 1 | 1 | 11.9817 | 78.7806 |
| 5 | 8 | 5 | 5 | 76.0800 | 80.3210 |
| 6 | 10 | 8 | 8 | 79.2959 | 80.4102 |
| 7 | 11 | 10 | 10 | 78.7712 | 80.5856 |
| 8 | 16 | 15 | 13 | 78.9506 | 81.8258 |

## 高分样本

| 样本 | nfp | nc | 最佳总分 | 最佳步 | 最佳体 QS | 最佳线圈工程 |
|---|---:|---:|---:|---:|---:|---:|
| `axisflip_case_0000018` | 8 | 3 | **81.8258** | 195 | 66.9787 | 66.2580 |
| `axisflip_case_0000044` | 8 | 3 | 81.3592 | 65 | 65.8927 | 64.8544 |
| `axisflip_case_0000051` | 8 | 4 | 81.3521 | 167 | 66.8593 | 62.6159 |
| `axisflip_case_0000070` | 8 | 3 | 81.1516 | 138 | 64.5643 | 65.2851 |
| `axisflip_case_0000103` | 8 | 4 | 80.8558 | 41 | 64.1749 | 63.7604 |

最高五条轨迹都来自 `nfp=8`，并覆盖 `nc=3/4`。其最佳体 QS 为 64.17--66.98，
线圈工程分量为 62.62--66.26。`axisflip_case_0000018` 同时给出本批次最高总分和
最高体 QS 分量；代表性完整物理评估可优先从该样本和工程分量更高的高分样本中选取。

## 代表性高分样本完整评估

完整评估选取两种高分结构：`axisflip_case_0000018` 是 50 条完整轨迹中的最高总分
样本，也具有最高体 QS 分量；`axisflip_case_0000023` 保持 80 分以上总分，同时给出
更高的线圈工程分量。两者分别覆盖 `nfp=8,nc=3` 与 `nfp=6,nc=4`。

| 样本 | 选择角色 | Adam200 最佳总分 | 最佳步 | 体 QS 分量 | 线圈工程分量 |
|---|---|---:|---:|---:|---:|
| `axisflip_case_0000018` | 最高总分、最高体 QS | **81.8258** | 195 | 66.9787 | 66.2580 |
| `axisflip_case_0000023` | 高分中的较优工程折中 | 80.3275 | 174 | 64.2464 | **71.2918** |

表中的总分来自冻结 Adam200 轨迹，两个分量来自同一 ABI-11 正式配置下的端点重算。
体 QS 和线圈工程分量均为分数，数值越高越好。后文的 `QH_(1,+1)` 是标准磁面上的
直接误差，数值越低越好；该误差与体 QS 分数的定义和量纲不同。

### 评估流程与并行安排

每个样本先并行评估 `a=0.04,0.05,0.06,0.08 m` 四个源 psi 拟合，随后选用
`a=0.08 m`。每个源面再并行计算 `s=0.12,0.24,0.36,0.49,0.64,0.81` 六个
候选层级，并对每个候选执行 Simsopt 最小二乘、标准 Newton 和独立 97 点稠密网格
检查。两个样本的线圈/磁面可视化、Poincare、Boozer 与 DESC 作业也彼此并行运行。

源 psi 在 `a=0.08 m` 下的留出 RMS 分别为 `8.89e-5` 和 `4.82e-4`。测试集合中，
两个样本的最大接受层级均为 `s=0.49`；最近外层 `s=0.64` 均被独立稠密残差门拒绝。

| 样本 | 接受 `s` | 体积 (`m^3`) | iota | Newton 残差 | 97 点相对 L2 | 97 点法向 B 正弦 p95 | 直接 `QH_(1,+1)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `axisflip_case_0000018` | 0.49 | 0.076781 | 1.181798 | 3.44e-13 | 3.20e-5 | 5.50e-5 | **0.001626** |
| `axisflip_case_0000023` | 0.49 | 0.045235 | 1.099510 | 1.69e-13 | 6.64e-5 | 7.30e-5 | 0.002174 |

| 样本 | `s=0.64` 的 97 点相对 L2 | `s=0.64` 的法向 B 正弦 p95 | 拒绝项 |
|---|---:|---:|---|
| `axisflip_case_0000018` | 6.93e-5 | 1.18e-4 | 法向 B p95 超过 `1e-4` |
| `axisflip_case_0000023` | 1.25e-4 | 1.33e-4 | 相对 L2 与法向 B p95 均超过 `1e-4` |

`s=0.49` 的 Newton 残差和两项独立稠密检查全部通过。`s=0.64` 的 Newton 本身
仍收敛到约 `2e-13`，外层拒绝来自独立稠密网格；这项检查限定了本次测试可报告的最大
标准磁面。

### `axisflip_case_0000018`：最高总分样本

该样本的接受磁面上 `|B|` 范围为 `0.6550--1.1126 T`。八条 Poincare 种子的
命中数为 `91,91,91,91,91,91,87,87`；四个截面保持径向有序，图中没有出现
明显的贯穿岛链。Boozer `|B|` 等值线整体沿正斜率方向延伸，并带有可见局部起伏。

![axisflip_case_0000018 的线圈与最大接受标准磁面](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/assets/coils_surface.png)

[交互式线圈与磁面](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/assets/coils_surface.html) | [交互式 Boozer `|B|`](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/assets/boozer_b.html)

![axisflip_case_0000018 的 Poincare 截面](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/assets/poincare.png)

![axisflip_case_0000018 的标准磁面 Boozer |B|](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/assets/boozer_b.png)

### `axisflip_case_0000023`：较优工程折中样本

该样本的接受磁面上 `|B|` 范围为 `0.6459--0.9565 T`。八条 Poincare 种子均
取得 79 次命中，四个截面保持径向有序。Boozer `|B|` 的主等值线同样沿正斜率
方向延伸，局部轮廓比 `axisflip_case_0000018` 更平滑。

![axisflip_case_0000023 的线圈与最大接受标准磁面](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/assets/coils_surface.png)

[交互式线圈与磁面](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/assets/coils_surface.html) | [交互式 Boozer `|B|`](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/assets/boozer_b.html)

![axisflip_case_0000023 的 Poincare 截面](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/assets/poincare.png)

![axisflip_case_0000023 的标准磁面 Boozer |B|](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/assets/boozer_b.png)

### DESC 松弛

| 样本 | 环向磁通 | 初始/最终嵌套 | 平均归一化力：初始 -> 最终 | 降低倍数 | 最终最大值 | 最终 p95 | 求解状态 |
|---|---:|---|---:|---:|---:|---:|---|
| `axisflip_case_0000018` | 0.008877 | 是 / 是 | 1.5837 -> 1.4218e-4 | 11139x | 1.40e-3 | 3.11e-4 | 达到 50 次迭代上限 |
| `axisflip_case_0000023` | 0.004572 | 是 / 是 | 2.5371 -> 1.9823e-3 | 1280x | 9.21e-3 | 4.31e-3 | 达到 50 次迭代上限 |

两例在 DESC 松弛前后均通过嵌套边界检查，平均归一化力分别降低约四个和三个数量级。
两次求解都在第 50 次迭代停止，优化器的 `success` 标志为 `false`。证据等级为
“显著松弛且边界保持嵌套”；数值收敛终止条件仍待更长 DESC 求解确认。

<details>
<summary><code>axisflip_case_0000018</code> 的完整 DESC 图组</summary>

![axisflip_case_0000018 DESC 初始边界](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/desc/boundary_initial.png)
![axisflip_case_0000018 DESC 最终边界](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/desc/boundary.png)
![axisflip_case_0000018 DESC Boozer 模谱](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/desc/boozer_modes.png)
![axisflip_case_0000018 DESC Boozer |B|](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/desc/boozer_B.png)
![axisflip_case_0000018 DESC QA 诊断](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/desc/qs_QA.png)
![axisflip_case_0000018 DESC QH 诊断](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/desc/qs_QH.png)
![axisflip_case_0000018 DESC QP 诊断](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/desc/qs_QP.png)
![axisflip_case_0000018 DESC iota 剖面](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/desc/iota.png)

</details>

<details>
<summary><code>axisflip_case_0000023</code> 的完整 DESC 图组</summary>

![axisflip_case_0000023 DESC 初始边界](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/desc/boundary_initial.png)
![axisflip_case_0000023 DESC 最终边界](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/desc/boundary.png)
![axisflip_case_0000023 DESC Boozer 模谱](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/desc/boozer_modes.png)
![axisflip_case_0000023 DESC Boozer |B|](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/desc/boozer_B.png)
![axisflip_case_0000023 DESC QA 诊断](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/desc/qs_QA.png)
![axisflip_case_0000023 DESC QH 诊断](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/desc/qs_QH.png)
![axisflip_case_0000023 DESC QP 诊断](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/desc/qs_QP.png)
![axisflip_case_0000023 DESC iota 剖面](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/desc/iota.png)

</details>

完整评估支持两个局部结论。第一，80 分附近的 v4 端点能够通过标准磁面与独立稠密
检查，且接受面具有 `1.6e-3--2.2e-3` 的直接 QH 误差。第二，较高原生总分可同时
覆盖不同的线圈工程折中：`case_0000018` 提供更高体 QS，`case_0000023` 以约
`2.73` 分的体 QS 差换取约 `5.03` 分的线圈工程增益。两例样本量只用于确认可行性，
50 条轨迹总体的磁面成功率需要更大的完整评估样本才能估计。

## 与已冻结手性实验的关系

compact-flexible v3 的 569 个合法源样本和 120 条固定正手性目标 Adam200 轨迹均
位于负 iota 分支；该批次最高分为 `68.8804`，达到 70 的数量为 0。两条负手性目标
续跑分别达到 `80.8523` 和 `84.9308`。v4 把构造参考轴翻到另一镜像类，在固定
正手性目标下得到 56/56 正 iota 合法起点，并使 37/50 条完整轨迹达到 70。

这组证据把参考轴固定符号确认为旧先验单边手性的主导来源，也确认目标手性匹配能
打开 80 分附近的收敛域。v3 与 v4 使用不同随机 stream，且 v4 统一了正式筛选配置；
两批最高分和成功率按完整配方结果比较。严格的单因素效应量可通过同 seed 的轴翻转
配对评分与配对 Adam 获得。

## 证据与复现

- 冻结远端运行根目录：
  `/home/scc/pb24511935/local_surface_evaluator_runs/axis_surface_prior_axisflip_stream_adam200_v4_20260902_d8de349`
- [协议副本](assets/axis_surface_prior_axisflip_v4_20260902/protocol.json)
- [运行清单](assets/axis_surface_prior_axisflip_v4_20260902/runtime_manifest.json)
- [分析汇总](assets/axis_surface_prior_axisflip_v4_20260902/analysis_summary.json)
- [110 条筛选记录](assets/axis_surface_prior_axisflip_v4_20260902/screening.csv)
- [50 条完整轨迹表](assets/axis_surface_prior_axisflip_v4_20260902/trajectories.csv)
- [体 QS 与 coil 配对端点表](assets/axis_surface_prior_axisflip_v4_20260902/component_endpoints.csv)
- [报告派生统计](assets/axis_surface_prior_axisflip_v4_20260902/report_metrics.json)
- [用户排空机器记录](assets/axis_surface_prior_axisflip_v4_20260902/drain_manifest.json)
- [完整评估运行清单](assets/axis_surface_prior_axisflip_v4_20260902/full_eval_manifest.json)
- `axisflip_case_0000018`：[磁面选择](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/selection.json)；[完整汇总](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000018/full/full_summary.json)
- `axisflip_case_0000023`：[磁面选择](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/selection.json)；[完整汇总](assets/axis_surface_prior_axisflip_v4_20260902/full_eval/axisflip_case_0000023/full/full_summary.json)

逐步 history、起点、最佳点和完整优化器清单保存在冻结远端运行根目录。报告中的
图和表由 `scripts/render_axis_surface_prior_axisflip_report.py` 从该目录重新计算。
