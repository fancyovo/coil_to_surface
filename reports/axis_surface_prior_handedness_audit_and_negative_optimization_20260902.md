# 解析先验手性审计与负手性 Adam200 实验报告

| 元数据 | 值 |
|---|---|
| 日期 | 2026-09-02 |
| 分支 | `codex/axis-surface-prior` |
| 群体与镜像审计代码 | `2dadc49c0fa74c12a4d26474ec914789b6423964` |
| 负手性续跑代码 | `729e592dfd7227d60d00fee68f63841bede65e1d` |
| 评分器 | 原生 ABI-11，SHA-256 `1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed` |

## 结论

Compact-flexible v3 解析先验编码了固定的负手性分支。3600 个源样本中有
569 个获得有限 iota 且原生状态为 `ok`，569 个全部为负 iota。冻结的正手性目标
Adam200 批次包含 120 条轨迹，初始点、最佳点和终点也全部为负 iota；每条轨迹的
全部已记录迭代都留在负侧。

该偏置来自解析构造中的固定符号螺旋相位。绕组面使用
`theta - nfp*phi`，等值线项使用
`m*theta - field_mode*nfp*phi + phase`，所有 `field_mode` 都取正值，构造中未提供
手性符号。随机相位和随机幅值会平移、扭曲等值线，同时保留其波矢方向。由链接数
确定的电流符号统一了轴向安培环流，并未随机化几何手性。

两条代表性端点在独立轴搜索口径下切换到匹配的负手性目标后，总分分别从
`69.6841` 升至 `76.2026`、从 `68.3992` 升至 `81.1498`。随后执行显式负手性
Adam200，`axisv3_case_01341` 达到 `80.8523`，`axisv3_case_02832` 达到
`84.9308`。两条轨迹都突破了 70 分。

这组结果解释了固定正手性目标下的 60--70 分平台：评分目标与先验生成的镜像分支
不一致。先前报告中的 `8.56%` 和 `12.78%` 保留为冻结正手性目标下的分数阈值
统计；它们不再承担“与旧 QUASR/Flow 同手性 QH 收敛域丰度”的含义。几何、线圈
工程、运行时间、Poincare 和 DESC 结果仍按各自原始口径有效。

## `|B|` 图的坐标约定

完整评估绘图函数 `save_periodic_colored_contours` 使用以下固定约定：

- 横轴从 0 增加到 `2*pi`，标注为 field-period angle `NFP phi`；
- 纵轴从 0 增加到 `2*pi`，标注为 Boozer `theta`；
- 数组通过 `closed.T` 与上述坐标对应；
- 两个坐标轴均保持递增方向。

因此，正斜率条纹对应 `theta - zeta = const`，负斜率条纹对应
`theta + zeta = const`，其中 `zeta=NFP*phi`。下面三幅图使用同一绘图函数和同一
坐标方向。旧 QUASR/Flow 参考样本呈正斜率，两个解析先验端点呈负斜率；方向变化
来自磁场分支手性。

| 旧 QUASR/Flow 参考，正手性 | `axisv3_case_01341`，负手性 | `axisv3_case_02832`，负手性 |
|---|---|---|
| ![旧 QUASR Flow 正手性参考的 Boozer B 等值线](assets/axis_surface_prior_handedness_audit_20260902/quasr_flow_reference_boozer_b.png) | ![axisv3_case_01341 的负手性 Boozer B 等值线](assets/axis_surface_prior_handedness_audit_20260902/axisv3_case_01341_boozer_b.png) | ![axisv3_case_02832 的负手性 Boozer B 等值线](assets/axis_surface_prior_handedness_audit_20260902/axisv3_case_02832_boozer_b.png) |

## 群体手性统计

群体审计直接读取 compact-flexible v3 的 3600 条 ABI-11 源评分记录和 120 条冻结
Adam200 轨迹。有限 iota 只存在于 569 条 `status=ok` 记录中，其余状态为
`no_axis` 1982 条、`no_surface` 760 条、`drift_rejected` 267 条和
`flux_rejected` 22 条。

| 样本集合 | 有确定符号 | 负 iota | 正 iota | 跨越 0 |
|---|---:|---:|---:|---:|
| 源样本 `status=ok` | 569 | 569 | 0 | 0 |
| 源样本 score >= 10 | 147 | 147 | 0 | 0 |
| 源样本 score >= 20 | 57 | 57 | 0 | 0 |
| 源样本 score >= 30 | 23 | 23 | 0 | 0 |
| 源样本 score >= 40 | 7 | 7 | 0 | 0 |
| 冻结 Adam200 初始点 | 120 | 120 | 0 | 0 |
| 冻结 Adam200 最佳点 | 120 | 120 | 0 | 0 |
| 冻结 Adam200 终点 | 120 | 120 | 0 | 0 |

569 个有效源样本的 iota 中位数为 `-0.8480`，范围为 `[-2.2816,-0.2546]`。
按 `nc=1,2,3,4` 分组以及按 `nfp=4,5,6,7,8` 分组后，每个非空组仍为 100%
负 iota。对独立等概率正负符号的描述性二项检验给出
`log10(p)=-170.985`；解析构造的确定性符号提供了更直接的因果解释。

![Compact-flexible v3 源样本的分数与 iota 符号分布](assets/axis_surface_prior_handedness_audit_20260902/population/source_handedness.png)

冻结的正手性目标 Adam200 批次中，120 条轨迹都属于 `all_negative` 模式。97 条
最佳分数达到 50 的轨迹和 96 条最佳分数达到 60 的轨迹也全部为负 iota。优化器
在 200 步局部更新内持续停留在同一连通手性域。

![冻结正手性目标 Adam200 的初始 最佳与终点 iota](assets/axis_surface_prior_handedness_audit_20260902/population/adam200_iota.png)

## 固定手性的代码来源

`flow_matching/axis_surface_prior_v2.py` 中的构造包含三组同符号约束：

1. 绕组面三角项采用 `2*theta - nfp*phi + phase`，螺旋起伏采用
   `theta - nfp*phi + phase/2`。
2. 等值线主项依次为 `(m,field_mode)=(1,1),(2,1),(3,2)`；
   `_solve_contour` 将它们统一写成
   `m*theta - field_mode*nfp*phi + phase`。
3. `current_a = copysign(..., link)` 依据参考等值线与构造轴的链接数选择电流方向，
   使总链接电流方向一致。

前两组约束固定了相位传播方向，第三组约束固定了电流与轴环流方向。解析先验 v2
和 v3 共用这套构造，因此两批实验频繁进入负手性具有确定的代码来源。

## 同一几何的双手性评分

双手性审计对两条保存端点执行独立轴搜索，并分别使用 `(M,N)=(1,+nfp)` 与
`(1,-nfp)` 评分。切换目标只改变有符号 QH 相关项；轴、psi、曲面、坐标、iota
和线圈工程分量在同一次独立评估中保持一致。

| 样本 | nfp/nc | 独立 `+nfp` 总分 | 独立 `-nfp` 总分 | `+nfp` 体 QS | `-nfp` 体 QS | 线圈工程 |
|---|---:|---:|---:|---:|---:|---:|
| `axisv3_case_01341` | 5/3 | 69.6841 | 76.2026 | 37.6037 | 53.1991 | 68.2045 |
| `axisv3_case_02832` | 7/4 | 68.3992 | 81.1498 | 35.7253 | 66.0838 | 67.1397 |

冻结正手性续跑保存的分数分别为 `68.8804` 和 `67.2446`。冻结续跑使用优化器
轴延续口径，表中的双手性重算使用历史无关的独立轴搜索。两个正手性列分别服务于
各自的评估模式，数值差异来自轴搜索与延续路径。

### 镜像对照

审计还对每条几何执行 `y` 反射、`z` 反射和绕 `x` 轴旋转 `pi`。空间反射交换
目标手性，正旋转保留目标手性。

| 样本 | 原几何 `-nfp` 与 y 镜像 `+nfp` 差值 | 原几何 `-nfp` 与 z 镜像 `+nfp` 差值 | 正旋转同目标差值 |
|---|---:|---:|---:|
| `axisv3_case_01341` | 0.18780 | 0.19859 | 0.01076 |
| `axisv3_case_02832` | 0.00683 | 0.00685 | 0.0000154 |

所有镜像对照的线圈工程分量严格相同。B 的总分对称性达到约 `7e-3`，A 的残差约
`0.2`，主要落在轴搜索相关的 psi、坐标和体 QS 数值路径。镜像交换手性的结论在
两条样本上成立；A 同时给出了该离散轴搜索口径下的数值容差尺度。

## 显式负手性 Adam200

注册实验协议为
`qh-axis-surface-compact-v3-top2-negative-hand-continue-adam200-64d-abi11-v1`。
每条样本使用一张 P107 GPU 并行运行。优化设置如下：

| 项目 | 设置 |
|---|---|
| 参数空间 | 保存端点的精确、未裁剪标准化线圈系数 |
| Flow 调用 | 0 |
| 目标 | `(M,N)=(1,-nfp)` |
| 更新 | Adam 200 步 |
| 梯度估计 | 每步 64 个新鲜随机正交方向，中心差分 |
| `h` / 学习率 | `0.0025` / `0.01` |
| beta | `(0.7,0.999)` |
| 原生评分 | ABI-11，psi grid 48 |

| 样本 | 目标 | 初始分 | 最佳分 | 最佳步 | 增益 | 终点分 | 最佳步 iota | 用时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `axisv3_case_01341` | `(1,-5)` | 75.8320 | **80.8523** | 196 | +5.0203 | 79.2414 | -1.02436 | 957.0 s |
| `axisv3_case_02832` | `(1,-7)` | 79.9665 | **84.9308** | 103 | +4.9643 | 80.1964 | -1.18906 | 1518.1 s |

两条 worker 状态均为 `complete`，各完成 200 次更新。A 的最佳体 QS 分量为
`64.5883`，线圈工程分量为 `61.2179`；B 的对应值为 `78.6028` 和
`63.6072`。两条轨迹的负手性目标从起点即超过 70，Adam200 又分别取得约 5 分
增益。

![两条显式负手性 Adam200 的总分与 iota 轨迹](assets/axis_surface_prior_handedness_audit_20260902/negative_adam200/analysis/negative_hand_adam200.png)

分量图把三种口径分开显示：保存源几何的独立 `+nfp` 评分、同一源几何的独立
`-nfp` 评分、负手性 Adam200 最佳点。目标切换首先提高体 QS 分量；后续优化继续
提高体 QS 和部分几何分量，同时允许线圈工程分量发生权衡。

![双手性重算与负手性 Adam200 最佳点的评分分量](assets/axis_surface_prior_handedness_audit_20260902/negative_adam200/analysis/signed_score_components.png)

## 结论边界与后续设计

本次审计保留以下结论：

- v2/v3 源样本数量、状态、运行时间和线圈工程统计；
- 两条已完整评估端点的曲面几何、Poincare、线圈工程和 DESC 结果；
- 所有 ABI-11 总分作为其清单所写有符号目标下的冻结数值；
- compact-flexible v3 在负手性目标下存在可超过 80 的可优化样本。

以下解释保持撤回状态：

- 将固定正手性目标下的 8.56% 与 12.78% 直接称为解析先验 QH 丰度；
- 将 v2/v3 与旧 QUASR/Flow 的正手性 QH 簇直接比较；
- 把 60--70 平台完全归因于 QS 本身或线圈工程限制。

下一版解析先验需要把 `chirality_sign in {-1,+1}` 作为显式生成参数，并同步作用于
绕组面螺旋项、等值线 field mode、目标元数据和评分目标。首个回归组应包含严格
镜像配对、50/50 手性抽样、按手性分层的合法率和 score 分布，以及镜像前后线圈
工程不变量。完成该合同后，可以选择均匀双手性探索，或把全部样本规范化到项目
默认的正手性目标。该改动仍处于设计阶段，项目默认协议继续使用
`qh-flow-screen32-adam200-64d-abi11-v1`。

## 证据与复现

- 群体审计作业：`52580`；镜像与双手性评分作业：`52581`。
- 负手性 smoke 作业：`52565`；两样本并行正式作业：`52566`。
- 负手性运行根目录：
  `/home/scc/pb24511935/local_surface_evaluator_runs/axis_surface_prior_v3_negative_hand_continue_20260902_729e592`。
- 群体统计：
  [`population/summary.json`](assets/axis_surface_prior_handedness_audit_20260902/population/summary.json)。
- 镜像评分：
  [`signed_mirror_scores.json`](assets/axis_surface_prior_handedness_audit_20260902/signed_mirror/signed_mirror_scores.json)。
- 负手性结果汇总：
  [`negative_hand_results.json`](assets/axis_surface_prior_handedness_audit_20260902/negative_adam200/analysis/negative_hand_results.json)。
- 负手性逐步历史：
  `assets/axis_surface_prior_handedness_audit_20260902/negative_adam200/trajectories/*/optimization/history.jsonl`。
- 冻结评分库 SHA-256：
  `1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed`。
- 冻结 checkpoint SHA-256：
  `39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f`。
