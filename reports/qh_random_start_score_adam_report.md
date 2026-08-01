# QH 随机起始分数与 Adam 可优化性实验报告

日期：2026-07-31  
条件：QH，$N_{\mathrm{FP}}=4$，3 根基线圈

## 1. 实验问题

本实验只研究一个问题：从 flow prior 的 IID 标准高斯潜变量出发时，起始 native score 的高低是否会影响后续标准 Adam 的优化效果。

这里的“随机起点”严格定义为

$$
z_0\sim\mathcal N(0,I),\qquad z_0\in\mathbb R^{3\times100}.
$$

本实验不使用 proxy 排序、proxy 优化、CEM、QUASR 反演样本或其他预筛选分布。这样得到的分数分布和 Adam 起点都属于同一个自然 flow prior，避免把“起始 score 的作用”与“起点生成方法不同”混在一起。

高分标准按本轮约定处理：优先把 $S\geq50$ 视为高分；若 4096 个新增 IID 样本中仍无 $S\geq50$，则把 $S\geq40$ 作为本实验可获得的高分层，同时把 $S\geq50$ 明确报告为空的极高分层，而不是事后降低标准并隐去这一事实。

## 2. 随机起点评分流程

每个随机潜变量经过完全相同的固定流程：

$$
z_0
\xrightarrow[256\ \text{steps}]{\text{FP32 RK4 flow}}
x
\xrightarrow{\text{normalizer}^{-1}}
\text{coil parameters}
\xrightarrow{\text{current native CUDA score}}
S.
$$

其中 flow checkpoint 固定为 30000 step EMA，SHA-256 为
`39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f`；native score 动态库固定为修复全局电流反号 bug 后的版本，SHA-256 为
`0b7342db471788385931385c25ded8095c72cfb7fcea1e21376a0475dafaa427`。

4096 个潜变量由单一 seed `20260805` 一次生成并保存。`case_id` 同时索引潜变量数组和 score 结果，后续起点面板要求 ID 为 $0,\ldots,4095$ 的一一映射并记录两个源文件的 SHA-256，防止高分记录与错误潜变量错配。

flow 解码在一张 RTX 5090 上批量执行。native score 使用四个持久 C++/CUDA worker，每张已分配 RTX 5090 对应一个 worker。Python 只负责批量调度和产物整理，不实现数值评分热路径。

## 3. Adam 实验定义

对选中的每个随机起点，目标函数都是

$$
S(z)=\operatorname{native\_score}\!\left(F_\theta(z)\right).
$$

每轮生成 $m=4$ 个相互正交且单位 RMS 的方向 $u_j$，使用扰动 $c=0.01$ 估计

$$
\hat g_t=
\frac{1}{4}\sum_{j=1}^{4}
\frac{S(z_t+c u_j)-S(z_t-c u_j)}{2c}u_j.
$$

随后执行最大化形式的标准 Adam：

$$
m_t=0.9m_{t-1}+0.1\hat g_t,
$$

$$
v_t=0.999v_{t-1}+0.001\hat g_t^2,
$$

$$
z_{t+1}=z_t+0.003\frac{\hat m_t}{\sqrt{\hat v_t}+10^{-8}}.
$$

没有权重衰减、学习率调度、梯度差截断、更新截断、参数截断、先验惩罚、proposal 搜索、回溯或 accept/reject。每轮固定评估 8 个正负扰动端点和 1 个更新后中心点。flow 解码仍为 FP32 RK4-256，score 仍为同一个修正后原生实现。

为减少随机梯度方向造成的混杂，所有分数层使用相同的方向随机种子，即共同随机数设计。每个分数层选择多个不同 IID 潜变量，以观察同一分数范围内部的几何差异。

## 4. 结果

### 4.1 4096 个 IID 起点的 native score 分布

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 4096 |
| mean | 4.3412 |
| median | 3.4938 |
| P90 | 9.4202 |
| P95 | 11.7650 |
| P99 | 21.0310 |
| P99.5 | 25.2304 |
| max | **41.0501** |
| `status=ok` | 2149 / 4096，52.47% |
| `status=ok` 内 mean / median | 8.0551 / 7.5072 |

![4096 个 IID 随机起点的 score 分布](assets/qh_random_score_pool_29960/iid_random_score_distribution.png)

高分尾部计数为：

| 门槛 | 数量 | 占全部样本比例 |
| ---: | ---: | ---: |
| $S\geq10$ | 326 | 7.959% |
| $S\geq20$ | 45 | 1.099% |
| $S\geq30$ | 6 | 0.1465% |
| $S\geq40$ | 1 | 0.0244% |
| $S\geq50$ | **0** | **0%** |

这给出了比原来 256 样本更清楚的尾部结论：自然 flow prior 并非完全不能产生较好的样本，但 30 分以上已经很少，40 分以上在 4096 次中只出现一次，50 分以上没有出现。因此后续若要求“随机 Adam 有多大概率从高分盆地启动”，不能把 15 或 20 分重新命名为高分；本实验按约定以 40 分作为实际可获得的高分层，并把 50 分以上报告为空的极高分层。

失败状态由 842 个 `no_axis`、268 个 `no_surface`、819 个 `drift_rejected` 和 18 个 `flux_rejected` 组成。总体低分的一部分来自无法通过磁轴或磁面门控，而不是所有样本都在同一个可行物理区域内连续变化。因此 Adam 分析要同时报告全体轨迹和从 `status=ok` 起点出发的轨迹。

### 4.2 Adam 起点面板

在完整分布上选择了 12 个不同潜变量：

| 起点 | `case_id` | 记录 score | 初始状态 | latent RMS |
| ---: | ---: | ---: | --- | ---: |
| 0 | 1375 | 0.0908 | `no_axis` | 1.0905 |
| 1 | 3273 | 2.0157 | `ok` | 0.9875 |
| 2 | 1164 | 5.0042 | `ok` | 0.9908 |
| 3 | 2414 | 7.9996 | `ok` | 1.0126 |
| 4 | 568 | 9.9966 | `ok` | 1.0005 |
| 5 | 2683 | 11.9833 | `ok` | 1.0069 |
| 6 | 548 | 14.9835 | `ok` | 1.0093 |
| 7 | 2044 | 19.7630 | `ok` | 0.9853 |
| 8 | 1220 | 24.9121 | `ok` | 0.9618 |
| 9 | 3549 | 29.8723 | `ok` | 0.9518 |
| 10 | 2912 | 38.6943 | `ok` | 0.9538 |
| 11 | 132 | **41.0501** | `ok` | 0.9200 |

![IID 分布与 Adam 起点覆盖](assets/qh_score_adam_start_panel_29960/iid_score_distribution_and_starts.png)

面板不是 12 个分数箱的频率抽样，而是为研究 score landscape 主动选取的连续覆盖。因此它可以回答“不同起始 score 下的中短程 Adam 轨迹有何差异”，不能用来重新估计自然先验中的成功概率；成功概率只能由上一节未重加权的 4096 样本统计给出。

### 4.3 随机池耗时与验收

| 阶段 | 资源 | 耗时 |
| --- | --- | ---: |
| 4096 个潜变量 FP32 RK4-256 解码 | 1 x RTX 5090 | 38.34 s |
| 4096 次 corrected native score | 4 x RTX 5090 | 5165.32 s |
| 合计核心计算 | 4 卡作业 | 5203.66 s，约 86.73 min |
| 平均每样本 score 墙钟摊销 | 4 卡 | 1.261 s |

四张卡的 preflight 与 postflight 均为 2 MiB、0% 利用率，说明计时没有与其他计算进程重叠，任务结束后也没有遗留 score worker。manifest 的 flow checkpoint 和 score 动态库哈希均与本报告第 2 节一致。

原始结果位于 [随机池目录](assets/qh_random_score_pool_29960/)，起点与映射位于 [Adam 面板目录](assets/qh_score_adam_start_panel_29960/)。`scored_cases.jsonl` 和 `random_latents.npz` 的 SHA-256 分别为 `49cccc0d7b6dcb8aa8a7f9e620f897817610278a2edac215aa77edcd02a9abb8` 与 `88bdeefab57f1d2f0320fb4cc339ae3a374eb25243b6ff2f70ccad614d16ea12`。

### 4.4 Adam 结果

Slurm 控制器恢复后，单步烟测 `29992` 先通过；正式数组 `29996` 完成起点 0--9，补充数组 `30025` 完成起点 10--11。12 个数组元素均为 `COMPLETED 0:0`，每条轨迹都严格执行 40 步。

| 起点 | 初始 score | 10 步最佳 | 20 步最佳 | 30 步最佳 | 40 步最佳 | 增益 | 总耗时 / s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.091 | 0.356 | 0.364 | 0.364 | 0.364 | 0.273 | 678.1 |
| 1 | 2.013 | 4.663 | 4.675 | 4.681 | 4.769 | 2.755 | 828.9 |
| 2 | 5.001 | 6.303 | 6.864 | 6.980 | 8.002 | 3.001 | 901.2 |
| 3 | 8.002 | 10.061 | 14.375 | 24.482 | **27.366** | **19.363** | 851.4 |
| 4 | 9.993 | 10.861 | 11.745 | 12.691 | 13.306 | 3.313 | 827.8 |
| 5 | 11.976 | 12.847 | 13.885 | 15.252 | 16.305 | 4.329 | 830.3 |
| 6 | 14.955 | 15.245 | 15.825 | 16.717 | 17.740 | 2.784 | 825.8 |
| 7 | 19.748 | 21.863 | 25.555 | 29.858 | **33.648** | **13.900** | 895.3 |
| 8 | 24.889 | 28.593 | 30.789 | 33.880 | **36.407** | **11.518** | 938.0 |
| 9 | 29.879 | 32.663 | 35.563 | 38.002 | **39.395** | **9.516** | 863.7 |
| 10 | 38.659 | 40.823 | 45.435 | 46.152 | **47.201** | **8.542** | 1044.2 |
| 11 | 41.053 | 42.580 | 43.828 | 44.650 | **45.241** | **4.188** | 856.6 |

![不同 IID 起始 score 下的 Adam 轨迹与结果](assets/qh_score_adam_start_sweep_29996/initial_score_vs_adam_outcome.png)

初始 score 与 40 步最佳 score 的 Pearson / Spearman 相关系数为 `0.940 / 0.951`。这个强相关主要说明较好的起点在相同预算后通常仍然较好；它不能直接说明高分点更容易优化，因为最佳 score 自身包含初始 score 这一基线。

更直接反映可优化难度的“最佳增益”与初始 score 的 Pearson / Spearman 相关系数只有 `0.249 / 0.517`。具体轨迹也显示明显的局部盆地差异：8.00 分起点获得了全组最大的 19.36 分增益，而相邻的 9.99、11.98、14.96 分起点只增加 2.78--4.33 分；19.75--29.88 分的三个起点则稳定增加 9.52--13.90 分。最高的 41.05 分起点最终只有 4.19 分增益，且低于 38.66 分起点的最终 47.20 分。

被拒绝的 0.091 分起点没有被 40 步 Adam 救回可用区域，最佳值仅 0.364。其余 11 个 `status=ok` 起点都提高了分数，但没有一条在 40 步内达到 50 分。除拒绝起点在第 12 步达到最佳外，其余 11 条轨迹都在第 40 步刷新最佳，因此这些结果是统一中短程预算下的响应，不是各起点的收敛上限。

12 条轨迹累计计算耗时为 10,341.3 s，平均每条 861.8 s（14.36 min）。高分样本通常进入 native score 的更多阶段，因此单条耗时在 678--1,044 s 之间变化。完整结果位于 [Adam sweep 目录](assets/qh_score_adam_start_sweep_29996/)；机器可读汇总为 [sweep_summary.json](assets/qh_score_adam_start_sweep_29996/sweep_summary.json)，逐轨迹压缩汇总为 [trajectory_summary.jsonl](assets/qh_score_adam_start_sweep_29996/trajectory_summary.jsonl)。

## 5. 结论与边界

4096 个无偏 IID 样本给出的自然先验结论不变：$S\geq40$ 约为 $2.4\times10^{-4}$ 的稀有事件，本轮没有观测到 $S\geq50$。因此随机抽样本身很难频繁提供极高分 Adam 起点。

对已经抽到的起点，初始 score 对固定预算后的绝对质量有很强预测力，但对可优化增益只有有限预测力。真正决定短程 Adam 能否大幅上升的是局部 landscape，而不是初始 score 一个标量。约 20--30 分的三个起点在本面板中都表现出良好、连续的上升，说明这一层是有价值的优化起点；但 8 分起点的异常大增益和 10--15 分层的平缓轨迹同时表明，不能仅凭 score 阈值推断盆地宽度。

本面板是为了覆盖 score 轴而主动选出的 12 个点，不是按自然先验频率抽取的独立重复试验。每个分层样本数很少，尤其 40 分以上只有一个，因此不能从 12 条轨迹估计“随机 Adam 成功率”，也不能把相关系数当作总体精确值。若后续目标是预测哪些随机起点值得投入长程 Adam，需要额外学习或测量局部平滑度、梯度一致性和可行域裕量，而不应只按初始 score 排序。

## 6. 当前最佳 47.2006 样本的完整物理评估

### 6.1 验收结论

12 条轨迹中最高分的 `start_10` 进一步通过了固定完整评估。结论是：**该样本确有较大、可守护求解的嵌套 QH 磁面，DESC 也把力误差降到小量；但 QH 起伏仍明显，和 47.20 的中等 score 一致。**

这次没有复用此前 71.7342 样本的 `a` 或 `s`。源 $\psi$ 并行测试了 `a=0.04,0.05,0.06,0.08`；四者的训练/验证误差均稳定，本样本实测选择 `a=0.08`，因为它在误差仍小的前提下覆盖了最大的通过半径：

| $a$ | $\psi$ 验证 RMS | 方向误差 P95 | 最大廉价筛选通过层 | 通过层平均半径 / m | 首个外层失败 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.04 | $6.79\times10^{-4}$ | $4.91\times10^{-5}$ | 0.49 | 0.02847 | 0.64 |
| 0.05 | $6.89\times10^{-4}$ | $6.06\times10^{-5}$ | 0.49 | 0.03557 | 0.64 |
| 0.06 | $7.02\times10^{-4}$ | $7.17\times10^{-5}$ | 0.49 | 0.04269 | 0.64 |
| **0.08** | **$7.38\times10^{-4}$** | **$9.59\times10^{-5}$** | **0.49** | **0.05716** | **0.64** |

磁轴闭合残差为 $4.52\times10^{-9}$。`a=0.08` 的误差虽略高于更小拟合域，但仍处于 $10^{-4}$ 方向误差量级，并把已验证物理半径扩展一倍，因此选择依据不是“误差最小即最好”，而是误差与有用体积的共同约束。

### 6.2 $\alpha+\nu+$ guarded 外扩

在选中的源 $\psi$ 上测试 `s=0.24,0.36,0.49,0.64`。`s=0.36` 是最大通过面，紧邻的 `s=0.49` 已被离网格残差和法向场门槛明确拒绝：

| $s$ | 判定 | $|V|$ / $\mathrm{m}^3$ | $\iota$ | 离网格 relative $L^2$ | 法向场 P95 | 相对 $\psi$ 面距离 P95 / mm |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.24 | 通过 | 0.03088 | 1.6890 | $2.58\times10^{-5}$ | $4.08\times10^{-5}$ | 0.135 |
| **0.36** | **通过并选中** | **0.04741** | **1.6971** | **$3.06\times10^{-5}$** | **$4.66\times10^{-5}$** | **0.185** |
| 0.49 | 拒绝 | 0.06353 | 1.6847 | $2.51\times10^{-2}$ | $1.64\times10^{-2}$ | 5.045 |
| 0.64 | 拒绝 | 0.05365 | 1.6431 | $1.27\times10^{-1}$ | $1.62\times10^{-1}$ | 47.395 |

选中层的 $\alpha$ 使用 120,000 个训练点、60,000 个验证点和 $(L,M,N)=(12,12,16)$。验证 relative $L^2$ 为 0.09652，$\min(1+\lambda_\theta)=0.2547>0$，拟合得到 $\iota=1.6781$。12 阶 $\nu$ 把表面 Simsopt relative $L^2$ 从 0.20148 降到 0.006801，映射 Jacobian 最小值为 0.6508。guarded 修正接受 5 步后得到

$$
\iota=1.697113,\qquad G=-6.943557,\qquad |V|=0.047414\ \mathrm{m}^3.
$$

最终面相对 $\psi$ 等值面的线性化法向距离 P95 为 0.185 mm；相对 $\alpha+\nu$ 初始面的双向位移 P95 为 1.787 mm，低于本面 2.353 mm 的保护阈值。

### 6.3 场线、$|B|$ 与线圈几何

8 条场线在四个环向截面上各取得 25 个命中，未越过候选边界；该采样没有显示逃逸或宏观磁岛。点数只够做验收级检查，不把这张稀疏图过度解释为高分辨率拓扑证明。

![47.2006 样本最大通过面上的庞加莱验证](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/assets/poincare.png)

直接 Boozer 面上的 $|B|$ 范围为 0.65933--0.80806 T，平均值为 0.73579 T。白底彩色等高线显示清楚的 QH 斜条纹，但闭合畸变和非理想起伏仍明显；这与 native score 的 `volume_qs=34.39` 一致，并不是“高 score 被完整评估推翻”。

![47.2006 样本直接 Boozer 面的彩色 |B| 等高线](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/assets/boozer_b.png)

![47.2006 样本完整线圈和最大通过面](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/assets/coils_surface.png)

交互产物：[Boozer $|B|$ HTML](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/assets/boozer_b.html)；[完整设备线圈与磁面 HTML](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/assets/coils_surface.html)。

### 6.4 DESC 复核

DESC 使用真实 Biot--Savart 场积分得到的环向磁通 $-3.68289\times10^{-3}\,\mathrm{Wb}$；它与 $\alpha$ 标定的 $-3.70349\times10^{-3}\,\mathrm{Wb}$ 相差约 0.56%，构成独立实现间的量级交叉检查。当前 DESC 环境明确为 JAX CPU。

| DESC 指标 | 结果 |
| --- | ---: |
| 初始 / 最终嵌套 | true / true |
| 初始 mean / P95 / max 归一化力误差 | 1.092 / 2.121 / 5.456 |
| 最终 mean / P95 / max 归一化力误差 | $2.80\times10^{-3}$ / $6.75\times10^{-3}$ / $1.55\times10^{-2}$ |
| optimizer cost / optimality | 0.009141 / 0.002406 |
| 迭代 / 函数评估 | 50 / 57 |
| optimizer success | false，达到迭代上限 |

求解前后体均保持嵌套，最终力误差通过当前验收门槛；但优化器达到 50 步上限，因此只能写成“物理结果通过”，不能写成“优化器形式收敛”。以下逐张引用本次全部 8 张 DESC 图。

![DESC initial boundary, score 47.2006](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/desc/boundary_initial.png)

![DESC final boundary, score 47.2006](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/desc/boundary.png)

![DESC iota versus rho, score 47.2006](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/desc/iota.png)

![DESC Boozer |B| colored contours, score 47.2006](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/desc/boozer_B.png)

![DESC Boozer modes versus rho, score 47.2006](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/desc/boozer_modes.png)

![DESC QA components versus rho, score 47.2006](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/desc/qs_QA.png)

![DESC QH components versus rho, score 47.2006](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/desc/qs_QH.png)

![DESC QP components versus rho, score 47.2006](assets/qh_score_adam_start10_47p200_full_eval_20260731/full/desc/qs_QP.png)

### 6.5 耗时和原始产物

| 阶段 | 资源 | 墙钟 |
| --- | --- | ---: |
| 4 个 source $\psi$ 候选 | 4 x RTX 5090 并行 | 1 min 15 s；单候选数值载荷约 9.8 s |
| 已知选中层的 $\alpha+\nu+$ guarded | 1 x RTX 5090，FP32 | 4 min 45 s |
| 稀疏外层边界发现 | 先 1 项、再 3 项并行 | 10 min 37 s |
| 可视化与 CPU DESC | 16 CPU | 5 min 40 s |
| 已知参数单路径 | source + 选中层 + 下游 | 约 **11 min 40 s** |

选中候选内部，$\alpha$ 总耗时 148.05 s，其中磁通标定 63.28 s、FP32 GPU QR 0.87 s；$\nu$ 总耗时 82.87 s。下游内部可视化 41.92 s、DESC 232.51 s，其中 DESC solve 为 142.38 s。

完整原始产物位于 [47.2006 完整评估目录](assets/qh_score_adam_start10_47p200_full_eval_20260731/)。选中 `boozer_guarded.npz` 的 SHA-256 为 `db0895246a74d93622763292292ee03d26e7ff0348e15f9bac02b54755af3965`，DESC `equilibrium.h5` 的 SHA-256 为 `399ddbb4afaeeaa5a497145c4ee74ea0587ef2a18ba5d2a9bc72d2ed64ecf7c7`。

## 7. $\eta=0.01$ 阶段最佳 58.1514 样本的完整评估

### 7.1 结论和 score 分解

该样本来自 `start_10` 的中断长轨迹，在第 52 步达到当前最佳 score 58.15137。这里先完成物理验收；后续 Adam 补跑尚在进行，因此 58.15137 不是 200 步最终上限。

完整评估结论为：**该线圈具有可由标准 LS/Newton 找到并经离网格验证的较大嵌套 QH 磁面；庞加莱和 DESC 都保持嵌套，DESC 最终力误差进入 $10^{-2}$ 量级。它明显优于普通随机样本，但面上仍有可见非 QH 起伏，不是接近完美的 QH 位形。**

当前 native score 及全部分量如下。所有分量均为 0--100 且越大越好。

| 项目 | 数值 |
| --- | ---: |
| total | **58.15137** |
| axis | 98.26035 |
| psi | 96.90558 |
| surface | 96.69667 |
| coordinate | 85.54687 |
| volume QS | 33.28596 |
| iota | 100.00000 |
| coil | 64.84444 |

native score 的体诊断给出 $\iota_{\min}=1.86671$、有效小半径约 0.03559 m、体积 0.02539 $\mathrm{m}^3$、体 QH residual 0.45172 和边缘 residual 0.59613。这里的体 residual 与后文单个标准 Boozer 面的面 QS error 使用不同统计定义和归一化，不能直接比较绝对数值；二者一致表明“QH 占优但仍有明显误差”。本次实际评估的输入已独立固化为 [evaluated_case.json](assets/qh_score_adam_eta001_58p151_full_eval_20260801/evaluated_case.json)，SHA-256 为 `e7a33bd80b660761d77b88f7308ac26720bceecc7d05fe71145b9a018d2ede18`；它不会被后续 `start_10` 续跑覆盖。

### 7.2 修正后的选面判据

本次源 $\psi$ 独立测试了 $a=0.04,0.05,0.06,0.08$，最终选择 $a=0.06$。其验证 RMS 为 $8.45\times10^{-4}$，方向误差 P95 为 $1.30\times10^{-4}$；廉价场线筛选通过到 $s=0.49$，对应平均小半径 0.04281 m，并在 $s=0.64$ 明确失败。选择依据仍是误差与物理覆盖范围的折中，不复用别的样本的 $a$ 或 $s$。

旧 `guarded` 求解器每次只允许一步 Newton，并要求中间残差单调、位移小且始终贴近初始 $\psi$ 面。它适合防止跳支，但不能判断磁面是否存在。本样本的 guard 对多个内层候选也给出拒绝；改用完整标准 LS/Newton 后，$s=0.24$ 和 0.36 均收敛并通过独立 $97\times97$ 网格检查。因此正式判据已改为：

1. 从 $\alpha+\nu$ 曲面开始，标准 LS 最多 100 步、完整 Newton 最多 30 步；
2. 最终独立网格 relative $L^2$ 和法向场 P95 均不高于 $10^{-4}$；
3. 环向绕行方向正确、曲面法向不退化；
4. 初始 $\psi$ 误差和最终位移只作跳支诊断，不作逐步硬门槛。

四个较大候选的结果如下：

| $s$ | 标准 LS/Newton | $|V|$ / $\mathrm{m}^3$ | $\iota$ | 离网格 relative $L^2$ | 法向场 P95 | 面 QH error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.24 | 通过 | 0.01679 | 1.88299 | $2.77\times10^{-5}$ | $4.42\times10^{-5}$ | $7.44\times10^{-5}$ |
| **0.36** | **通过并选中** | **0.02623** | **1.89480** | **$3.06\times10^{-5}$** | **$4.99\times10^{-5}$** | **$1.675\times10^{-4}$** |
| 0.49 | 离网格拒绝 | 0.03767 | 1.98791 | $7.94\times10^{-4}$ | $1.19\times10^{-3}$ | 未采用 |
| 0.64 | 离网格拒绝 | 0.05028 | 1.99231 | $1.65\times10^{-3}$ | $2.91\times10^{-3}$ | 未采用 |

$s=0.49/0.64$ 在求解配点网格上也返回 `success=true` 且残差约 $10^{-13}$，但离网格误差高出门槛 8--29 倍，并分别偏离初始面约 0.33、0.50 个小半径。这是欠分辨配点伪解，说明最终标准不能简化成只看 Simsopt 的 `success`；有效定义是“LS/Newton 收敛并通过独立连续场近似检查”。

选中面的 QH error 为 $1.67508\times10^{-4}$；同一面上的 QA/QP error 分别为 $2.31213\times10^{-3}$ 和 $2.40960\times10^{-3}$，QH 比两个竞争对称性约低一个数量级。标准 LS 用时 7.06 s，随后 Newton 在第 0 步已满足精确方程；这说明 $\alpha+\nu$ 初值位于正确吸引域，但初值本身尚未达到最终离网格精度。

### 7.3 庞加莱、$|B|$ 和三维几何

8 条场线在四个截面上各得到 25 个命中，全部保持在选中边界内；图中没有宏观逃逸或明显磁岛散射。它仍是验收级稀疏追踪，不替代专门的高分辨拓扑研究。

![58.1514 样本选中面庞加莱验证](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/assets/poincare.png)

直接标准 Boozer 面上的 $|B|$ 为 0.63950--0.76461 T，平均 0.70192 T。白底彩色等高线呈稳定 QH 斜条纹，同时保留局部闭合畸变，和中等 `volume_qs` 分量一致。

![58.1514 样本直接 Boozer 面彩色 |B| 等高线](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/assets/boozer_b.png)

![58.1514 样本完整线圈和选中磁面](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/assets/coils_surface.png)

交互产物：[Boozer $|B|$ HTML](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/assets/boozer_b.html)；[完整设备线圈与磁面 HTML](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/assets/coils_surface.html)。

### 7.4 DESC 复核

DESC 使用环向磁通 $-1.86018\times10^{-3}\,\mathrm{Wb}$，当前明确走 16 CPU 路径。初始和最终体都保持嵌套。优化器达到 50 步上限而形式上 `success=false`，但最终归一化力误差已通过当前默认验收：

| DESC 指标 | 结果 |
| --- | ---: |
| 初始 / 最终嵌套 | true / true |
| 初始 mean / P95 / max 归一化力误差 | 0.9123 / 2.0203 / 4.2988 |
| 最终 mean / P95 / max 归一化力误差 | $3.04\times10^{-3}$ / $6.63\times10^{-3}$ / $1.31\times10^{-2}$ |
| optimizer cost / optimality | 0.06282 / 0.00622 |
| 迭代 / 函数评估 | 50 / 62 |
| optimizer success | false，达到迭代上限 |

以下逐张引用本次全部 8 张成功 DESC 图：

![DESC initial boundary, score 58.1514](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/desc/boundary_initial.png)

![DESC final boundary, score 58.1514](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/desc/boundary.png)

![DESC iota versus rho, score 58.1514](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/desc/iota.png)

![DESC Boozer |B| colored contours, score 58.1514](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/desc/boozer_B.png)

![DESC Boozer modes versus rho, score 58.1514](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/desc/boozer_modes.png)

![DESC QA components versus rho, score 58.1514](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/desc/qs_QA.png)

![DESC QH components versus rho, score 58.1514](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/desc/qs_QH.png)

![DESC QP components versus rho, score 58.1514](assets/qh_score_adam_eta001_58p151_full_eval_20260801/full_standard_s_0p36/desc/qs_QP.png)

### 7.5 耗时和产物

| 阶段 | 资源 | 墙钟 |
| --- | --- | ---: |
| 4 个 source $\psi$ 候选 | 4 x RTX 5090 并行 | 1 min 17--20 s |
| 单候选 $\alpha$ | 1 x RTX 5090，FP32 | 166.0 s |
| 三半径 $\nu$ 和曲面重参数化 | 1 x RTX 5090 + CPU | 617.5 s |
| 4 个标准 LS/Newton 候选 | 4 x RTX 5090 并行 | 1 min 04--22 s |
| 可视化、庞加莱与 CPU DESC | 16 CPU | 5 min 40 s |

完整原始产物位于 [58.1514 完整评估目录](assets/qh_score_adam_eta001_58p151_full_eval_20260801/)。选中 `boozer_standard.npz` 的 SHA-256 为 `ac7fa3430e0ce3ed8ef3a44a4a655adb20b20067b30485e6941336d9d727f5f7`；DESC `equilibrium.h5` 的 SHA-256 为 `49bf4ebe5d17ca5ebde5c76a435433efd957c03858f7e6c39d264a7c7f43f6de`。下游内部可视化为 38.18 s、DESC 为 239.08 s，总计 330.40 s。

## 8. Adam 的 $\beta_1,\beta_2$ 与局部二阶阶段

### 8.1 当前更新式和时间尺度

当前 `start_10` 有 3 个基线线圈，因此潜变量维数为 $d=3\times100=300$。每一步只抽取 $k=4$ 个正交方向，在每个方向上计算一对中心差分：

$$
\widehat g_t=\frac{1}{k}\sum_{j=1}^{k}
\frac{f(x_t+h u_{t,j})-f(x_t-h u_{t,j})}{2h}u_{t,j},
\qquad h=0.01.
$$

随后使用没有权重衰减、裁剪或接受拒绝的标准 Adam：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)\widehat g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)\widehat g_t^2,
$$

$$
x_{t+1}=x_t+\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

目前使用 $(\beta_1,\beta_2)=(0.9,0.999)$。它们的典型记忆长度约为 $1/(1-\beta)$，分别是 10 步和 1000 步。对只有 40--200 步的黑箱优化而言，$\beta_2=0.999$ 在绝大部分时间内接近“保留全历史二阶尺度”，而不是快速适应当前局部盆地。由于每步只观测 4 个方向，$\beta_1$ 的平滑是必要的；不能仅凭更新变小便断言 $\beta_1=0.9$ 过于保守。

### 8.2 现有轨迹给出的直接证据

`start_10` 的长轨迹已经两次跨过 score 中的硬可行性边界，产生了远大于平滑区梯度的异常差分。

| 事件 | 扰动端点 | 方向差 | 梯度 RMS | 更新 RMS | 后果 |
| --- | --- | ---: | ---: | ---: | --- |
| 第 51 步 | 2/8 为 `drift_rejected` | 最大绝对值 57.86 | 1020.34 | 0.00651 | 污染一、二阶矩 |
| 第 88--93 步 | 全部 `ok` | 平滑 | 11.64--18.26 | 0.00081 降至 0.00067 | score 从 59.0125 升至 59.3632 |
| 第 94 步 | 2/8 为 `no_axis` | -58.24、-58.12 | 1028.54 | 0.00643 | 中心 score 立即降至 58.2751 |
| 第 95--100 步 | 全部 `ok` | 恢复正常 | 10.40--21.06 | 0.00580 降至 0.00350 | 中心 score 继续降至 56.6093 |

第 94 步尤其明确：到第 93 步时算法正在用小步稳定上升，但下一步因两个探针越过 `no_axis` 门槛，更新尺度放大约 9.6 倍并离开当前最佳点。到第 134 步，历史最佳仍为 59.3632，更新 RMS 已降到 $5.27\times10^{-4}$。这说明当前问题同时包含两部分：

1. $\beta_2=0.999$ 对异常二阶矩遗忘过慢。第 94 步异常在 40 步后的相对权重仍约为 $0.999^{40}=0.961$；若 $\beta_2=0.99$ 则为 0.669，若为 0.95 则为 0.129。
2. 单纯降低 $\beta_1$ 或 $\beta_2$ 不能保证更好。更短的记忆会更快忘掉异常，但也会让仅由 4 个方向得到的高方差梯度更直接地控制更新；跨硬门槛当步仍可能过冲。

因此，现有证据支持把 $\beta_2=0.999$ 视为偏保守候选，但尚不支持直接指定唯一的新值。$\beta_1=0.9$ 目前也没有“明显过大”的证据；$\beta_1=0.95$ 的约 20 步记忆反而很可能加重越过极值后的惯性。

### 8.3 决定 beta 的最小对照实验

后续应复用完全相同的 IID 起点、方向随机种子、$h=0.01$ 和 $\eta=0.01$，先比较四个有明确含义的配置：

| 配置 | 目的 |
| --- | --- |
| $(0.9,0.999)$ | 当前基线 |
| $(0.9,0.99)$ | 只缩短二阶矩记忆，隔离 $\beta_2$ 作用 |
| $(0.8,0.99)$ | 同时提高一阶矩响应速度 |
| $(0.5,0.99)$ | 检查低动量是否被四方向噪声破坏 |

证据不能只看最终中心值，而应同时比较前 10/20/40 步的历史最佳增益、达到最佳值的步数、从历史最佳回落的幅度、`ok` 扰动端点比例、硬门槛后更新 RMS 的放大倍数。起点至少覆盖低、中、高三种初始 score；另从第 93 步这类已知近极值快照重置动量做一组 20 步局部实验。这样才能区分“早期爬升快”与“极值附近不震荡”两个目标。若没有该对照，调整 beta 只是经验猜测。

### 8.4 为什么不做完整 Newton

潜变量有 300 维，完整对称 Hessian 有

$$
\frac{300\times301}{2}=45150
$$

个独立元素。仅坐标中心差分的一次完整梯度就需要 600 次 score，而当前一步只需 8 次；有限差分 Hessian、Newton--CG 的多次 Hessian 向量积或全空间 BFGS 所需的可靠梯度都不符合当前代价。更重要的是，score 包含 `no_axis`、`drift_rejected` 等硬门槛，跨门槛处并不存在可供 Newton 使用的平滑二阶模型。

### 8.5 可行的局部超线性方案

最实际的方案是“全空间 Adam 找盆地，固定低维子空间后做带信赖域的 BFGS”，并且局部阶段必须从已保存的历史最佳点重新开始，而不是从已经回落的当前点开始。

1. 在历史最佳点构造 $r=4$ 的固定正交子空间，优先包含最近的有效 Adam 位移和稳定差分方向，其余方向随机补齐。
2. 在该子空间内用 $2r=8$ 个中心差分端点求投影梯度，成本与当前每步相同。固定方向后，连续几步的梯度差才满足 BFGS 割线关系；当前每步更换随机四维子空间的梯度不适合直接积累 BFGS Hessian。
3. 使用阻尼 BFGS 或小型信赖域求步长；额外的线搜索候选可以并成一个 GPU 批次。只接受实际 score 提高的候选，失败则缩短信赖域。
4. 若子空间内停滞，再更换子空间；不构造 300 维 Hessian。

在 $r=4$ 时，显式局部二次模型只有 $1+r+r(r+1)/2=15$ 个系数，也可用约两个八点批次做线性最小二乘拟合，再解一个小型信赖域子问题。它比直接对 SPSA 梯度套全空间 L-BFGS 更可靠，因为后者的相邻梯度来自不同随机子空间，曲率对容易失真。

切换局部阶段前还需要一个可计算的平滑性门槛：历史最佳在最近若干 Adam 步中改善趋缓；所有 $x\pm h u_j$ 均为 `ok`；用 $h$ 与 $h/2$ 得到的方向导数符号和尺度基本一致；局部曲率在重采样时稳定。只有这些条件成立，子空间 BFGS 才可能表现出超线性收敛。若探针仍跨硬门槛，就应继续使用一阶方法或缩小 $h$ 和信赖域，不能声称已经进入 Newton 区域。

当前最有证据的改进顺序是：先做受控 beta 对照；再增加“从历史最佳点启动”的平滑性诊断；最后只在通过诊断的局部子空间切换 BFGS/二次信赖域。完整 Hessian 不应进入近期实现计划。
