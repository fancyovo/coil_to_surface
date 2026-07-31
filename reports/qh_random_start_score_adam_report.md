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

尚未运行。代码和起点面板已经准备完成，但 Slurm 控制器在提交烟测前持续返回 socket timeout；待调度服务恢复后先运行单起点、单步烟测，再提交 12 起点、每条 40 步的正式数组。

## 5. 结论与边界

当前已经确定的是自然随机起点的 score 覆盖，而不是 Adam 的因果结论。4096 样本足以说明 $S\geq40$ 是约 $2.4\times10^{-4}$ 的稀有事件，且本轮没有观测到 $S\geq50$；起始 score 对 Adam 增益、最终分数和端点有效率的影响仍需等待第 4.4 节实验完成。
