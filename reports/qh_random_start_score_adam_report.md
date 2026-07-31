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

本节在正式随机池和 Adam 作业验收后填写。

## 5. 结论与边界

本节在正式随机池和 Adam 作业验收后填写。
