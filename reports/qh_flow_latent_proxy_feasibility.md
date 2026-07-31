# QH flow 潜空间代理可行性报告

> 日期：2026-07-31  
> 状态：仅完成理论分析与实验设计；尚未编写代码、反演数据或提交训练作业。

## 1. 结论摘要

这个实验在工程上高度可行：现有代码已经支持批量 FP32 RK4 反向积分，QUASR QH 数据只有
170,755 个样本，反演结果可以一次生成并常驻 RTX 5090 显存；代理网络也可以沿用当前 flow
的非因果 Transformer 结构，但应缩小到约 400 万参数，训练成本远低于真实物理 score。

方案的理论边界需要先说清楚：

> 如果 flow 已经精确地把标准高斯先验映射成 QUASR QH 数据分布，那么 QUASR 样本反演所得
> 潜变量本身就应服从同一个标准高斯分布。此时“反演 QUASR”与“随机高斯”在理论上不可区分，
> 最优分类器只能恒定输出 $0.5$。

但这并不否定当前实验，反而准确描述了它的目标。当前 flow 明显没有达到上述理想状态：QUASR
样本反演后再正向积分，能够以很高精度还原原样本，因而绝大部分仍处在 QUASR 的高分区域；直接
从标准高斯随机采样后解码的样本则几乎都处在低分区域。两条路径已经在数据空间表现出显著分布差异。

由于当前 ODE 映射已经验证为近似可逆，这个差异必然在潜空间留下对应的可分信息。拟议代理就是用
一个远比 flow 解码加物理 score 廉价的网络，蒸馏这种差异。真正未知的不是“差异是否存在”，而是
约 400 万参数的集合 Transformer 能捕捉多少差异、能把随机先验中的有效区域富集多少。

这个代理严格来说是**当前有限精度 flow 下，反演 QUASR 潜变量相对于标准高斯先验的密度比代理**。
在当前标签定义下，它可以作为操作意义上的“好/坏概率”，但尚不是连续物理 score 的代理，也不能
区分 QUASR 正类内部的优劣。

综合判断如下：

| 问题 | 判断 |
| --- | --- |
| 能否批量反演并训练 | 高度可行 |
| 是否需要真实 score | 本阶段不需要 |
| 当前是否存在正负类分布差异 | 存在；已有正向解码质量差异是直接证据 |
| 有限代理能捕捉多少 | 需要由 held-out 分类与富集率实验确定 |
| 若能稳定分开，能否快速筛先验起点 | 可以，且不需要先解码或运行物理 score |
| 输出能否称为物理“好样本概率” | 不能；只能称为 QUASR 潜空间支持度 |
| 是否最终能提高真实优化成功率 | 本阶段无法证明，后续必须单独验证 |

因此建议执行这个实验，并将其定义为一次严格受控的**潜空间二样本分类兼支持度代理实验**。

## 2. 理论上到底在分类什么

记条件为

$$
c=(N_{\mathrm{FP}},n_c),
$$

当前 flow 的确定性 ODE 映射为

$$
F_c:\boldsymbol z\mapsto\boldsymbol x,
\qquad
\boldsymbol z\sim p_0=\mathcal N(0,I),
$$

其中 $\boldsymbol x$ 是标准化、规范化后的线圈 token。拟议正类和负类分别是

$$
\boldsymbol z_+=F_c^{-1}(\boldsymbol x),
\qquad
\boldsymbol x\sim p_{\mathrm{QH}}(\boldsymbol x\mid c),
$$

以及

$$
\boldsymbol z_-\sim p_0(\boldsymbol z\mid c).
$$

若训练时正负类各占一半，Bayes 最优分类器为

$$
D^*(\boldsymbol z,c)
=
\frac{p_{\mathrm{inv}}(\boldsymbol z\mid c)}
{p_{\mathrm{inv}}(\boldsymbol z\mid c)+p_0(\boldsymbol z\mid c)},
$$

其中

$$
p_{\mathrm{inv}}
=
(F_c^{-1})_\#p_{\mathrm{QH}}.
$$

若 flow 完美满足

$$
(F_c)_\#p_0=p_{\mathrm{QH}},
$$

则必然有

$$
p_{\mathrm{inv}}=p_0,
\qquad
D^*(\boldsymbol z,c)=\frac12.
$$

这意味着训练得越完美的生成 flow，越不应允许这个代理区分两类；而当前实验正是要学习有限训练
后的剩余失配。当前代理可学习的信号主要来自以下几项：

1. flow 对 QUASR 分布覆盖不足或概率质量分配不准；
2. 当前网络容量、训练误差和有限 ODE 积分造成的映射偏差；
3. QUASR 有限数据集自身形成的经验支持区域；
4. 数据规范化、条件分组或数值流程中的非物理捷径。

这里已经有明确的可利用信号。记随机先验经过当前 flow 后的生成分布为

$$
p_{\mathrm{gen}}=(F_c)_\#p_0.
$$

对于可逆映射，分布的总变差距离和 KL 散度等可分性度量在双射变换下保持不变，例如

$$
D_{\mathrm{KL}}(p_{\mathrm{inv}}\Vert p_0)
=
D_{\mathrm{KL}}(p_{\mathrm{QH}}\Vert p_{\mathrm{gen}}).
$$

因此，只要数据空间里的 $p_{\mathrm{QH}}$ 与 $p_{\mathrm{gen}}$ 不同，潜空间里的
$p_{\mathrm{inv}}$ 与 $p_0$ 也不同。现有 30k 物理加权模型生成的 2,048 个样本中，最高 score
为 23.32，而 QUASR QH 的 P10 约为 41.31；再结合高精度反演闭环，这已经证明当前两类潜变量
分布并未重合。代理实验要回答的是这个判别边界对有限网络而言有多复杂，而不是重新证明差异存在。

但还要注意：这里把**全部 QUASR QH** 都标成正类，没有区分其中高分和低分样本。因此即使分类
完全成功，输出也只表示“像 QUASR QH 反演潜变量”，不能区分 QUASR 内部的优劣，更不能替代
现有物理 score。

## 3. 推荐的完整实验流程

### 3.1 固定数据、checkpoint 和规范化

使用已经校验的数据：

- QH 总数：170,755；
- train/validation/test：153,747 / 8,500 / 8,508；
- 数据路径：`~/local_surface_evaluator_data/quasr_qh_flow_v1/`；
- 保持现有基于样本 ID 的稳定 split，不在代理训练时重新随机切分；
- 使用 30,000 step EMA checkpoint 及其中保存的 `CoilNormalizer`；
- checkpoint SHA-256：`39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f`。

现有规范化会按 $(N_{\mathrm{FP}},n_c)$ 固定总绝对电流、消除全局电流符号并逐特征标准化。
反演前必须复用这套逻辑，不能重新拟合统计量。

代理可以在全部条件组上联合训练，但每个正样本必须配一个**完全相同条件**的负样本：

$$
(N_{\mathrm{FP}},n_c)_-=(N_{\mathrm{FP}},n_c)_+.
$$

否则分类器只需识别不同的 $N_{\mathrm{FP}}$ 或线圈数比例，就能得到虚假的高准确率。当前优化最
关心 $N_{\mathrm{FP}}=4,n_c=3$，所以该组必须单独报告指标，不能只看全数据 micro average。

### 3.2 一次性批量反演

按 $(N_{\mathrm{FP}},n_c)$ 分组，在 4 张 GPU 上分片执行：

$$
\boldsymbol z_+=F_c^{-1}(\boldsymbol x),
$$

正式配置固定为：

- EMA 权重和 ODE 状态均为 FP32；
- RK4；
- 从 $t=1$ 积分到 $t=0$；
- 256 步；
- `torch.no_grad()`；
- 每个 rank 写独立 shard，最后只合并 manifest，不让多个进程竞争同一文件。

此前 3 个样本上的 256 步双向 RK4 已验证：标准化 token RMS 闭环误差为
$9.11\times10^{-8}$ 到 $1.15\times10^{-7}$，线圈位置 RMS 误差为
$2.26\times10^{-8}$ 到 $4.57\times10^{-8}\,\mathrm m$。这足以作为当前正式反演精度。

保存的每条正类记录至少包含：

- latent FP32 tensor；
- QUASR ID、split、$N_{\mathrm{FP}}$ 和 $n_c$；
- checkpoint SHA-256、normalizer 摘要；
- RK4 方法、步数和代码 commit；
- 分片 SHA-256。

在每个 split 和条件组抽固定子集做反向再正向闭环。若批量实现的误差明显高于既有结果，则不
进入代理训练。

### 3.3 负类生成

主负类直接按正类条件逐条生成：

$$
\boldsymbol z_-\sim\mathcal N(0,I).
$$

训练负类可在 GPU 上按 batch 即时生成；validation/test 负类应使用固定且互不重叠的随机种子，
以保证指标可复现。这样既不需要保存大量负类，也不会让训练只记住一份固定噪声。

还应增加一个小规模数值对照集：

$$
\widetilde{\boldsymbol z}_-
=F_c^{-1}(F_c(\boldsymbol z_-)).
$$

它与主负类物理上等价，却经历了和正类相同的 ODE 算术路径。如果代理能区分正类和直接高斯，
却不能区分正类和 round-trip 高斯，说明它主要利用了数值积分痕迹，而不是有意义的潜空间结构。
该对照只需在 validation/test 子集执行，不必把全部负类都多积分两次。

### 3.4 显存驻留

即使把全部样本都补齐到 5 根线圈，正类 latent 的 FP32 上界也只有

$$
170755\times5\times100\times4
\approx326\ \mathrm{MiB}.
$$

实际值还会更低。因而每个 rank 都可以把正类按条件分组常驻显存；负类用 `randn_like` 即时
生成。相对于 32 GiB RTX 5090，这部分内存不是瓶颈，也不需要 CPU DataLoader 参与热路径。

## 4. 代理模型

### 4.1 网络结构

沿用 flow 的集合 Transformer 设计，但不建议直接复制 30,333,540 参数的完整模型。当前每个
样本最多只有 5 个 token，目标又只是二分类，原模型容量很容易记住有限正样本。

第一版建议：

| 项目 | 配置 |
| --- | --- |
| token | 每根线圈一个 100 维潜变量 token |
| Transformer | 4 层，width 256，8 heads，SwiGLU hidden 704 |
| 规范化 | RMSNorm、PreNorm |
| attention | 非因果、无 RoPE、无位置编码 |
| 条件 | learned $N_{\mathrm{FP}}$ embedding，注入每个 block |
| 聚合 | 对有效线圈 token 做 masked mean pooling |
| 输出 | RMSNorm、线性层得到单个 logit |
| 参数量 | 约 380 万 |

mean pooling 与无位置编码保证输出对线圈 token 排列不变。代理不需要 flow time embedding；
线圈数由 token 数或 mask 自然给出。

### 4.2 损失与训练

训练使用平衡的 1:1 正负 batch 和

$$
\mathcal L_{\mathrm{cls}}
=\operatorname{BCEWithLogitsLoss}(\ell,y).
$$

sigmoid 只用于记录

$$
D(\boldsymbol z,c)=\sigma(\ell),
$$

不在 loss 前手动调用。建议配置为 4 卡 DDP、BF16 autocast、FP32 参数/优化器/损失归约、
fused AdamW。由于训练集可以常驻显存，batch 应从每卡 8,192 起测，并优先增大到吞吐平台。

训练按 held-out validation AUROC 和 log loss 早停，不按训练准确率决定是否收敛。完整模型训练
至少使用 3 个随机种子，以排除一次偶然初始化。

### 4.3 必须同时训练的简单基线

至少保留以下对照：

1. 只使用 $\|\boldsymbol z\|_2$ 的一维分类器；
2. 使用每个样本潜变量均值、方差和范数的 logistic regression；
3. 小型逐 token MLP 加 mean pooling。

如果 Transformer 并未明显超过这些基线，说明可分信号主要只是半径、均值或方差偏移，没有
证据表明它学到了多线圈和 Fourier 分量之间的结构关系。

## 5. 正确性检查与验收标准

### 5.1 防止伪高准确率

需要逐项排除以下捷径：

- train/validation/test 在反演前就固定，绝不把同一个 QUASR ID 放入多个 split；
- 各 split 使用独立负类随机种子；
- 正负类的 $(N_{\mathrm{FP}},n_c)$ 数量逐组严格一致；
- 模型无位置编码，并检查线圈排列不变性；
- 复用 flow 的电流尺度和全局符号规范，不能重新引入电流正负号捷径；
- 不把 ID、split、文件编号或来源元数据输入模型；
- 对 QUASR 近重复样本做最近邻审计；若现有 ID 哈希 split 存在明显近重复泄漏，再按几何簇重切
  一个额外审计 split；
- 比较直接高斯负类与 round-trip 高斯负类，排除 RK4 算术痕迹；
- 在小子集比较 256 与 512 步反演后的代理输出，确认结论不依赖积分步数。

### 5.2 核心指标

除 train/validation/test 的 BCE、accuracy、ROC-AUC、PR-AUC 外，还应报告：

- 每个 $(N_{\mathrm{FP}},n_c)$ 组的 ROC-AUC，特别是 $N_{\mathrm{FP}}=4,n_c=3$；
- macro average，避免大组掩盖小组失败；
- Brier score 和 ECE，描述 sigmoid 输出的校准程度；
- 正负类 latent 范数、逐维均值/标准差和协方差谱；
- Transformer 相对简单基线的增益；
- 3 个训练种子的均值和标准差。

为了直接对应“快速筛选起点”，还应把阈值设为随机先验的固定通过率。例如在只保留随机起点
前 10%、1% 和 0.1% 时，统计 held-out 正类的保留率，并定义

$$
\text{enrichment}(q)
=
\frac{\Pr(D(\boldsymbol z_+)>\tau_q)}{q},
$$

其中 $\tau_q$ 使得随机负类只有比例 $q$ 通过。这个量比普通 accuracy 更直接回答代理能否把
QUASR 支持区域富集到候选池前部。

### 5.3 如何解释结果

| 观察 | 结论 |
| --- | --- |
| train/test 都接近 AUC 0.5 | 与已有解码质量差异不一致；先查训练、条件配平和模型容量，不能直接宣布 flow 已完美高斯化 |
| train 很高、test 接近 0.5 | 模型记住了有限 QUASR latent，应缩小模型或改善 split |
| test 很高，但范数基线同样高 | 主要是简单径向分布失配；可筛选，但结构信息有限 |
| Transformer 在 held-out 和 round-trip 对照上都显著胜过基线 | 存在可泛化的结构性支持差异，可作为起点预筛代理 |
| 总体 AUC 高，但 $N_{\mathrm{FP}}=4,n_c=3$ 低 | 对当前 QH 优化目标无实际帮助 |

若以“代理可用于强预筛”为目标，一个有说服力但非物理验收的结果应至少包括：held-out
$N_{\mathrm{FP}}=4,n_c=3$ AUC 明显高于 0.5、不同种子稳定、Transformer 明显超过范数基线，
并在 1% 随机通过率下保留远高于 1% 的 held-out 正类。这里不应预先把某个 AUC 数字规定成
必达指标；重点是量化当前已知分布差异中有多少能被便宜代理恢复。

## 6. 概率输出的准确含义

在 1:1 人工采样的训练分布中，经 held-out calibration 后的 sigmoid 最多可以解释为

$$
\Pr(\text{样本来自反演 QUASR}\mid\boldsymbol z,c).
$$

它不是

$$
\Pr(\text{真实物理 score 高}\mid\boldsymbol z,c),
$$

也不是某个起点经 Adam/CEM 后成功的概率。报告和接口中建议命名为 `support_probability` 或
`quasr_likeness`，不要命名为 `good_probability`。

在平衡类先验下，其 odds 近似密度比：

$$
\frac{D}{1-D}
\approx
\frac{p_{\mathrm{inv}}}{p_0}.
$$

因此未来可以从大量标准高斯起点中只保留高 $D$ 候选，再对少量候选做 flow 解码和真实 score。
这正是该代理最合理的用途。它不适合作为新的物理目标被无限优化，否则同样可能产生对分类器
漏洞的对抗性样本。

## 7. 速度与资源判断

### 7.1 一次性反演

256 步 RK4 每个样本需要 1,024 次 velocity network evaluation，总工作量为

$$
170755\times1024
\approx1.75\times10^8
$$

个 sample-forward 等价调用。这是本实验最大的单次成本，但高度规则、无梯度、可以按样本在 4 卡
上完全并行。正式反演前只需用每卡 4,096、8,192 和 16,384 的 batch 做短吞吐测试，然后按

$$
T_{\mathrm{inverse}}
\approx
\left\lceil\frac{N}{4B}\right\rceil
1024\,T_{\mathrm{forward}}(B)
$$

外推即可。

现有 3 样本 landscape 的 32.65 秒包含多种闭环步数和大量方向生成，是极小 batch，不能拿来
线性估计全数据耗时。现有 flow 训练吞吐约 40 万样本/秒，说明大 batch 能充分利用 5090；合理
预期是批量反演处于分钟到几十分钟量级，但在做吞吐 pilot 前不应给出更精确承诺。

### 7.2 代理训练与使用

代理只有约 380 万参数，不需要 ODE 积分，也不调用 C++/CUDA 物理 score。训练数据常驻显存后，
其训练和验证预计比反演更轻。未来筛选时也只运行一次小 Transformer 前向，不需要先 flow 解码，
因此可以对很大的随机候选池批量评分。

本阶段完全不运行真实 score，因而不会出现当前原生 score 约秒级每样本的瓶颈。最终报告仍应记录：

- 每组反演 batch 吞吐和总墙钟；
- 代理训练 samples/s、step 数和收敛墙钟；
- 单卡及 4 卡代理推理 samples/s；
- GPU preflight/postflight 和峰值显存。

## 8. 推荐执行边界

建议下一步只做以下闭环：

1. 批量反演并验证数值闭环；
2. 训练简单基线和小型 Transformer；
3. 在完全 held-out 的 QUASR 正类与新随机负类上做二样本评估；
4. 输出每组指标、富集曲线、latent 统计图和数值对照；
5. 不调用真实 score，不提交物理完整评估。

若得到稳定的结构性区分能力，就把模型冻结为潜空间预筛器。下一阶段才需要另外回答“代理前
$q\%$ 的随机起点，真实 score 或后续优化成功率是否得到富集”。这一步必须使用真实物理标签，
但它不属于当前用户指定的无 score 实验。

最终判断是：**工程实现简单、GPU 利用率高，而且当前 flow 的数据空间失配已经保证潜空间存在
可分信息，值得直接实验。实验成败取决于便宜代理能捕捉多少这种失配，以及能否在很低的随机通过率
下保留足够多的反演 QUASR latent。**

