# QH Flow Matching 生成与拒绝迭代实验计划

日期：2026-07-28  
状态：仅完成方案设计；尚未迁移 QUASR 数据，尚未实现或提交训练作业  
实验分支：`qh-flow-matching-rl`  
基线提交：`9051f1f`

## 1. 目标与结论先行

本实验只关注 QH。目标是把 flow matching 模型训练成一个面向高分 QH 线圈的可迭代
采样分布：

1. 先用 QUASR 中**全部 QH 样本**训练基础生成模型；
2. 模型只接收 $N_{\mathrm{FP}}$ 和形状为 $n_c\times100$ 的高斯噪声，其中 $n_c$ 是
   基线圈数；不输入 QS、$\iota$、score 或其他物理条件；
3. 每轮从模型生成一批样本，用真实 C++/CUDA 黑箱评分器评分；
4. 拒绝低分样本，只用通过物理门槛且排名靠前的样本继续训练 flow；
5. 重复上述过程，观察生成分布是否超过原始 QUASR QH 分布。

这里的“RL”不是 PPO、策略梯度或可微代理模型。第一版准确地说是**基于真实奖励的拒绝
筛选与迭代最大似然训练**：每轮把生成模型在参数空间中的概率质量搬向 elite 集，行为上
相当于一个由神经网络表示、能表达线圈间强相关性的“大号 CEM”。

该路线可行，但有一个必须先解决的前置问题：当前 score 已经能惩罚低 $\iota$ 退化，
却还没有把“目标 QH 优于 QA/QP”直接写进总分。上一轮 CEM 的最高分样本实际更接近 QP。
生成模型比 CEM 表达力更强，若不先补 QH 选择性门控，它只会更有效地搜索 score 漏洞。

## 2. 本轮范围与非目标

### 2.1 本轮要完成

- 从旧服务器的完整 QUASR 中提取所有 `helicity=1` 样本的线圈数据；
- 只迁移线圈 Fourier 系数、电流、$N_{\mathrm{FP}}$、基线圈数、样本 ID 和必要元数据，
  不复制 80 GB 级的完整曲面与其他对象；
- 建立直接处理 100 维线圈 token 的 flow matching 模型；
- 在 4 张 RTX 5090 上用 DDP/BF16 训练；
- 建立稀疏但可信的生成样本 score 监控；
- 验证第一代模型是否已经能生成一定比例的好 QH；
- 第一代通过后，再实现真实评分、拒绝筛选、elite replay 和下一代微调闭环。

### 2.2 第一版明确不做

- 不使用旧 `../gan/task.md` 中的 VAE、proxy score 网络或代理梯度；
- 不先压缩到 PCA/VAE latent；flow 直接生成标准化后的 100 维线圈 token；
- 不做因果语言模型，不使用 RoPE、绝对位置编码或线圈序号 embedding；
- 不做 PPO、DPO 或其他需要奖励模型的算法；
- 不把 DESC 放进在线奖励；DESC 只用于最终极少量顶尖样本的离线复核；
- 不以覆盖全部 QH 多样性为目标，但仍保留最低限度的数值稳定性和去重检查。

## 3. QUASR QH 数据迁移

### 3.1 为什么在旧服务器先提取

完整 QUASR 的大部分体积来自曲面、Simsopt 序列化对象和其他诊断。本实验训练只需要每个
基线圈的 33 个 $x$ 系数、33 个 $y$ 系数、33 个 $z$ 系数和 1 个电流，即

$$
3\times33+1=100
$$

个数。即使有一百万个五基线圈 QH 样本，纯 FP32 token 也只有约 2 GB。因此应在数据所在
的旧服务器完成筛选和解包，再迁移紧凑结果，而不是复制完整 QUASR。

### 3.2 选择规则

读取完整 metadata，选择所有满足

$$
\mathrm{helicity}=1
$$

的记录。不按 metadata `qs_error`、$\iota$、磁面大小或线圈数做进一步筛选，避免基础模型
只看到人为挑选的高分尾部。缺文件、反序列化失败、曲线阶数不是 16 或 token 维度不是
100 的样本不静默丢弃，而是写入失败清单。

### 3.3 新数据格式

新服务器上的数据独立放在

`~/local_surface_evaluator_data/quasr_qh_flow_v1/`，不放进 Git 仓库。按
$(N_{\mathrm{FP}},n_c)$ 分组，输出分片式 `safetensors`：

- `tokens`: $[N,n_c,100]$，FP32；
- `id`: $[N]$，INT64；
- `nfp`: $[N]$，INT16；
- `n_base_coils`: $[N]$，INT8；
- 小型 metadata 使用 JSONL/Parquet 单独保存；
- 每个分片建议 4096--16384 个样本；
- manifest 记录样本数、形状、源 metadata 哈希、提取代码提交和每个分片的 SHA256。

训练、验证、测试划分按样本 ID 的稳定哈希完成，并在每个
$(N_{\mathrm{FP}},n_c)$ 组内保持一致比例。建议 90%/5%/5%，且任何样本只属于一个划分。

### 3.4 迁移步骤与验收

1. 在旧服务器只读统计完整 metadata 的 QH 数量、$N_{\mathrm{FP}}$ 分布、线圈数分布、
   缺失文件和曲线阶数；
2. 先提取 100 个样本做格式 smoke，并用现有 `load_quasr_field_input` 逐值对照；
3. 提交旧服务器侧的全量提取任务，生成分片、manifest 和失败清单；
4. 优先尝试旧服务器到新服务器的直接增量传输；若网络或认证不允许，则以本地 WSL 为
   中继，不在 Windows 工作区长期保存完整副本；
5. 新服务器复算所有 SHA256，并逐组核对样本数；
6. 随机抽取至少 100 个样本，在新服务器重新构造评分器输入，检查系数、电流单位、
   $N_{\mathrm{FP}}$ 和基线圈数；
7. 数据验收通过前不开始模型训练。

通过条件是：成功样本数等于“QH metadata 数减去有明确错误记录的数量”，所有分片哈希
一致，随机复核逐值一致，且电流单位统一为 A。

## 4. 线圈集合的表示与规范固定

### 4.1 token 与变长集合

一个 token 就是一个基线圈：

$$
\boldsymbol c_j=
[\boldsymbol x_j,\boldsymbol y_j,\boldsymbol z_j,I_j]\in\mathbb R^{100}.
$$

模型处理 $n_c$ 个 token 的集合。训练批次优先按 $n_c$ 分桶，避免 padding；必要时使用
attention mask。线圈数不作为额外条件输入，因为噪声张量的第一维已经给出 $n_c$。

线圈顺序没有物理含义。模型不使用位置编码，并在每个训练 step 随机置换 token 顺序。
这样完整 self-attention 网络对线圈置换等变：输入线圈置换后，输出速度以相同方式置换。

### 4.2 电流规范

真空磁力线几何对所有电流同时缩放近似不变，但 score 的工程项会感知绝对电流。为关闭
这条无物理意义的优化方向，训练数据按 $(N_{\mathrm{FP}},n_c)$ 固定总绝对电流：

$$
I_j\leftarrow I_j
\frac{I_{\mathrm{ref}}(N_{\mathrm{FP}},n_c)}{\sum_k|I_k|},
$$

其中 $I_{\mathrm{ref}}$ 取对应训练组的中位数。全局电流符号用“绝对值最大电流为正”固定。
原始电流仍保存在数据元信息中，但不进入第一版生成目标。

### 4.3 特征尺度

不同 Fourier 阶数的量级可相差数个数量级，不能直接对原值做 MSE。只用训练集计算每个
100 维特征的均值和标准差：

$$
\widetilde c_k=\frac{c_k-\mu_k}{\max(\sigma_k,\sigma_{\min})}.
$$

统计量按所有有效 coil token 计算，不按样本重复加权。标准化后只对极端异常值做宽松的
$[-8,8]$ 防护，并记录裁剪比例；若正常训练数据有超过 $10^{-4}$ 的值被裁剪，应先诊断
数据而不是放宽阈值。normalizer、数据 manifest 和 split 必须一起冻结并带哈希。

## 5. Flow Matching 模型

### 5.1 默认规模

第一版采用约 2800 万参数的非因果 Llama 风格集合 Transformer：

| 项目 | 默认值 |
| --- | ---: |
| token 输入/输出 | 100 / 100 |
| `d_model` | 512 |
| block 数 | 8 |
| attention heads | 8 |
| head dimension | 64 |
| SwiGLU hidden | 1408 |
| attention | 全连接、非因果 |
| 位置编码 | 无 |
| norm | RMSNorm，PreNorm |
| dropout | 0 |

每层 attention 约 105 万参数，SwiGLU 约 216 万参数；8 层主体约 2570 万参数，加上条件
投影、输入输出层后约 2800 万。序列长度通常只有 2--5，注意力计算不是瓶颈；继续放大到
上亿参数不太可能改善集合建模，反而会在有限 QH 数据上增加过拟合。

全量数据统计后允许按以下规则调整：QH 少于 1 万时降到 384 宽、8 层，约 1400 万参数；
QH 超过 5 万时保留默认 2800 万。第一轮不做大规模模型扫参。

### 5.2 时间与 $N_{\mathrm{FP}}$ 注入

flow 时间 $t\in[0,1]$ 使用固定 Fourier/sinusoidal embedding，经两层 MLP 投影到 512 维。
$N_{\mathrm{FP}}$ 使用离散 learned embedding。两者相加形成条件向量：

$$
\boldsymbol h_{\mathrm{cond}}
=\operatorname{MLP}(\operatorname{TimeEmbed}(t))
+\operatorname{Embed}_{\mathrm{nfp}}(N_{\mathrm{FP}}).
$$

每个 block 都用独立线性投影把该向量加到全部 token 的 residual stream。这里不加入线圈
数、score、QS 或其他条件。$N_{\mathrm{FP}}$ 的 embedding 表只覆盖数据中实际出现的整数，
越界输入直接报错，不做未经训练的外推。

### 5.3 block 结构

每层严格采用 PreNorm：

$$
\boldsymbol x\leftarrow\boldsymbol x+
\operatorname{MHA}(\operatorname{RMSNorm}(\boldsymbol x)+\boldsymbol h_{\mathrm{cond}}),
$$

$$
\boldsymbol x\leftarrow\boldsymbol x+
\operatorname{SwiGLU}(\operatorname{RMSNorm}(\boldsymbol x)+\boldsymbol h_{\mathrm{cond}}).
$$

MHA 使用 PyTorch SDPA/FlashAttention 可用后端，`is_causal=False`，不构造 RoPE 或位置
矩阵。输出端使用 final RMSNorm 和 $512\rightarrow100$ 线性层预测每个 token 的速度。

## 6. 基础 Flow Matching 训练

### 6.1 目标函数

对标准化数据 $\boldsymbol x_1$ 和同形状独立高斯噪声
$\boldsymbol x_0\sim\mathcal N(0,I)$，第一版使用 rectified flow 的线性路径：

$$
\boldsymbol x_t=(1-t)\boldsymbol x_0+t\boldsymbol x_1,
\qquad t\sim\mathcal U(0,1),
$$

目标速度为

$$
\boldsymbol v^*=\boldsymbol x_1-\boldsymbol x_0.
$$

训练损失是所有有效 token 和 100 个特征上的平均 MSE：

$$
\mathcal L_{\mathrm{FM}}
=\mathbb E\left[
\left\|\boldsymbol v_\theta(\boldsymbol x_t,t,N_{\mathrm{FP}})
-\boldsymbol v^*\right\|_2^2
\right].
$$

第一版不使用 minibatch OT、扩散加权或非线性插值。只有线性 flow 明确出现轨迹交叉、
采样质量差且实现正确性已验证后，才考虑 OT flow matching。

### 6.2 优化配置

- 4 卡 DDP，不使用 FSDP；模型太小，FSDP 通信没有收益；
- BF16 autocast，参数和 AdamW 状态保持 FP32；
- AdamW，初始学习率 $2\times10^{-4}$，$\beta=(0.9,0.95)$，weight decay 0.01；
- 1000 step warmup，之后 cosine decay；
- gradient norm clip 1.0；
- EMA 参数，初始 decay 0.999，稳定后 0.9999；
- 每卡 batch 从 512 起测，目标全局 batch 4096--16384；
- 数据量较小时按 step 训练，不以 epoch 作为停止条件；
- checkpoint 包含模型、EMA、optimizer、scheduler、normalizer、数据哈希、RNG 和 Git 提交。

### 6.3 ODE 采样

从 $t=0$ 的高斯噪声积分到 $t=1$：

$$
\frac{d\boldsymbol x}{dt}
=\boldsymbol v_\theta(\boldsymbol x,t,N_{\mathrm{FP}}).
$$

训练监控默认使用 16 步 Heun；正式第一代验收使用 32 步 Heun。抽取固定样本比较
16/32/64 步的 token RMSE、几何统计和 score 排名；若 32 与 64 步的关键指标差异小于
1%，固定 32 步。ODE 中不反复裁剪参数，只在最终反标准化后应用电流规范和安全边界。

## 7. 四卡训练与评分监控

### 7.1 Slurm 运行方式

所有训练、全量生成和评分都通过 Slurm 提交，不在登录节点运行重任务。主训练申请 4 张
RTX 5090、16 CPU，使用 `torchrun --nproc-per-node=4`。每个 rank 固定一张卡，DDP 使用
NCCL。优先启用 BF16、fused AdamW、SDPA 和 `torch.compile`，但每项优化都先与 eager
模式做数值对照。

由于序列很短，训练阶段的瓶颈可能是 DDP 通信和 Python/data loader，而不是 GPU 算力。
正式配置以实测 samples/s 为准；如果 4 卡吞吐不超过单卡的 2.5 倍，应先增大每卡 batch
和减少同步频率，而不是继续增加模型参数。

### 7.2 廉价几何监控

训练前期的随机输出可能是高频震荡、塌缩或相交线圈，不值得调用完整 score。每隔
200--500 step 用 EMA 模型生成固定条件样本，并在 GPU 上矢量化采样 Fourier 曲线，检查：

- 非有限值和标准化值越界；
- 线圈长度、包围盒、主半径；
- P95/max 曲率和高阶 Fourier 能量；
- 线圈间最小距离；
- 电流 L1 规范误差；
- 生成样本的重复率和到训练集最近邻距离。

阈值优先取 QH 训练集各指标的 0.5%--99.5% 分位，并额外保留绝对安全界限。该筛选只决定
是否值得调用昂贵 score，不参与基础 flow 的训练损失。

### 7.3 稀疏真实 score 监控

只有廉价几何通过率离开初始近零区间后，才开始真实评分。建议：

- 每 2000 step 或每 20 分钟触发一次，取先到者；
- 每次固定抽取 128--256 个生成样本，按 $(N_{\mathrm{FP}},n_c)$ 分层；
- 四个 rank 各在本卡调用同一 C++/CUDA 原生评分器；
- 模型显存保留，评分器在同进程或独立持久 worker 中使用剩余显存；
- 若并存导致吞吐下降或内存碎片，则在 checkpoint 边界暂停训练，释放模型后集中评分；
- 所有评分输入和完整输出写入只增不改的缓存，缓存 key 包含 token 字节、score ABI、
  配置哈希、动态库哈希和目标 helicity。

必须同时报告：几何 eligible 比例、全部生成样本的保守平均奖励、eligible 条件下的平均/
中位/P90 score、`ok` 比例、好 QH 比例、$\iota$、磁面大小、QH 残差和目标 helicity
margin。未通过几何门槛的样本在“全部样本平均奖励”中按零处理，不能只报告筛选后的均值。

## 8. QH 奖励在拒绝迭代前必须补齐

当前原生路径完成 $\psi$、$\alpha$ 和体点计算后，再评估不同 helicity 的边际成本很小。
在启动拒绝迭代前，应让一次 score 同时输出同一体积上的

$$
e_{\mathrm{QA}},\qquad e_{\mathrm{QH}},\qquad e_{\mathrm{QP}}.
$$

定义 QH 相对竞争模式的 margin：

$$
\Delta_{\mathrm{QH}}
=\log\frac{\min(e_{\mathrm{QA}},e_{\mathrm{QP}})+\epsilon}
{e_{\mathrm{QH}}+\epsilon}.
$$

只有 $\Delta_{\mathrm{QH}}>0$ 才表示目标 QH 优于 QA/QP。最终用于拒绝排序的奖励应包含：

1. 当前综合 score；
2. $|\iota|<1$ 的已有惩罚；
3. 尺寸在 $a/R\ge0.03$ 后饱和的已有规则；
4. 经 QUASR 标定的 QH margin 软门控；
5. `status=ok`、体点有效率和最低磁面实用性硬门槛。

margin 的数值尺度必须在完整 QH 数据上标定，不能直接凭一个样本设常数。基础 flow 可以
在 margin 完成前训练，但拒绝迭代不能在未冻结奖励版本时启动。

## 9. 第一代生成模型验收

### 9.1 固定测试

基础训练完成后，从 EMA 模型生成固定 8192 个样本。每个
$(N_{\mathrm{FP}},n_c)$ 组至少分配 256 个，其余按训练集组频率分配。使用 32 步 Heun，
固定随机种子，并对全部几何 eligible 样本运行真实 score。

“好 QH”不只看一个总分。正式阈值在完整 QUASR QH 重评分后冻结，初步定义为：

$$
\begin{aligned}
&\mathrm{status}=\mathrm{ok},\\
&S\ge \operatorname{median}(S_{\mathrm{QH,heldout}}),\\
&\iota_*\ge1,\\
&\Delta_{\mathrm{QH}}>0,\\
&a/R\ge0.02,\\
&f_{\mathrm{valid}}\ge0.95.
\end{aligned}
$$

若完整数据重评分的中位数与当前 66.8 的参考差别很大，以新 held-out 标定为准。

### 9.2 第一阶段通过条件

- 8192 个样本中，几何 eligible 比例至少 70%；
- 原生 score 的 `ok` 比例至少达到 held-out QH 的 70%；
- “好 QH”比例至少 1%，且 95% Wilson 下界高于 0.5%；
- 至少得到 50 个非重复好 QH 样本；
- 好样本不能只来自单一 $(N_{\mathrm{FP}},n_c)$ 组；
- 32/64 步采样对好样本判定的一致率至少 95%；
- 随机 token 置换后，模型输出的等变误差低于 $10^{-5}$（FP32 测试）；
- 最近邻审计能区分精确训练集记忆与新样本，并在报告中分别统计。

这一定义把“一定比例”具体化为可统计验收的 1% 下限。若好样本比例不足，但生成分布能
复现 held-out QH 的 score 和物理分量，应先检查阈值；若连 held-out 分布都复现不了，
则基础 flow 失败，不进入拒绝迭代。

## 10. 拒绝筛选与迭代训练

### 10.1 每一代的流程

对第 $g$ 代模型：

1. 按固定的 $(N_{\mathrm{FP}},n_c)$ 配额生成 $P_g$ 个样本；
2. 运行廉价几何门槛，未通过者奖励为零；
3. 对通过者使用冻结版本的原生 QH reward 评分；
4. 在每个 $(N_{\mathrm{FP}},n_c)$ 组内按 reward 排名，取前 5%--10%；
5. 再应用 QH margin、$\iota$、尺寸和有效率硬门槛；
6. 精确去重，并把全部已评分样本及完整诊断加入永久 replay；
7. 用接受样本、历史 elite 和原始 QUASR QH 混合训练下一代 flow；
8. 在冻结测试噪声与 held-out 条件上评估，确认不是仅训练 loss 下降。

必须在组内选 elite。若全局排名，模型会把概率全部迁移到最容易得分的
$(N_{\mathrm{FP}},n_c)$，使其他输入形状失去意义。用户不要求组内广泛多样性，但共享模型
仍应能响应它声称支持的各个 $N_{\mathrm{FP}}$ 和线圈数。

### 10.2 下一代训练集

建议初始正样本混合比例为：

- 60% 本代与历史 elite，按 reward 分位加权采样；
- 40% 原始 QUASR QH。

flow matching 对接受样本使用**重新采样的新高斯噪声**，不需要保留生成这些样本时的原始
ODE 噪声。随着 elite 池稳定，可把 QUASR 比例降到 10%--20%，但不建议降为零：保留少量
真实数据能防止模型数值塌缩到评分器的极窄漏洞。

上一代 replay 中接近门槛的 hard negatives 单独保留，不进入正样本 FM loss；它们用于
统计、阈值审计和发现 reward hacking。若后续需要显式对比学习，应另立实验，不能偷偷
改变 flow 目标。

### 10.3 建议预算

当前 3 基线圈原生 score 的四卡实测均摊约为
$0.85\,\mathrm{s/candidate}$。据此：

| 每代候选数 | 预计四卡评分墙钟 |
| ---: | ---: |
| 2048 | 约 29 分钟 |
| 8192 | 约 1.93 小时 |
| 12288 | 约 2.90 小时 |
| 16384 | 约 3.87 小时 |

模型采样和再训练预计显著短于评分，但实际值必须由 4 卡 pilot 测量。第一轮拒绝实验建议
$P_0=2048$ 做全链路 smoke；稳定后使用 $P_g=8192$。不一开始使用数万候选，避免在奖励
漏洞或缓存错误上浪费数小时。

## 11. 分阶段执行顺序

### 阶段 A：数据迁移与审计

- 完成第 3 节的全量提取、传输、哈希和随机逐值复核；
- 输出 QH 样本总数、各 $N_{\mathrm{FP}}$、各线圈数和组合交叉表；
- 输出各 Fourier 特征、电流、长度、曲率和 metadata QS 的分布；
- 冻结 `dataset_manifest_v1.json`。

### 阶段 B：单元测试与小数据过拟合

- normalizer 正反变换误差；
- 电流规范和全局符号规范；
- 线圈置换等变测试；
- attention mask 与按线圈数分桶结果一致；
- 单卡 FP32/BF16 一致性；
- 64 个样本上过拟合，确认 FM loss 和重采样统计显著改善；
- 16/32/64 步 ODE 数值收敛。

### 阶段 C：四卡基础 flow pilot

- 4 卡 DDP 启动、checkpoint 恢复和确定性 smoke；
- 测每卡 batch 256/512/1024，选择吞吐最优点；
- 比较 eager、SDPA、compile 和 fused optimizer；
- 跑 2000--5000 step，验证廉价几何通过率开始上升；
- 此阶段只做最多 128 个真实 score 的小监控。

### 阶段 D：基础模型正式训练与第一代验收

- 按 step 训练到验证 FM loss、几何通过率和生成 score 同时平台；
- 保留 raw 与 EMA checkpoint，正式采样只用 EMA；
- 按第 9 节生成 8192 个样本并完成第一代验收；
- 对高分样本运行 QH/QA/QP 选择性审计；
- 至少抽取若干顶尖样本走旧 Boozer/Poincare 路径，防止仅信任在线 score。

### 阶段 E：第一轮拒绝迭代

- 先冻结含 QH margin 的 reward ABI 和配置；
- 生成 2048 个样本跑通 generation、prefilter、score、cache、elite、replay、再训练；
- 全链路无误后扩到每代 8192；
- 每代都保留模型、采样 manifest、全部 score、elite 和下代训练数据哈希；
- 与“不更新模型、只继续从第一代 flow 采样”的等预算对照比较。

### 阶段 F：是否超过 QUASR

只有第一轮拒绝迭代确实提升 held-out 固定指标后才进行。最终比较：

- 生成 score 分布与完整 QUASR QH 分布；
- 达到 QUASR P50/P90/P99 的比例；
- QH margin、$\iota$、磁面尺寸和工程分量；
- 等预算基础 flow 随机采样对照；
- 顶尖样本的旧 Boozer/Poincare 与必要的 DESC 离线复核。

## 12. 正确性与防 reward hacking

开发阶段至少需要以下交叉验证：

1. 同一 QUASR token 在旧序列化加载器、紧凑数据集和原生 score 输入中逐值一致；
2. 同一生成样本重复 score 的差异不超过既有确定性阈值；
3. QH、QA、QP 残差在同一体点和同一 $\alpha$ 上计算，避免比较不同采样噪声；
4. 训练监控样本与正式验收样本的随机种子完全分离；
5. score cache 不跨 ABI、动态库、配置或 normalizer 版本复用；
6. 顶尖样本检查电流尺度、线圈碰撞、高频振荡、极小磁面和低 $\iota$；
7. 每代报告全体候选而不只报告 elite，防止筛选后均值虚高；
8. 至少保留一个完全不更新模型的等预算采样对照；
9. 最终高分样本必须通过目标 helicity 热图和旧稳定物理路径，不能只看 native score。

## 13. 主要风险与应对

### 13.1 QH 样本数量不足

若完整 QH 少于 1 万，2800 万参数可能过大。应降到约 1400 万参数，加强 token 置换和
噪声重采样，不靠 dropout 掩盖过拟合。若基础 flow 只记忆训练样本，先缩模型，而不是
直接进入拒绝迭代。

### 13.2 集合 token 的对应关系增加 flow 难度

线圈无序使同一集合有 $n_c!$ 种排列。无位置编码和随机置换保证分布定义正确，但线性
interpolant 仍可能产生不必要的 token 配对噪声。若基础模型失败，可比较确定性几何排序、
Hungarian token matching 或 minibatch OT；第一版不预先增加这些复杂性。

### 13.3 4 卡对短序列加速不明显

序列最多几个 token，单 step 计算密度低。应优先增大全局 batch、使用 fused kernel 和
减少同步，而不是换 FSDP。若 4 卡加速仍低于 2.5 倍，报告真实结果，但正式训练仍按用户
要求使用 4 卡；评分阶段可继续充分利用四卡。

### 13.4 拒绝迭代塌缩

用户不要求广泛多样性，但完全塌缩会导致重复评估、梯度退化和单点 reward hacking。
因此仍保留精确去重、训练集最近邻、每组最低 elite 数和少量 QUASR replay。这些约束是
为了优化稳定，不是为了训练通用生成模型。

### 13.5 score 再次被利用

这是最大风险。低 $\iota$、磁面尺寸和 QH/QP 混淆已经各出现过一次。拒绝迭代前冻结
QH margin，在线报告竞争 helicity，并对每代顶尖样本运行独立旧路径。若新漏洞出现，应
停止该代，不把其 elite 回灌模型。

## 14. 计划中的代码与产物布局

建议新增：

- `flow_matching/data.py`：分片、normalizer、分桶和置换；
- `flow_matching/model.py`：非因果 Llama 集合 flow；
- `flow_matching/flow.py`：FM loss 和 ODE sampler；
- `flow_matching/train.py`：DDP/BF16/EMA/checkpoint；
- `flow_matching/replay.py`：不可变 score cache 和 elite replay；
- `scripts/export_quasr_qh_flow.py`：旧服务器全量紧凑提取；
- `scripts/train_qh_flow.py`：基础训练；
- `scripts/sample_score_qh_flow.py`：生成、筛选和四卡评分；
- `scripts/run_qh_rejection_round.py`：单轮拒绝迭代；
- `scripts/slurm_qh_flow_*.sh`：训练、评分和迭代 Slurm 入口；
- `tests/test_flow_*.py`：规范、等变、FM、采样和缓存测试；
- `reports/qh_flow_matching_experiment_report.md`：后续实验报告。

所有重型输出写入 `runs/qh_flow_matching/`，数据写入独立的
`~/local_surface_evaluator_data/quasr_qh_flow_v1/`。Git 只保存代码、配置、轻量 manifest、
压缩统计和最终报告，不提交全量数据、checkpoint 或 score replay。

## 15. 开始执行后的第一个停点

用户批准本计划后，第一步只做**数据迁移阶段 A**。交付以下信息后再开始模型实现：

1. 完整 QH 样本数；
2. $(N_{\mathrm{FP}},n_c)$ 交叉分布；
3. 成功/失败提取数量及失败原因；
4. 紧凑数据总大小、分片数和 SHA256 验收；
5. 100 个随机样本的新旧逐值复核；
6. 根据实际样本数最终确认使用约 1400 万还是 2800 万参数模型。

在这个停点之前不提交正式训练作业，也不开始拒绝迭代。
