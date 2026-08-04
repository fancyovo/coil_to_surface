# QH 原生评分器局部物理梯度探索计划

**日期：** 2026-08-04  
**分支：** `qh-blackbox-gradient`  
**状态：** 仅完成方案设计，尚未修改评分器或提交远端实验

## 1. 目标与结论摘要

当前潜空间 Adam 每步用 4 个正交反向方向估计梯度，因此需要 8 次完整 ABI-9 score，再对更新后的中心做一次完整 score。四张 GPU 可以把 8 个端点分成两批并行，但磁轴搜索和长周期磁面筛选仍被重复执行。已有剖面显示，一次完整 score 约 5--8 秒，其中磁轴与磁面追踪占绝大部分；100000 点体场、$\alpha/\iota$ 线性最小二乘和 QS 归约只占很小一部分。

本次探索的目标不是把整个含离散分支的评分器改造成“全局精确可微”的程序，而是实现一个**固定当前可行分支的局部物理梯度**：

$$
g_x\approx \frac{\partial \widetilde S}{\partial x},
$$

其中 $x$ 是线圈 Fourier 系数和电流，$\widetilde S$ 是在当前磁轴、候选磁面和有效点分支附近定义的连续局部目标。随后通过 flow 解码器得到潜空间梯度

$$
g_z=J_F(z)^Tg_x.
$$

真实 ABI-9 score 始终保留为接受、回滚和拓扑判定标准。近似梯度只负责提出方向，不负责证明新样本仍有磁轴、磁面或正确 helicity。

探索顺序按“先建立可信接口，再尽快触及最重要的体 QH，最后才补昂贵几何依赖”组织：

1. 梯度 oracle、前向 cache 和分支指纹。
2. score 代数与线圈工程项梯度，作为低风险接口验收。
3. 固定几何前端的体 QH 梯度，这是第一项真正影响优化效率的核心工作。
4. $\alpha/\iota$ 线性最小二乘的隐式 VJP。
5. 仅在证据表明必要时，加入 $s/\psi$、射线根和采样点移动。
6. 磁轴隐式梯度和场线追踪伴随最后考虑；离散候选选择不做梯度穿透。

验证位型不使用“QUASR 样本反向追踪后直接得到的近局部最优点”作为主体。核心样本来自先前从随机 flow 潜变量出发的 Adam 完整轨迹，覆盖约 74--87 分的早期、中期、成熟期和可行性边界；另用不同 $N_{\rm FP}$ 的两线圈轨迹交叉验证。

## 2. 当前基线与预期收益

当前四方向反向差分为

$$
\widehat g_z=
\frac{1}{K}\sum_{k=1}^{K}
\frac{S(F(z+c u_k))-S(F(z-c u_k))}{2c}u_k,
\qquad K=4,quad c=0.005.
$$

其优点是完全不依赖评分器内部结构，缺点是每步重复 8 次完整前端，而且只得到四维随机子空间内的估计。已有 $N_{\rm FP}=4$、两线圈作业中，200 步 Adam 用时 3460.8 秒，平均每步约 17.3 秒；单步记录中也出现过约 27 秒的墙钟。

局部 VJP 路线希望把一次新迭代缩减为：

$$
\text{上一步已接受的前向 cache}
\rightarrow \text{一次局部 VJP}
\rightarrow \text{一次 flow VJP}
\rightarrow \text{一次候选真 score}.
$$

候选真 score 若通过，就同时生成下一步 cache；若失败，则回滚并缩短信赖步长。理想情况下每步只新增一次完整 score，而不是 9 次。即使 flow VJP 和局部物理 VJP 各花数秒，只要总墙钟和长尾明显低于现有基线，仍有实际价值。

## 3. 可微边界与分支指纹

ABI-9 QH 分数可写为

$$
S=S_0\,[0.1+0.9q_\iota]\,[0.1+0.9q_h],
$$

其中 $S_0$ 是 axis、$\psi$、surface、coordinate、volume-QS、$\iota$ 和 coil 七个分量的加权和，权重为

$$
(10,10,10,10,42,10,8).
$$

在固定分支内部，$q_\downarrow$、smoothstep、加权 RMS、Biot--Savart 场和线性最小二乘都是可微的。以下部分只分段光滑或不连续：

- 磁轴候选的出现、消失、排序和椭圆拓扑越界；
- 最大可行 $s$ 水平切换，以及 `ok/no_surface/drift_rejected/flux_rejected` 状态切换；
- 有效点 compaction 和径向 bin 成员变化；
- `min/max/clip`、P95 活跃样本和最近线圈点对切换；
- QH helicity gate、$\iota$ gate 和尺寸饱和点的折点；
- flow 电流规范化中最大电流索引和符号切换。

因此每次可微前向必须记录 branch fingerprint，至少包括：

- score ABI、代码 commit、动态库 SHA-256、目标 helicity；
- 选中磁轴的位置、候选序号、拓扑类别和稳定 margin；
- 选中 surface level、通过的 level 集合和长周期状态；
- flux edge、体点索引哈希、径向 bin 计数和有效点数；
- coil P95/max 的活跃样本、最近点对和电流最大项；
- 总分中各个 clip/min/gate 当前所处分支。

方向扰动若改变 fingerprint，不纳入“同一光滑分支上的梯度相对误差”，但必须计入 branch-change rate。不能通过只保留稳定方向来隐藏实际优化中的不可行性问题。

## 4. 分层梯度组分

| 层级 | 累计加入的梯度 | 难度 | 对高分优化的重要性 | 本层目的 |
|---|---|---:|---:|---|
| G0 | score 代数、cache、branch fingerprint、方向导数 oracle | 低 | 必需基础 | 保证以后比较的是同一目标和同一分支 |
| G1 | coil engineering 的分段解析梯度 | 低到中 | 中低 | 验证 API、参数顺序、单位和 flow VJP |
| G2 | 固定 axis/$s$/$\psi$/体点/$\iota$ 的 volume-QS 与 $G$ 梯度 | 中高 | **最高** | 以最小范围触及当前 42% 权重的核心分量 |
| G3 | $\alpha/\iota$ LS 隐式 VJP与 coordinate/$\iota$ 分量 | 高 | 高 | 修复固定 $\iota$ 对方向和尺度的系统偏差 |
| G4 | $s$ LS、flux 标定、固定 level 射线根和体点移动 | 很高 | 中到高，待证据决定 | 让局部目标跟随磁面几何变化 |
| G5 | 磁轴固定点与场线 ODE 的隐式/伴随梯度 | 很高 | 低到中 | 只在轴冻结导致明显误差时实现 |
| G6 | 候选发现、最大面选择、状态和硬 gate | 不做连续梯度 | 作为约束极重要 | 始终由真实 score 接受、回滚和重建 cache |

### 4.1 G0：梯度 oracle 和实验性 API

生产 ABI-9 前向接口保持不变。新分支增加独立的实验性接口，概念上分为：

```text
score_and_build_cache(coils, config) -> exact_result, cache
local_component_vjp(cache, component_mask) -> gradients, diagnostics
destroy_cache(cache)
```

cache 只能用于生成它的中心位型，不能跨不一致 fingerprint 复用。VJP 返回实际物理输入顺序下的梯度：每根线圈的 $x/y/z$ Fourier 系数和电流，并分别输出 `coil`、`volume_qs`、`coordinate`、`iota`、gate 与局部总目标的梯度，便于做累计消融。

G0 同时建立两个 oracle：

1. **内部光滑 oracle**：对某个固定 cache 的同一局部代理做多尺度中心差分，验证实现本身。
2. **完整黑箱 oracle**：重新运行真实 ABI-9 score，衡量局部代理对真实方向响应解释了多少。

内部 oracle 用于判定代码是否正确；完整 oracle 用于判定这个正确的局部梯度是否真的值得用于优化。二者不能混为一个指标。

### 4.2 G1：线圈工程梯度

线圈 Fourier 曲线 $\boldsymbol r(t;c)$ 对系数线性，因此以下量可以直接解析微分：

$$
L=\int|\boldsymbol r'(t)|\,dt,
\qquad
\kappa=\frac{|\boldsymbol r'\times\boldsymbol r''|}{|\boldsymbol r'|^3},
$$

以及高阶模能量和电流尺度。P95、最大曲率、最近线圈点对和最小半径在活跃索引固定时使用该活跃项的分段导数；一旦活跃项切换则由 fingerprint 标记。第一版不使用 softmin 偷换正式 score，softmin 只可作为后续独立优化代理对照。

该层应能给出非常严格的数值回归，但它的物理收益有限：高分 QH 解的主要改进通常来自 volume-QS，而不是 coil 分量。G1 通过后应立即进入 G2，不在工程项上做长时间优化实验。

### 4.3 G2：固定几何前端的体 QH 梯度

固定当前 axis、$s$、$\psi$、体点、权重和 $\iota$ 后，点级 QH residual 为

$$
r_i=
\frac{(\iota-N_{\rm FP})A_i-GC_i}{|\boldsymbol B_i|^3},
$$

$$
A_i=(\boldsymbol B_i\times\nabla\psi_i)\cdot\nabla|\boldsymbol B_i|,
\qquad
C_i=\boldsymbol B_i\cdot\nabla|\boldsymbol B_i|.
$$

这部分对 $\boldsymbol B$、$\nabla\boldsymbol B$、$G$ 和固定权重下的 RMS 连续可微。实现分两步：

1. 先实现点级 QS 归约对 $\boldsymbol B$、$\nabla\boldsymbol B$、$G$ 的 VJP，并做标准伴随恒等式验证

   $$
   \langle Jv,w\rangle=\langle v,J^Tw\rangle.
   $$

2. 再实现分段 Biot--Savart 对线圈端点、Fourier 系数和电流的 CUDA VJP。体点先产生场伴随，场核再按 block 做局部归约，避免 100000 点直接对约 200--500 个参数产生高竞争原子加。

$G$ 对电流的直接导数必须包含在 G2 中，并严格沿用 ABI-9 的

$$
G=\frac{\mu_0I_{\rm link}}{2\pi}
$$

及当前电流符号约定。QA、QH、QP 三个连续误差和 QH helicity advantage 的内部梯度都应返回，不能只微分 QH RMS 而漏掉总分 gate。

G2 是第一个 go/no-go 关键点。如果其局部方向与完整 score 在早期和中期可行位型上没有稳定正相关，就先诊断冻结了哪些依赖，不直接继续堆 G3--G5。

### 4.4 G3：$\alpha/\iota$ 线性最小二乘隐式 VJP

$\alpha/\iota$ 拟合可写为带正则的最小二乘：

$$
c^*=\arg\min_c\|W(Ac-b)\|_2^2+\lambda\|Lc\|_2^2.
$$

G3 固定 axis、$s/\psi$ 和体点集合，但允许 $A$、$b$、相对磁场权重、$\alpha$ 系数和 $\iota$ 随线圈变化。反向不显式构造 $\partial c^*/\partial x$，而复用增广 QR 的 $R$ 因子，通过两次三角求解得到伴随。实现中不显式形成病态的 $A^TA$。

该层加入：

- volume-QS 对拟合 $\iota$ 的间接梯度；
- coordinate 分量和 $\iota$ 分量的梯度；
- $\alpha$ 法向投影和列归一化的依赖；
- 固定 bin 内权重的连续依赖。

体点集合、bin 边界和 compact 顺序仍被冻结。如果 G3 明显提高完整 score 方向余弦，说明 LS 隐式依赖是必要项；如果改进很小，则可暂不进入更昂贵的 G4。

### 4.5 G4：$s/\psi$ 与移动采样几何

G4 再分两级，避免一次性引入过多难以定位的误差。

**G4a：** 固定磁轴和采样网格，对 $s$ 的 ridge LS 做隐式 VJP，并让 $\psi(s)$ 四阶标定系数随 $s$ 与磁场变化。先冻结已选 surface level 和体点物理位置，验证 $\nabla\psi$ 数值变化。

**G4b：** 加入固定 level 下的射线根和体点移动。边界半径 $r_b$ 满足

$$
s(r_b,\theta,\phi;x)-s_{\rm edge}=0,
$$

在 $\partial s/\partial r$ 不接近零时可用

$$
\frac{\partial r_b}{\partial x}
=-
\frac{\partial s/\partial x}{\partial s/\partial r}
$$

得到隐式导数。随后还需考虑柱坐标体积权重、采样点位置和场值随位置移动的变化。若根导数病态、点跨 bin 或有效性改变，则该方向标记为分支变化，不返回伪精确梯度。

G4 不负责对“哪个 level 最大”求导。选中 level 变化仍由真实 score 检测并重建 cache。

### 4.6 G5：磁轴隐式梯度

固定当前磁轴候选 $q^*$ 后有

$$
P(q^*,x)-q^*=0,
$$

因此

$$
\left(I-\frac{\partial P}{\partial q}\right)
\frac{\partial q^*}{\partial x}
=\frac{\partial P}{\partial x}.
$$

左侧只有 $2\times2$，但右侧需要场线 ODE 对线圈参数的 tangent 或 adjoint，轴上全部 Hermite 数据也随之变化。该实现成本高，而且 axis 分量在当前高分轨迹上通常已接近饱和。

只有满足以下证据之一才进入 G5：

- G3/G4 在多个非局部最优样本上的完整方向余弦仍系统性偏低，且误差与轴位移强相关；
- 候选真 score 经常因同一椭圆轴缓慢移动而拒绝，而不是候选切换；
- 冻结轴与“扰动后重新找同一轴”的受控差分明确给出主导误差。

磁轴候选出现/消失、候选排序和椭圆拓扑切换永远不通过该公式强行穿透。

## 5. 从物理 token 梯度到潜空间梯度

必须先在物理线圈参数空间验证 G1--G3，再接 flow。这样如果潜空间结果错误，可以区分是评分器 VJP 还是 flow VJP 的问题。

当前 `integrate_flow` 使用 `torch.no_grad()` 的 FP32 RK4-256。探索分支需要增加数值步骤完全相同的可微版本，并使用分块 activation checkpointing，避免保存 256 步的全部 Transformer 激活。先比较每 8、16、32 个 ODE step checkpoint 的墙钟和显存，再选最便宜且不溢出的方案；不改用更低精度或更低阶 ODE 来掩盖成本。

flow 输出后的逆归一化不是单纯的逐维 affine：电流还会固定 $L^1$ 范数，并把最大绝对电流规范为正。因此 VJP 链为

$$
g_z
=J_F(z)^T
J_{\rm norm^{-1}}^T
J_{\rm current\ gauge}^T
g_x.
$$

当前最大电流索引和符号固定时，电流 gauge 分段可微；发生主导电流切换时必须由 fingerprint 标记。最终需用潜空间中心差分独立验证整个链，而不能只验证 PyTorch autograd 没有报错。

## 6. 跨优化尺度的验证位型

### 6.1 主验证轨迹：$N_{\rm FP}=6$、两基本线圈

该轨迹从 128 个 IID flow 潜变量中筛选起点，再连续运行 Adam 到 700 步；不是 QUASR 样本反向追踪结果。完整的每步潜变量、线圈和 ABI-9 结果均已保存。

| 阶段 | Adam step | score | volume-QS 分量 | 用途 |
|---|---:|---:|---:|---|
| 早期可行 | 0 | 74.4358 | 55.831 | 梯度应较明显，检查是否能抓住主要上升方向 |
| 过渡期 | 50 | 80.5050 | 70.633 | 检查梯度随优化尺度缩小后的稳定性 |
| 中高分 | 200 | 83.4689 | 78.777 | 已完整保存且此前优化仍稳定上升 |
| 成熟期 | 400 | 86.1233 | 83.685 | 梯度较小但仍有后续真实增益 |
| 可行边界 | 491 | 86.6414 | 85.444 | 压力测试；不要求局部梯度能跨越 `no_surface` 边界 |

对应资产为：

- `reports/assets/qh_small_condition_adam_nfp6_nc2_20260803/adam/trajectory/`
- `reports/assets/qh_small_condition_adam_nfp6_nc2_continue400_20260803/adam/trajectory/`
- `reports/assets/qh_small_condition_adam_nfp6_nc2_continue700_20260803/adam/trajectory/`

step 491 不能单独代表梯度有效性。它后续大量方向跨入 `no_surface`，适合验证 branch guard，但如果只看该点，很容易把“已接近可行域边界”误判成“梯度方法没有信号”。

### 6.2 交叉验证轨迹：$N_{\rm FP}=4$、两基本线圈

第二条轨迹同样由 128 个随机潜变量筛选后开始，完整保存 201 个状态：

| 阶段 | Adam step | score | volume-QS 分量 | 用途 |
|---|---:|---:|---:|---|
| 早期 | 0 | 78.8418 | 61.206 | 不同 $N_{\rm FP}$ 的早期信号 |
| 中期 I | 50 | 80.6125 | 63.471 | 与主轨迹同分数段但不同几何 |
| 中期 II | 100 | 82.3214 | 68.098 | 验证累计梯度组分的泛化 |
| 高分 | 184 | 85.7731 | 77.069 | 已知真实局部最优附近 |

资产位于 `reports/assets/qh_small_condition_adam_nfp4_nc2_20260803/adam/trajectory/`。

step 185 的 score 从 85.7731 回撤到 84.7042，历史四方向中有一对产生异常大的 13.17 分差和梯度 RMS 337.1。该点及其八个端点只进入**脏梯度压力测试**，不混入正常光滑方向的均值。

### 6.3 离散分支压力样本

另外保留两个不用于主梯度精度平均的压力点：

- $N_{\rm FP}=6$ step 300：选中 level 从 0.16 切到 0.08，score 暂降到 79.6831，用于验证 surface branch fingerprint。
- $N_{\rm FP}=6$ step 700：最后 67 轮没有更新，多数端点为 `no_surface`，用于验证可行边界下不会返回误导性的“大梯度”。

## 7. 梯度正确性验证

### 7.1 方向与步长

每个核心中心固定 32 个 seeded RMS-normalized 正交潜空间方向，并在所有 G1--G4 累计版本中复用。完整 score 的中心差分尺度使用

$$
h\in\{0.01,\ 0.005,\ 0.0025,\ 0.00125\}.
$$

$h=0.005$ 是当前 Adam 的实际扰动尺度；$0.01$ 检查更宽的优化尺度；更小两档只用于观察收敛或噪声放大，不能自动把最小 $h$ 当作真值。先前实验已证明过小差分会被离散毛刺主导，因此 oracle 采用“跨两个相邻尺度方向一致”而不是“$h$ 越小越可信”的规则。

对直接物理 token 的内部 VJP 单元测试，另使用按 Fourier Parseval 物理范数归一化的方向和更细的相对步长，因为固定 cache 的光滑目标不存在完整 score 的候选切换。

### 7.2 必须报告的指标

对每个中心、每个 $h$、每个累计梯度层报告：

1. 32 维预测方向导数与中心差分方向导数的 cosine similarity。
2. Pearson、Spearman、符号一致率和 top-quartile 上升方向命中率。
3. 最小二乘标定后的斜率、截距和相对 RMS；防止方向对但尺度完全错。
4. 完整 fingerprint 保持率，以及 `no_axis/no_surface/drift/flux` 各类变化率。
5. 各 score 分量方向导数及其对总预测的贡献，不只报告一个总 cosine。
6. G1、G1+G2、G1+G2+G3、后续 G4 的累计增益，判断每层是否真正解释新增方差。

结果分为两组：

- **same-branch 指标**：只回答连续 VJP 是否正确；
- **all-direction 指标**：把分支变化作为失败或约束事件，回答优化时是否实用。

### 7.3 预注册验收门槛

内部实现正确性要求比完整 score 有用性更严格：

| 验证层 | 最低门槛 |
|---|---|
| G1 工程项 | same-branch 32 方向 cosine $\ge0.999$，中位相对误差 $\le10^{-3}$ |
| 点级 QS VJP | adjoint identity 相对误差 $\le10^{-4}$（FP32 归约可放宽至 $5\times10^{-4}$） |
| Biot--Savart VJP | 固定点方向导数 cosine $\ge0.995$，中位相对误差 $\le1\%$ |
| 固定 cache 的 G2/G3 | same-surrogate cosine $\ge0.98$，中位相对误差 $\le5\%$ |
| 对完整 ABI-9 score 的实用性 | 在两个早期/中期中心上，$h=0.005$ 的方向 cosine 中位数 $\ge0.5$ 且符号一致率 $\ge70\%$ |

完整 score 门槛不要求 step 491 满足；该点主要检查边界识别。若 G2 对内部代理完全正确但对完整 score 的方向 cosine 低于 0.5，则说明冻结依赖遗漏过多，应根据分量误差决定进入 G3 或 G4，而不是改动阈值掩盖问题。

## 8. 优化有效性实验

梯度验证通过后，只在非局部最优的 step 0、50、200 以及另一轨迹 step 0、50 上做中短程优化。每个起点比较：

1. **现有基线**：4 方向反向差分 + 低动量 Adam。
2. **纯局部 VJP**：累计到当前已通过的最高梯度层，每步一个候选真 score。
3. **混合校正**：局部 VJP 加 1 个 antithetic 黑箱方向，用于在线尺度校准和错误方向检测。

三组必须使用相同起点、相同 flow checkpoint、相同 ABI-9 库、相同最大墙钟和相同真实 score 接受规则。比较不只按“20 步谁分高”，还按：

- 每新增 1 分所需墙钟；
- 每次完整 score 调用的增益；
- P50/P95 单步耗时和是否存在新长尾；
- 真 score 非下降接受率、回滚率和 cache 重建率；
- volume-QS、coil、surface、$\iota$ 与 helicity advantage 的变化；
- 是否更早进入 `no_surface` 边界锁定。

步长不固定照搬某一个样本。每个中心以其历史 Adam 最近 20 个正常更新的 latent RMS 为参考，测试 $0.5\times$、$1\times$、$2\times$ 三档信赖步长；step 0 使用该轨迹初期的实际更新尺度。这样能覆盖不同优化阶段，而不是用成熟期的小步长压制早期梯度。

第一轮每组最多 20 步。如果纯 VJP 在至少三个非成熟起点上没有比 SPSA 更好的“分数增益/墙钟”，则停止优化扩展，返回梯度误差分解；不提交数小时长跑来掩盖单步效率问题。

## 9. 性能与资源方案

开发期正确性测试先在单卡完成，避免多卡并发掩盖单例延迟。批量的 32 方向完整 score oracle 在四卡上统一收集后并行评分，每张卡只运行当前已验证的原生 worker；计时前检查 GPU 空闲，结束后检查无残留进程。

性能目标分两级：

1. 局部物理 VJP加 flow VJP 的墙钟不高于一次完整 score 的 1.5 倍，并且没有数据相关的无界迭代。
2. 含一次候选真 score 的完整优化步，P50 墙钟不超过现有四方向 SPSA 基线的 60%，P95 不超过 P50 的 2 倍。

G1 可在 C++ 端完成，成本应远小于 0.1 秒。G2 的主要开发量是 $\nabla\boldsymbol B$ 对线圈端点的伴随核；其运行目标是亚秒到数秒。G3 复用已有 QR 因子，理论上不应成为主要墙钟。G4/G5 的实现成本和寄存器压力更高，只有前层证据不足时才投入。

所有实验固定保存：

- 中心位型、潜变量、完整 score 和 component；
- component mask、cache/fingerprint、梯度版本和数值精度；
- 32 个方向、所有 $h$ 的端点和状态；
- token 梯度、normalizer VJP、flow VJP 和最终 latent 梯度；
- 每阶段 CUDA event 与墙钟、GPU pre/postflight；
- commit、flow checkpoint 和 score/gradient 动态库 SHA-256。

## 10. 开发阶段与停止条件

### 阶段 A：基础设施

- 新增实验性 gradient/cache API，不改 ABI-9 正常前向结果。
- 建立固定样本 manifest、branch fingerprint 和多尺度方向 oracle。
- 为 forward-only 路径做逐字段回归，要求启用梯度代码前后 ABI-9 输出一致。

**停止条件：** cache 改变现有 score、引入额外 CPU 回退或不能可靠识别已知 level 切换时，不进入梯度实现。

### 阶段 B：G1 与 flow VJP

- 实现工程项分段解析梯度。
- 实现可微 FP32 RK4-256、inverse normalization 和 current gauge VJP。
- 分别做 token 空间和 latent 空间差分验证。

**停止条件：** G1 无法达到严格内部门槛，先修参数布局、单位或 VJP 链，不进入 QS。

### 阶段 C：G2 核心体 QH

- 点级 QS VJP、$G$ 电流导数和 Biot--Savart VJP。
- 在九个核心中心上完成 G1 对 G1+G2 的累计比较。
- 运行最多 5 步的纯 VJP 提案烟测，所有候选仍由真 score 验收。

**停止条件：** 内部梯度正确但早期/中期完整 score 无稳定正相关时，先定位遗漏依赖；不直接进入长跑。

### 阶段 D：G3 与中短程对照

- 加入 $\alpha/\iota$ QR 伴随。
- 完成 20 步 SPSA、纯 VJP、混合校正的同起点比较。
- 决定 G4 是否有证据必要。

**成功条件：** 在多个非成熟起点上，真 score 增益/墙钟稳定优于现有 SPSA，且没有降低可行率或通过错误 gate 刷分。

### 阶段 E：条件性扩展

- 只有分量误差明确指向 $s/\psi$ 或轴依赖时才做 G4/G5。
- 只有局部优化得到超过既有样本的候选时，才运行一次完整 $\alpha+\nu$/Simsopt/DESC 验收。

## 11. 预期风险与处理原则

1. **近局部最优导致梯度很小。** 主验证集包含 74、78、80、82、83 分等非成熟点；93 分样本只可作末端负对照，不作为主体。
2. **小差分放大 score 毛刺。** 同时扫描 $h=0.01$ 到 0.00125，以跨尺度一致性判断，不迷信最小 $h$。
3. **局部代理方向正确但尺度错误。** 单独报告线性标定和 trust ratio，优化器使用真 score 校准步长。
4. **梯度把样本推过可行边界。** fingerprint、真 score、回滚和 cache 重建始终存在；step 491/700 专门验证这一点。
5. **手写 $\nabla\boldsymbol B$ VJP 推导错误。** 先做点级 adjoint identity，再做固定点方向差分，最后才连接总 QS。
6. **flow VJP 显存过大。** 使用分块 checkpoint；不降低 RK4 精度。若仍不可接受，再评估离散 adjoint，而不是回退到错误的低精度解码。
7. **新增梯度拖慢正常 score。** 实验接口与 ABI-9 forward 分离；未请求梯度时不得分配 cache 或执行反向。
8. **总分更高但物理质量退化。** 中短程报告全部 component 和 face/volume QH；真正突破既有最好值时仍执行标准完整物理评估。

## 12. 最终交付预期

本方向最终应交付：

- 独立、可关闭、不改变 ABI-9 前向的 C++/CUDA 局部 VJP API；
- 固定的跨优化尺度样本 manifest 和多尺度梯度验证程序；
- 每一梯度层的 component-level 正确性、完整 score 解释度和耗时消融；
- SPSA、纯 VJP 和混合校正的同起点中短程优化比较；
- 明确的 go/no-go 结论：哪一层已经足以加速，哪一层仍缺关键物理依赖；
- 若产生新的真实高分候选，再按固定完整评估流程提供面 QS、Poincare、白底彩色 $|B|$ 等高线、三维 HTML 和 DESC 全部图。

本计划的核心判断标准不是“成功写出一个梯度数组”，而是：该梯度能否在**非局部最优、不同优化阶段、不同 $N_{\rm FP}$ 的真实轨迹点**上，以更少完整 score 调用和更短稳定墙钟获得可由 ABI-9 真分及完整物理支线共同确认的改进。
