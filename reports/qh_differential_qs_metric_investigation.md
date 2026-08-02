# QH 微分体积 QS 指标矛盾调查

日期：2026-08-02

## 1. 结论先行

当前看到的矛盾不是 QH 磁面本身造成的，主要是微分 QS 实现中的一个确定性单位约定错误：

1. 微分 QS 的 $f_C$ 公式结构正确。
2. Python 和 C++/CUDA 都把“整圈角坐标”约定下的 $G$，直接代入了使用弧度角坐标的 $f_C$ 公式，使 $G$ 恰好多了 $2\pi$。
3. 错误的 $G$ 会显著抬高 QA 和 QH 残差；QP 项不含 $G$，因而几乎不受影响。这正好解释了“QP 反而远低于 QH”的异常现象。
4. QA、QH、QP 当前保存的输出还使用了不一致的螺旋度归一化：QH 保存原始值，QA 的螺旋度范数为 1，QP 却已经除以 $N_{\mathrm{FP}}$。原始三列不能直接横向比较。
5. 对两个已经完成完整物理评估的高分 QH 样本作精确代数重算后，修正指标均恢复为明确的 QH 排序。归一化 QH 误差分别比 QP 低约 6.0 倍和 10.3 倍，比 QA 低约 24 倍和 41 倍。
6. 独立的 LS/Newton Boozer 面傅里叶诊断也显示 QH 误差比 QA/QP 低 26--94 倍，与修正后的微分指标定性一致。

因此，当前 QH/QA/QP 的异常相对大小已经有明确解释，不需要诉诸“显著 QH 的磁面恰好具有很差的体平均 QH”这种物理解释。

## 2. 当前指标究竟在算什么

代码使用的微分 QS 条件为

$$
f_C = (M\iota-N)(\mathbf B\times\nabla\psi)\cdot\nabla B
      -(MG+NI)\mathbf B\cdot\nabla B,
$$

并统计无量纲量 $f_C/B^3$ 的体积加权 RMS。真空区中 $I=0$。

这一公式和 DESC 的实现一致：DESC 也使用相同的两项结构。问题不在公式的符号组合，而在传入公式的 $G$ 属于哪一种角坐标约定。

本项目的体积链路明确使用：

- 几何环向角 $\phi$ 和磁面角 $\theta$ 均以弧度表示；
- $\psi$ 是环向磁通除以 $2\pi$，代码诊断名也是 `edge_psi_toroidal_per_radian`；
- 相位导数按 $m\theta-nN_{\mathrm{FP}}\phi$ 计算。

在弧度 Boozer 坐标中，协变表示为

$$
\mathbf B = I\nabla\theta + G\nabla\zeta + \cdots .
$$

对轴对称真空环向场，安培定律给出

$$
B_\zeta = \frac{\mu_0 I_{\mathrm{link}}}{2\pi R}.
$$

由于 $\nabla\zeta=\mathbf e_\zeta/R$，弧度约定下应当使用

$$
G_{\mathrm{rad}}=\frac{\mu_0 I_{\mathrm{link}}}{2\pi}.
$$

当前 Python 和 CUDA 实际使用的是

$$
G_{\mathrm{current}}=\mu_0 I_{\mathrm{link}}
                    =2\pi G_{\mathrm{rad}}.
$$

它对应的是把环向角归一化到 $[0,1)$ 的“整圈”坐标所使用的协变系数，不能直接代入弧度形式的微分公式。

相关实现位置：

- Python 的磁通标定和 $G$：[volume_qs.py](../stellarator_eval/volume_qs.py#L610)、[volume_qs.py](../stellarator_eval/volume_qs.py#L883)
- C++/CUDA 的 $G$ 和微分指标：[score_pipeline.cu](../gpu_backend/src/score_pipeline.cu#L2982)、[score_pipeline.cu](../gpu_backend/src/score_pipeline.cu#L3033)
- DESC 的同一公式与 $G=\langle B_\zeta\rangle$：[\_omnigenity.py](../../DESC/desc/compute/_omnigenity.py#L676)、[\_profiles.py](../../DESC/desc/compute/_profiles.py#L1855)

以 score 61.339 的样本为例，三个基准线圈电流为
$223824.45$、$304200.81$、$162663.09\ \mathrm A$，$N_{\mathrm{FP}}=4$。当前对称复制规则给出的链接电流为

$$
I_{\mathrm{link}}=2N_{\mathrm{FP}}\sum_k|I_k|
                 =5.525506875\times10^6\ \mathrm A.
$$

于是

$$
G_{\mathrm{current}}=6.9435567\ \mathrm{T\,m},\qquad
G_{\mathrm{rad}}=1.1051014\ \mathrm{T\,m}.
$$

两者严格相差 $2\pi$。

## 3. 为什么错误会表现为 QP 特别低

记

$$
x=\frac{(\mathbf B\times\nabla\psi)\cdot\nabla B}{B^3},
\qquad
y=\frac{G\mathbf B\cdot\nabla B}{B^3}.
$$

当前目标 QH 使用 $(M,N)=(1,N_{\mathrm{FP}})$，于是

$$
e_{\mathrm{QA}}=\iota x-y,
\qquad
e_{\mathrm{QH}}=(\iota-N_{\mathrm{FP}})x-y.
$$

而纯 QP 的 $(M,N)=(0,N_{\mathrm{FP}})$ 消去了 $G$ 项：

$$
e_{\mathrm{QP}}=-N_{\mathrm{FP}}x.
$$

所以 $G$ 多出 $2\pi$ 时，QA 和 QH 都被错误的 $y$ 项显著污染，QP 则不变。当前现象并不是 QP 在物理上胜过 QH，而是 QP 恰好绕开了出错的变量。

## 4. 还存在一个展示尺度问题

$f_C$ 对 $(M,N)$ 是线性的。如果把同一个螺旋度写成 $(kM,kN)$，原始 $f_C$ 也会乘以 $k$。因此不同模式的非零残差在比较前必须声明归一化。

当前 CUDA 输出混用了三种口径：

- `qs_global_error`：QH 原始 RMS，没有除以 $\sqrt{M^2+N^2}$；
- `qs_qa_global_error`：QA 原始 RMS，但 QA 的范数本来就是 1；
- `qs_qp_global_error`：已经除以 $N_{\mathrm{FP}}$，即单位螺旋度 QP 值。

评分内部后来把目标 QH 除以 $\sqrt{M^2+N^2}$，但保存给分析使用的三列仍不一致。这会进一步放大直观误判。

$\sqrt{M^2+N^2}$ 是当前评分选择的一种可声明的标度，不是唯一的坐标不变量。合理做法是同时输出：

$$
\epsilon_{\mathrm{raw}}=\operatorname{RMS}(f_C/B^3),
\qquad
\epsilon_{\mathrm{hel}}=rac{\epsilon_{\mathrm{raw}}}{\sqrt{M^2+N^2}},
$$

并且所有 QA/QH/QP 都遵循同一规则。精确 QS 时的零值不受归一化影响；有限误差的跨模式数值比较则必须使用统一口径。

## 5. 两个高分样本的精确重算

本次没有重新运行 GPU。当前生产配置令 $\iota$ 在体积内为常数，因此保存的 QA、QH、QP RMS 已经包含重算所需的全部二阶矩。

令旧 $G$ 为新 $G$ 的 $k=2\pi$ 倍，并令 $N=N_{\mathrm{FP}}$。由

$$
e_{\mathrm{QH}}^{\mathrm{old}}=e_{\mathrm{QA}}^{\mathrm{old}}-Nx
$$

可以精确恢复

$$
\left\langle e_{\mathrm{QA}}^{\mathrm{old}}x\right\rangle
=\frac{(\epsilon_{\mathrm{QA}}^{\mathrm{old}})^2
      +N^2\epsilon_{\mathrm{QP}}^2
      -(\epsilon_{\mathrm{QH}}^{\mathrm{old}})^2}{2N}.
$$

修正后的逐点误差为

$$
e_{\mathrm{QA}}^{\mathrm{new}}
=\frac{1}{k}e_{\mathrm{QA}}^{\mathrm{old}}
 +\left(1-\frac{1}{k}\right)\iota x,
$$

$$
e_{\mathrm{QH}}^{\mathrm{new}}
=\frac{1}{k}e_{\mathrm{QA}}^{\mathrm{old}}
 +\left[\left(1-\frac{1}{k}\right)\iota-N\right]x.
$$

对同一批采样点和权重取二阶矩即可得到精确的新 RMS：

| 样本 | 旧 QH 原始值 | 旧 QH/螺旋度 | 旧 QA | QP/螺旋度 | 修正 QA | 修正 QH 原始值 | 修正 QH/螺旋度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| score 61.339 | 0.345847 | 0.083880 | 0.457984 | 0.029614 | 0.119169 | 0.020415 | 0.004951 |
| score 63.691 | 0.339298 | 0.082292 | 0.457505 | 0.030118 | 0.120479 | 0.012113 | 0.002938 |

修正后：

- score 61.339 样本的单位螺旋度 QH 比 QP 低 6.0 倍，比 QA 低 24.1 倍；
- score 63.691 样本的单位螺旋度 QH 比 QP 低 10.3 倍，比 QA 低 41.0 倍；
- 即使不做 QH 螺旋度归一化，两个样本的修正 QH 原始值也都已经低于 QP 的单位值。

这直接消除了用户指出的矛盾。

注意：这里只能精确重算保存了完整二阶统计的全局体积 RMS。边缘项没有保存对应的 QA/QP 二阶矩与协方差，因此不能从旧产物精确重构修正后的总 score；总 score 必须在修正代码后重新计算。

## 6. 与独立 Boozer 面结果交叉验证

完整评估中的 LS/Newton Boozer 面使用独立的面上傅里叶 QS 定义，不依赖这里的体积 $G$ 实现。两个样本均表现出明确的 QH 优势：

| 样本与选中面 | 面 QA 误差 | 面 QH 误差 | 面 QP 误差 | QH 相对 QA/QP |
|---|---:|---:|---:|---:|
| score 61.339，$s=0.64$ | $3.4830\times10^{-3}$ | $1.3327\times10^{-4}$ | $3.5146\times10^{-3}$ | 低 26.1/26.4 倍 |
| score 63.691，$s=0.36$ | $4.1134\times10^{-3}$ | $4.4102\times10^{-5}$ | $4.1245\times10^{-3}$ | 低 93.3/93.5 倍 |

证据文件：

- [score 61.339 标准面摘要](assets/qh_adam_topology_fixed_61p339_full_eval_20260801/candidates/s_0p64/standard_rho_1/summary.json)
- [score 63.691 标准面摘要](assets/qh_adam_low_momentum_63p691_full_eval_20260802/candidates/s_0p36/standard_rho_1/summary.json)

面傅里叶误差和体积微分 RMS 的定义、采样区域与归一化不同，所以绝对数值不应直接相等；但它们对“哪个对称模式明显占优”的判断应一致。修正后的微分指标满足这一要求，旧指标不满足。

## 7. 对已有结果的影响

### 7.1 已失效的部分

当前生产库及其历史语料中的以下量不能再按物理标定后的微分 QH 指标解释：

- 体积 QH、QA 数值及其相对 QP 的比较；
- 依赖该体积项的 volume-QS score 分量；
- 依赖 QA/QH/QP 相对关系的 QH 竞争门；
- 使用旧 score 进行的绝对阈值解释和不同版本之间的数值比较。

Python 与 C++/CUDA 使用了同一个错误公式常量，因此二者互相吻合不能构成独立验证。现有单元测试只验证了 $G$ 的符号和电流整体反号不变性，没有验证安培定律给出的绝对幅值，所以没有发现共享的 $2\pi$ 错误。

### 7.2 仍然有效的部分

- 已找到的线圈几何本身没有因此失效；
- 磁轴、$\psi$、磁面存在性、Poincare、LS/Newton、DESC 等不使用这项错误 $G$ 的独立结果仍然有效；
- 两个高分样本的完整评估明确证明它们确实具有 QH 磁面；
- 旧 score 和优化历史仍可作为“旧评分 ABI 下可复现的搜索记录”，但不能继续当作物理标定后的 QH 分数。

## 8. 建议的修正与验收顺序

本报告只完成调查，没有静默修改生产评分。建议下一步按以下顺序处理：

1. Python 和 C++/CUDA 同时改为 $G=\mu_0 I_{\mathrm{link}}/(2\pi)$。
2. 对 QA/QH/QP 统一输出原始 RMS 和单位螺旋度 RMS，字段名明确区分，禁止再混表。
3. 增加安培定律绝对量级测试：给定轴对称真空环向场，必须恢复 $G=\mu_0 I/(2\pi)$。
4. 增加与 DESC 的独立点对点 $f_C$ 对照；不能只用 Python 对 CUDA，因为两者容易共享同一约定错误。
5. 用固定样本集重新标定 volume-QS score、QH 竞争门和相关阈值，并重评两个完整评估样本及一批 QUASR QH/QA 对照。
6. 更新 score ABI、动态库哈希和语料版本；旧语料保留但显式标为旧定义，不能与修正分数混合训练或比较。

## 9. 最终判断

用户观察到的现象主要是代码单位约定错误，不是三种对称性天然具有不可比较的巨大尺度，也不是这些显著 QH 磁面在体平均意义下突然失去 QH。

更准确地说：

- **主要矛盾来源**：$G$ 多乘了 $2\pi$；
- **次要误导来源**：QA/QH/QP 输出归一化不一致；
- **修正后的物理结论**：两个已完整验证样本的微分体积指标都明确选择 QH，且独立 Boozer 面诊断给出相同排序；
- **工程结论**：当前生产 score 需要版本化修正和重新标定，不能只改一行常量后继续沿用旧阈值与旧语料。
