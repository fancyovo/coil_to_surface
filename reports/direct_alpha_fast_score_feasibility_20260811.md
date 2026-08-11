# 跳过 $s$、直接拟合 $\alpha/\iota$ 的快速评分可行性

日期：2026-08-11  
分支：`codex/direct-alpha-fast-score`  
范围：本报告只做理论与现有实现审计，不修改 score，也不运行数值实验。

## 结论先说

这个想法有一部分是成立的，而且确实值得做一个小规模原型：把

$$
\alpha(X,Y,\phi)
=\theta+\lambda(X,Y,\phi)-\iota(X,Y,\phi)\phi
$$

代入

$$
\boldsymbol B\cdot\nabla\alpha=0,
\qquad
\boldsymbol B\cdot\nabla\iota=0,
$$

只要 $\lambda$ 和 $\iota$ 都用线性基底展开，两个方程可以合并成一次固定规模的线性最小二乘。它不需要 Newton，也没有未知迭代次数。从数值形式上说，这条路符合项目追求的“GPU、固定预算、无长尾”。

但原始设想不能直接作为当前体 QS score 的完整替代，原因不是实现细节，而是少了一个物理量：**现有微分 QS 指标显式需要物理磁通梯度 $\nabla\psi$，而 $\alpha$ 和 $\iota$ 的直场线方程本身通常不能稳定、唯一地给出它。**

更具体地说：

| 判断 | 结论 |
|---|---|
| 一次联合线性 LS 能否拟合 $\lambda$ 和空间函数 $\iota$ | 能 |
| $\boldsymbol B\cdot\nabla\iota=0$ 能否自动把 $\iota$ 变成好的径向坐标 | 不能；常数就是精确零残差解 |
| 加大该方程权重能否解决问题 | 不能；反而更鼓励 $\iota$ 退化为常数 |
| $\alpha$ 和 $\iota$ 能否在某些样本上恢复磁面与物理磁通 | 能，但要求 $\iota(\psi)$ 单调且磁剪切不能太小 |
| 拟合残差能否直接等价于“最大有效磁面大小” | 不能；必须至少做径向分层和独立点验证 |
| 完全不拟合任何径向不变量还能否计算当前已验证的 $f_C$ | 不能 |
| 能否先做成便宜的快速代理 | 可以，而且这是最合理的第一步定位 |

因此我建议把第一版定义为**直接不变量代理 score**，而不是立即声称它给出了与当前相同的“物理体 QS”。只有当额外的 Clebsch 闭合、单调性和物理磁通恢复都通过时，才计算并输出当前定义的 $f_C$。

## 联合最小二乘实际在解什么

先在磁轴附近建立轴随动坐标 $(X,Y,\phi)$，令

$$
\theta=\operatorname{atan2}(Y,X).
$$

展开

$$
\lambda=\sum_j c_j f_j(X,Y,\phi),
\qquad
\iota=\sum_k d_k g_k(X,Y,\phi).
$$

对 $\alpha$ 正确求导后有

$$
\nabla\alpha
=\nabla\theta+\nabla\lambda
-\iota\nabla\phi
-\phi\nabla\iota.
$$

所以两个逐点残差是

$$
r_\alpha
=\boldsymbol B\cdot\nabla\theta
+\sum_j c_j\boldsymbol B\cdot\nabla f_j
-\sum_k d_k
\left[
g_k\boldsymbol B\cdot\nabla\phi
+\phi\boldsymbol B\cdot\nabla g_k
\right],
$$

$$
r_\iota
=\sum_k d_k\boldsymbol B\cdot\nabla g_k.
$$

$r_\alpha$ 和 $r_\iota$ 都对未知系数 $c_j,d_k$ 线性，因此可以一次求解

$$
\min_{c,d}
\left\|W_\alpha r_\alpha\right\|_2^2
+\eta\left\|W_\iota r_\iota\right\|_2^2
+\gamma_\lambda\left\|L_\lambda c\right\|_2^2
+\gamma_\iota\left\|L_\iota d\right\|_2^2.
$$

这里有三个容易写错的点：

1. 不能漏掉 $-\phi\boldsymbol B\cdot\nabla\iota$。只有在 $\boldsymbol B\cdot\nabla\iota=0$ 被精确满足时，它才不影响沿场方程；数值拟合时提前删掉会让两个残差不自洽。
2. $\lambda$ 必须是角向周期函数；$\alpha$ 本身允许含 $-\iota\phi$ 的世俗项。
3. 在 $\phi$ 的周期切口两侧，$\nabla\alpha$ 相差一个与 $\nabla\iota$ 平行的量。若 $r_\iota=0$，这不改变沿场不变量，也不改变 $\nabla\iota\times\nabla\alpha$；若只有近似成立，训练结果会依赖切口位置，因此需要成对周期样本或显式的切口一致性诊断。

当前实现的 $\alpha/\iota$ 拟合并不是这个问题。它已经拥有可信的 $s$ 和 $\rho$，拟合的是

$$
\alpha=\theta+\lambda(\rho,\theta,\phi)-\iota(\rho)\phi,
$$

并先从 $\boldsymbol B$ 中扣除相对 $s$ 面的法向小量，再拟合切向场。生产默认的 $\iota$ 是 $u=\rho^2=\psi/\psi_{\rm edge}$ 的三次多项式。对应实现见 [score_pipeline.cu](../gpu_backend/src/score_pipeline.cu)，物理与评分定义见 [GPU 原生体 QS 报告](gpu_native_volume_qs_score_report.md)。讲义也明确把 $\iota$ 写成 $\iota(\psi)$，而不是把它预先当作任意三维标量，见 [模块 4](../../../讲义/模块4-坐标系与周期性工具.md)。

## 为什么 $\iota$ 通常不能替代 $s$

### 常数退化不是调权重能修好的

方程

$$
\boldsymbol B\cdot\nabla\iota=0
$$

只说 $\iota$ 沿磁力线不变。任意常数都严格满足它。因此这条方程没有提供“从轴向外递增”的信息，也没有固定 $\iota$ 的横向尺度。

$r_\alpha$ 会利用极向推进率确定 $\iota$ 的平均值，所以联合问题不一定得到 $\iota=0$；但 $r_\iota$ 对常数部分完全没有约束。把 $r_\iota$ 权重 $\eta$ 调得更大，只会更强地压制 $\iota$ 的空间变化。若真实磁剪切本来就小，数值结果会自然靠近

$$
\iota(X,Y,\phi)\approx\iota_0,
$$

此时 $\alpha$ 仍可能拟合得很好，但 $\nabla\iota\approx0$，所以 $\iota$ 完全不能区分不同磁面。

这里要区分两个概念：

- 作为旋转变换，近常数的 $\iota$ 可以完全合理；
- 作为径向坐标，近常数的 $\iota$ 是退化的。

不能为了让算法可逆而奖励更大的 $|\nabla\iota|$，因为那等于人为奖励磁剪切，会改变原本的物理目标。

现有 69 个样本的生产三次 $\iota(\psi)$ 标定中，$\iota_{\max}-\iota_{\min}$ 的中位数为约 $0.302$，说明“所有样本都低剪切”并不成立；但这个分布有明显样本依赖，尤其不能据此保证优化后高分 QH 位形的 $\iota(\psi)$ 单调。第一项实验应直接检查每个样本的 $d\iota/d\psi$ 是否变号、最小绝对值和条件数，而不是先假定可逆。

### 两个不变量必须彼此独立

局部 Clebsch 表示需要两个独立的场线不变量：

$$
\boldsymbol B=\nabla\psi\times\nabla\alpha.
$$

磁力线是两个不变量等值面的交线。相关的几何论述可见 [Guiding-centre Lagrangian and quasi-symmetry](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/guidingcentre-lagrangian-and-quasisymmetry/83EF94257C74E7919D01CB0DF8C019A2)。只得到 $\alpha$，并不能得到一族嵌套磁面；还必须有第二个单值、横向梯度不为零的不变量。

若希望用 $\iota$ 充当第二个不变量，至少要满足

$$
\left|\nabla\iota\times\nabla\alpha\right|>0
$$

且 $\iota$ 的等值面闭合、嵌套、在目标体积内不出现径向折返。低剪切、$\iota(\psi)$ 极值和有理面附近的非唯一性都会破坏这个条件。普通最小二乘的训练 residual 很小，并不自动证明这些全局性质。

### “最后再标定”只有在可逆时才成立

在理想情况下，如果拟合出的确实是物理旋转变换，并且

$$
\frac{d\iota}{d\psi}\neq0,
$$

那么可以反过来写 $\psi=\psi(\iota)$，并令

$$
F(\iota)=\frac{d\psi}{d\iota},
\qquad
\nabla\psi=F(\iota)\nabla\iota.
$$

此时还可以增加第二次、仍然是线性的拟合：

$$
\boldsymbol B
\approx
F(\iota)\,
\nabla\iota\times\nabla\alpha.
$$

若 $F$ 只用 $\iota$ 的低阶基底展开，这一步可恢复物理磁通尺度，再积分得到 $\psi(\iota)$。这就是原始想法真正闭合的版本。

它的限制也很清楚：当 $d\iota/d\psi$ 很小时，$\nabla\iota$ 很小，而 $F=d\psi/d\iota$ 很大。两个量的乘积也许仍有限，但最小二乘会非常病态；当 $\iota(\psi)$ 不单调时，单值的 $F(\iota)$ 根本不存在。因此“最后再标定”不是普适的免费后处理，而是一条需要先验收可逆性的条件分支。

## 为什么当前体 QS 不能只靠 $\alpha$ 和 $\iota$

当前代码计算的是指定 helicity $(M,N)$ 的 two-term 微分 QS 残差。真空场中 $I=0$，其核心形式为

$$
A=(\boldsymbol B\times\nabla\psi)\cdot\nabla B,
\qquad
C=\boldsymbol B\cdot\nabla B,
$$

$$
f_C=(M\iota-N)A-MGC.
$$

代码中明确使用

$$
\nabla\psi
=\frac{d\psi}{ds}\nabla s.
$$

这可在 [score_pipeline.cu](../gpu_backend/src/score_pipeline.cu) 的 `compute_qs_metric_kernel` 直接核对。two-term 形式本身也明确包含 $\nabla\psi$，见论文 [Measures of quasisymmetry for stellarators](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/measures-of-quasisymmetry-for-stellarators/01B9DFE86A23964F331E0E0615B4E7A2)。

任意把 $\psi$ 换成一个未标定的标签 $\chi=g(\psi)$ 会令第一项乘上 $g'(\psi)$，而 $MGC$ 不会同比例变化。因此它不是简单的坐标重命名。只有恢复了正确的物理磁通尺度，才能与当前已交叉验证的 $f_C$ 保持同一含义。

这也符合 QS 的理论结构：严格 QS 本身蕴含单值磁通函数的存在，而不是只需要一个角向场线标签。必要充分条件中显式出现 $\boldsymbol B\times\boldsymbol u=\nabla\psi$，见 [Necessary and Sufficient Conditions for Quasisymmetry](https://arxiv.org/abs/2004.11431)。

所以新路径有两种诚实的输出：

1. 没有恢复 $\nabla\psi$ 时，只输出“直场线/全局不变量代理”，不能把它命名为当前的体 QS error；
2. 通过 $F(\iota)$ 闭合验收后，才使用 $\nabla\psi=F\nabla\iota$ 计算同一定义的 $f_C$。

## 残差能否自然代表有效磁面大小

“外部磁面越坏，整个体积里的拟合 residual 越大”这个直觉有价值，但直接取一个全局平均 residual 不够。它至少混合了五件事：

- 真正的磁岛或混沌；
- 预先选取的几何采样管有多大；
- 基底阶数够不够；
- 轴附近与外层点数如何加权；
- $|\boldsymbol B|$、$|\nabla\alpha|$、$|\nabla\iota|$ 的尺度。

因此同一个磁场只要更换采样管半径或基底阶数，就可能得到显著不同的 residual。反过来，足够灵活的局部基底能在没有全局嵌套面的区域拟合出很小的局部沿场误差；常数 $\iota$ 更是天然零 residual。于是“残差小”等价于“存在大磁面”并不成立。

更合理的连续版本不需要离散搜索最大面，但仍要保留几何径向位置 $r$。在固定的轴随动采样管中，先在独立验证点上计算无量纲残差，例如

$$
e_\alpha
=\frac{|\boldsymbol B\cdot\nabla\alpha|}
{|\boldsymbol B|\,|\nabla\alpha|+\epsilon},
$$

$$
e_\iota
=\frac{|\boldsymbol B\cdot\nabla\iota|}
{|\boldsymbol B|\,|\nabla\iota|+\epsilon}.
$$

再按几何半径分层得到 $e(r)$，定义平滑可信度

$$
p(r)=\sigma\!\left(\frac{\tau-e(r)}{T}\right),
$$

最后计算

$$
V_{\rm eff}
=\int_{V_{\rm tube}}p(r)\,dV.
$$

这能把原来的“13 个候选层中选一个”改成连续有效体积，避免离散跳变，也不需要追求严格最大面。但它仍不是无条件的磁面体积：$r$ 是几何半径，不是磁通坐标，$V_{\rm eff}$ 必须用现有可信面结果标定。

还必须加两类防作弊门：

- **非退化门**：检查 $\nabla\iota\times\nabla\alpha$ 的有效秩、$\iota$ 等值面的闭合与径向顺序；常数 $\iota$ 不能靠零 residual 得高分。
- **独立点门**：训练点和验证点必须分开；有效体积只能由验证 residual 形成，不能由训练 residual 形成。

若不做这两项，新 score 最容易学到的不是“大而好的嵌套磁面”，而是“用平滑低阶函数在固定采样管里把沿场导数平均压低”。

## 值得实际尝试的版本

我建议下一步先实现一个明确可失败的原型，而不是直接替换生产 score：

```text
已有磁轴初值
  -> 建立固定、轴随动的几何采样管
  -> 一次联合线性 LS 拟合 lambda(X,Y,phi) 与 iota(X,Y,phi)
  -> 在独立点计算 r_alpha、r_iota 和径向可信度
  -> 检查 iota 是否非退化、等值面是否有序
      -> 不通过：只输出低置信度代理分，不计算物理体 QS
      -> 通过：线性拟合 F(iota)，恢复 grad(psi)=F grad(iota)
          -> 检查 B ~= F grad(iota) x grad(alpha)
          -> 通过后计算当前定义的 f_C
```

第一版不做最大磁面搜索，也不做长周期追踪。它的目的不是证明新方法必然正确，而是回答三个可量化问题：

1. 在多少 QH 样本上，$\iota$ 确实能作为非退化、单调的径向标签？
2. 直接拟合 residual 与当前可信磁面大小、当前 $f_C$ 的排序关系有多强？
3. 假高分率是否足够低，能否安全用作优化 score 或至少作为廉价前筛？

建议验收表如下：

| 项目 | 与什么比较 | 通过标准的性质 |
|---|---|---|
| $\iota$ 数值 | 当前 $\iota(\psi)$、长场线相位推进 | 均值、径向趋势和符号一致 |
| 径向可逆性 | 当前 $s/\psi$ 层 | $\iota$ 等值面闭合、有序，$d\iota/d\psi$ 不反号 |
| Clebsch 闭合 | 独立体点上的 $\boldsymbol B$ | $\boldsymbol B-F\nabla\iota\times\nabla\alpha$ 残差受控 |
| 有效体积 | 当前连续磁面大小与完整 Poincare | 高分段排序近似保持，尤其不能出现小面假高分 |
| QS | 当前物理 $f_C$ 与完整面 QS | 高分段相关和 top-k 重合，而不只看全体相关 |
| 鲁棒性 | QUASR 好样本、随机样本、优化轨迹点 | 失败返回有限低分，无异常长尾 |
| 速度 | 同一张空闲 RTX 5090、同一轴初值 | 报告 P50/P95 和各阶段，不只报单样本最好值 |

有一个便宜但很关键的前置判定：先复用已有结果统计 $d\iota/d\psi$ 的变号率和最小绝对值。如果大量目标 QH 样本在所需体积内接近零剪切或不单调，那么“用 $\iota$ 代替 $s$”应在写 CUDA 之前停止；这不是调基底或调权重能修复的问题。

## 速度上可能省多少

生产主线在“已有磁轴初值、严格续接”的 69 样本统计中，单次 score 的 P50 约为 $0.55$--$0.67\ \mathrm{s}$，随磁轴核验模式略有差异。grid-48 基线的主要阶段是：

| 当前阶段 | P50 |
|---|---:|
| 磁轴链路 | 删除 FP64 复核后约 $140\ \mathrm{ms}$ |
| $s/\psi$ 拟合与独立验证 | 约 $143\ \mathrm{ms}$ |
| 连续磁面筛选 | 约 $119\ \mathrm{ms}$ |
| $\alpha/\iota$ 拟合 | 约 $85\ \mathrm{ms}$ |
| 磁通标定 | 约 $38\ \mathrm{ms}$ |

直接路线理论上可删除当前 $s/\psi$、连续磁面筛选和磁通标定的大约 $300\ \mathrm{ms}$，所以值得探索。但不能据此直接预言新 score 只剩 $0.25\ \mathrm{s}$：

- 当前 $\iota$ 只有 4 个径向系数；新方案若给 $\iota(X,Y,\phi)$ 使用与 $\lambda$ 相近的三维基底，QR 列数可能接近翻倍；高瘦 QR 的主成本近似随列数平方增长，求解时间可能接近四倍。
- 若再拟合 $F(\iota)$，会增加一次较小的线性问题和独立闭合验证。
- 磁轴与线圈准备仍然保留，约 $160\ \mathrm{ms}$ 的下界不会消失。

因此要获得实际加速，$\iota$ 必须使用明显更小、带物理约束的低阶基底，或者利用块结构消元；不能简单复制 $\lambda$ 的 2268 列基底。速度目标应以完整原型的单卡 P50/P95 决定，而不是以“删除了三个阶段”决定。

## 最终判断

原始提议里最有价值的部分，是用一次联合线性问题同时寻找直场线标签和另一个沿场不变量，并用连续 residual 代替离散最大面搜索。这有机会得到更平滑、更便宜的优化目标。

原始提议里不成立的部分，是默认把物理旋转变换 $\iota$ 同时当成普适径向坐标，并认为联合 LS residual 会自然给出磁面大小和现有体 QS。这个等价关系只在 $\iota(\psi)$ 单调、剪切不退化、Clebsch 闭合成立时存在；这些条件对 QH 不能预设为真。

所以这条路线**值得做原型，但必须按条件分支命名和验收**：

- 未恢复物理磁通时，它是“直接不变量代理 score”；
- 成功恢复 $F(\iota)$ 并通过独立闭合后，它才是“跳过原 $s$ 拟合的物理体 QS score”；
- 若低剪切导致 $\iota$ 退化，则必须承认这条路线不适用于该样本，而不能靠更高权重、更多基底或奖励 $|\nabla\iota|$ 强行修成径向坐标。

如果最终需要一个对低剪切也普适的版本，就必须另求一个非退化径向不变量 $\chi$，再拟合 $\iota(\chi)$。它可以采用比当前 $s$ 更快的算法，也可以避免离散磁面搜索，但从物理角色上说，$\chi$ 仍然就是一个新的 $s$；这部分信息不能被 $\alpha$ 单独消掉。
