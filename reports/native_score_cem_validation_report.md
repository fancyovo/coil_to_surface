# 原生 GPU score 的 CEM 黑箱优化验证

## 1. 结论先行

本轮任务完成了两件事：

1. 原生 C++/CUDA score 已合并到 `main`，旧的“磁轴、$psi$、Simsopt Boozer
   LS/Newton、DESC”路径没有被覆盖；
2. 在新分支上从随机 PCA latent 出发，用固定预算 CEM 分别优化 QA 和 QH score，
   再把最高分 QH 样本交给旧路径和 DESC 做独立复核。

主要结论如下。

- **score 可以作为黑箱优化目标。** QH 从随机起点 `10.005` 提升到 `71.325`，
  QA 从 `9.977` 提升到 `69.822`。QH 已超过 QUASR 独立 1000 样本的 P90
  `70.89`，但尚未超过数据集最大值 `78.02`。
- **高分样本不是伪解。** 旧路径在多个尺度上都找到 Boozer LS/Newton 可解面；
  Poincare 图显示 16 条场线形成连续嵌套层。体积 $0.0752\,\mathrm{m}^3$ 的中等大面上
  Simsopt QH error 为 `1.25%`，最大计算可解面体积 $0.1231\,\mathrm{m}^3$，QH error
  为 `4.78%`。
- **DESC 给出了正面的补充验证。** 固定最大边界求解后初态和末态均为 nested；
  最终归一化 force 的 mean、p95、max 分别为 $2.04\times10^{-4}$、
  $5.75\times10^{-4}$、$3.89\times10^{-3}$。优化器因达到 50 次上限而未返回
  `success`，因此应表述为“低残差、嵌套、但未按停止条件正式收敛”。
- **不能把结果解释成已经得到极佳 QH。** 原生 score 的 `volume_qs` 分量只有
  `3.45/100`；旧路径中 QA/QH/QP 三种误差处于相近量级，DESC 的 QH 图也只显示
  中等对称性。当前实验验证的是“总 score 能优化出真实且总体较好的位形”，尚未严格
  证明 QH 标签具有很强的目标特异性。

因此，对本次问题最准确的回答是：**CEM 能有效优化当前 score，且高分随机样本确实
对应较大嵌套磁面和小到中等的 QS error；但 score 的下一轮改进应增强大体积 QS 分量
和 QA/QH/QP 之间的区分力。**

## 2. 分支与主线状态

合并前的稳定主线为 `c5ac4d8`。原生 GPU score 分支以非 fast-forward 方式合并到
`main`，合并提交为：

```text
7f8db81 Merge native GPU score pipeline
```

核对结果：

- `stellarator_eval/pipeline.py`、`stellarator_eval/surface.py`、
  `stellarator_eval/cli.py` 相对合并前稳定主线没有变化；
- 原生 score 以新增 C++/CUDA backend、脚本和独立接口进入主线；
- 旧磁面、Simsopt Boozer 和 DESC 路径仍可直接运行，本报告的全面复核正是调用该路径；
- CEM 实验在独立分支 `score-cem-validation` 上进行；
- 当前测试为 `31 passed`。

这满足“合并新 score，但不覆盖旧 LS/Newton 和 DESC 路径”的要求。

## 3. CEM 实验设计

### 3.1 搜索空间

搜索使用旧 `coil` 项目已经训练好的 whitened PCA 表示：

- 一个 base coil；
- `nfp=3`；
- 每根 base coil 使用 64 维 latent；
- latent 逐维限制在 $[-3,3]$；
- PCA 文件和原生动态库均记录 SHA-256，结果可追溯。

线圈电流的 $L_1$ 总量固定为随机中心的值。QH 实验固定为约
$2.048\times10^5\,\mathrm{A}$，QA 实验固定为约 $2.825\times10^5\,\mathrm{A}$。
这样可以防止优化器仅通过整体缩放电流来抬高 score。

### 3.2 固定预算 CEM

两组实验参数一致：

| 参数 | 数值 |
| --- | ---: |
| population | 32 |
| elite | 8 |
| generation | 8 |
| 每个目标总评估数 | 256 |
| 初始标准差 | 0.35 |
| 平滑系数 | 0.55 |

每代从对角高斯分布采样，选择最高分的 elite。若平滑系数为 $\eta$，更新为

$$
\boldsymbol\mu_{t+1}
=(1-\eta)\boldsymbol\mu_t+\eta\,\overline{\boldsymbol z}_{E},
$$

$$
\boldsymbol\sigma_{t+1}
=(1-\eta)\boldsymbol\sigma_t+\eta\,\operatorname{std}(\boldsymbol z_E).
$$

CEM 的代数和种群数固定，所以优化过程有确定的最大工作量。每个候选通过一个持久化
原生 C++/CUDA worker 评分；Python 只负责采样、PCA 解码和进程通信，不参与 score
内部数值计算。

### 3.3 评分链路

原生 score 对每个候选依次执行：

1. 线圈几何与工程约束；
2. 磁轴候选搜索、拓扑筛选和磁轴追踪；
3. 稳定版同类的密集均匀点云 $psi$ 线性 QR 拟合；
4. 多个 $psi$ level 的快速磁面筛选；
5. 磁通标定；
6. 10 万体点上的 Clebsch $\alpha$ 线性最小二乘；
7. 体 QS 统计和总 score 聚合。

这条评分链路没有大参数量 LS/Newton 或 DESC solve。CEM 本身虽然逐代更新，但每次
score 的耗时稳定、失败状态显式，且总评估次数固定。

## 4. CEM 优化结果

![CEM 收敛曲线](assets/native_score_cem_validation/cem_convergence.png)

| 指标 | QH | QA |
| --- | ---: | ---: |
| 随机起点 score | 10.005 | 9.977 |
| 最优 score | **71.325** | **69.822** |
| 最优代数 | 7 | 6 |
| 第 8 代有效样本 | 25/32 | 15/32 |
| 8 代总耗时 | 519.43 s | 504.83 s |
| 平均每次评估 | 2.029 s | 1.972 s |

两组任务各用一张空闲 RTX 5090 并行运行，因此实际等待时间约 9 分钟，而不是两者
相加的 17 分钟。开发阶段的 GPU preflight 确认分配到的 GPU 在启动时利用率为零，
没有其他计算进程污染计时。

与 QUASR 独立 1000 样本基准比较：

| QUASR 指标 | score |
| --- | ---: |
| P10 | 35.33 |
| median | 64.07 |
| P90 | 70.89 |
| max | 78.02 |

QH 最优样本超过 P90，QA 最优样本位于 median 与 P90 之间。QUASR 参考集全链路成功率
为 798/1000；本轮 CEM 的第 8 代有效率分别达到 78.1% 和 46.9%，说明分布已经从大量
`no_axis` 随机样本移动到稳定可评分区域。

QH 最优样本的原生 score 分解为：

| 分量 | 分数 |
| --- | ---: |
| axis | 95.25 |
| psi | 99.84 |
| surface | 83.13 |
| coordinate | 90.20 |
| volume_qs | **3.45** |
| coil | 66.06 |

这张表很重要。总分提升主要来自磁轴、$psi$、磁面存在性和坐标质量；体 QS 分量仍低。
因此总分没有被“没有磁面”的样本欺骗，但存在另一种偏好：优化器可以先把容易改善的
几何与坐标项做到很好，而不必同步得到非常纯的目标 QS。

最优样本的关键原生诊断为：

| 诊断 | 数值 |
| --- | ---: |
| 磁轴 residual | $2.33\times10^{-8}$ |
| $psi$ angle p95 | $9.16\times10^{-6}$ |
| 原生选择面体积 | $0.01226\,\mathrm{m}^3$ |
| 原生选择面 $a/R$ | 0.0725 |
| $alpha$ 法向磁场相对残差 | $5.09\times10^{-6}$ |
| 原生 $\iota$ | 0.2905 |
| 线圈平均长度 | 5.287 m |
| 线圈曲率 p95 / max | 8.85 / 13.82 $\mathrm{m}^{-1}$ |
| 最小线圈间距 | 0.0761 m |
| 最小线圈到轴距离 | 0.3186 m |

## 5. 旧路径独立复核

### 5.1 扫描结果

旧稳定路径重新拟合 $psi$，扫描

$$
a\in\{0.05,0.08,0.12,0.16,0.20\}\,\mathrm{m},
$$

并对每个 $a$ 最多选择 6 个候选进入 Simsopt Boozer LS/Newton。结果如下：

| $a$ [m] | 最大成功 $\psi$ level | 体积 [$\mathrm{m}^3$] | Simsopt QA | Simsopt QH | Simsopt QP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.30 | 0.01020 | 0.683% | 0.690% | 0.048% |
| 0.08 | 0.36 | 0.03183 | 0.706% | 0.810% | 0.152% |
| 0.12 | 0.36 | 0.07524 | 1.048% | **1.250%** | 0.540% |
| 0.16 | 0.30 | **0.12308** | 3.481% | **4.784%** | 3.672% |
| 0.20 | 0.004 | 0.00205 | 0.662% | 0.650% | 0.010% |

![旧路径磁面扫描](assets/native_score_cem_validation/surface_sweep.png)

$a=0.20$ 时 $psi$ 拟合范围过大，筛选退回极小面，不应解释为位形突然变好。
$a=0.12$ 是本次扫描中“体积较大且 QH 仍约 1%”的较好折中；$a=0.16$ 是最大
计算可解面，用于最严格的可视化和 DESC 检查。

各个独立 $a$ 拟合得到的 Simsopt $\iota$ 在 $-0.548$ 到 $+0.451$ 之间跳变，不构成
连续的物理 $\iota(\rho)$ profile。这说明独立 Boozer LS 仍存在分支或角度规范差异。
因此本报告用 Poincare 和 DESC $\iota$ 做交叉验证，不把该列作为最终物理结论。

### 5.2 三维几何和磁场强度

![线圈和最大磁面](assets/native_score_cem_validation/coils_surface.png)

[打开可旋转的三维线圈与最大磁面](assets/native_score_cem_validation/coils_surface.html)

最大面上的外部线圈场为：

$$
|B|_{\min}=0.3050\,\mathrm{T},\qquad
\langle|B|\rangle=0.4558\,\mathrm{T},\qquad
|B|_{\max}=0.6108\,\mathrm{T}.
$$

![最大面 Boozer 坐标下的磁场强度](assets/native_score_cem_validation/boozer_b.png)

[打开 Boozer 磁场强度 HTML](assets/native_score_cem_validation/boozer_b.html)

热力图存在清晰的螺旋斜带，但不是单一完美 QH 模式。这与 Simsopt QH error
`4.78%`、原生 `volume_qs=3.45/100` 的判断一致。

### 5.3 Poincare

![最大面 Poincare 复核](assets/native_score_cem_validation/poincare.png)

16 条初始线在四个环向截面上分别得到约 223--259 个交点。内层到外层形成连续闭合
曲线，没有出现“score 很高但根本不存在磁面”的作弊样本。最外层点相对黑色候选边界
有少量外溢，所以 $a=0.16$ 面更适合作为偏激进的最大面诊断，而不是保守生产边界。

## 6. DESC 补充验证

DESC 使用最大 Boozer 面作为固定边界，参数为：

| 参数 | 数值 |
| --- | ---: |
| `NFP` | 3 |
| 边界谱阶数 | $M=N=6$ |
| 平衡分辨率 | $L=M=N=8$ |
| 环向磁通 | $-0.0153735\,\mathrm{Wb}$ |
| pressure/current | 0 / 0 |
| 最大迭代 | 50 |

输入曲面为左手坐标，DESC 自动翻转极向角。初态和末态均通过 nested 检查。

| DESC 指标 | 初态 | 末态 |
| --- | ---: | ---: |
| normalized force mean | 1.034 | $2.04\times10^{-4}$ |
| normalized force p95 | 2.438 | $5.75\times10^{-4}$ |
| normalized force max | 6.812 | $3.89\times10^{-3}$ |

优化器运行 50 代、56 次函数评估，最终 cost 为 $9.95\times10^{-9}$，optimality 为
$7.75\times10^{-7}$。停止原因是达到迭代上限，而不是 NaN、非嵌套或坐标折叠。

![DESC 边界](assets/native_score_cem_validation/desc/boundary.png)

![DESC Boozer 磁场](assets/native_score_cem_validation/desc/boozer_B.png)

![DESC Boozer modes](assets/native_score_cem_validation/desc/boozer_modes.png)

![DESC QH error](assets/native_score_cem_validation/desc/qs_QH.png)

![DESC iota](assets/native_score_cem_validation/desc/iota.png)

DESC 的 $\iota$ 从轴附近约 0.29 平滑增加到边界约 0.38，比独立磁面 LS 给出的跳变值
更适合作为整体 profile 参考。DESC QH 指标随半径变差，边界并非高精度 QH，这再次
说明本轮高总分主要证明“磁面和总体质量好”，不能证明目标特异性已经充分优化。

## 7. 耗时

### 7.1 原生 score

QH 最优样本的一次原生评分耗时 `2.150 s`：

| 阶段 | 时间 [s] |
| --- | ---: |
| 磁轴搜索 | 0.895 |
| 磁轴追踪 | 0.019 |
| $psi$ 点生成 | 0.031 |
| $psi$ QR 拟合 | 0.220 |
| $psi$ 验证 | 0.171 |
| 磁面筛选 | 0.746 |
| 磁通标定 | 0.003 |
| $\alpha$ 矩阵装配 | 0.002 |
| $\alpha$ QR 求解 | 0.055 |
| QS 统计 | $4.36\times10^{-4}$ |

当前小型 CEM 脚本在单 GPU 上顺序评分 32 个候选，所以均摊约 2 秒/样本。先前 1000
样本正式评测使用 4 张 RTX 5090 和每卡最优并行设置，端到端吞吐为
`0.879 s/sample`；本轮小实验没有为吞吐重写 CEM 调度器。

### 7.2 旧路径和 DESC

| 任务 | 时间 |
| --- | ---: |
| 五组 $a$ 的稳定旧路径数值计算总和 | 18.14 s |
| 单组 $a$ 总时间范围 | 2.36--5.20 s |
| Poincare 16 条线 | 约 9 s |
| DESC initial force | 6.34 s |
| DESC solve | 141.84 s |
| DESC 最终图 | 约 38.3 s |
| 复用扫描后的最终绘图 + DESC 作业 | 232.62 s |

DESC 0.16 使用独立 CPU JAX 环境，未使用 RTX 5090。它是离线补充验证，不属于目标
的原生快速 score 路径，也不应计入线圈到 score 的生产耗时。

## 8. 开发中修正的问题

1. 完整评估最初依赖环境中未声明的 Plotly，导致旧路径扫描成功后在绘图阶段退出。
   现已改为静态 Matplotlib PNG，以及数据内嵌的 Three.js 交互 HTML；Python 端不再
   依赖 Plotly。
2. 评估脚本现在会复用已经完成的 `a` 扫描，绘图或 DESC 失败后无需重新计算旧路径。
3. 新服务器的稳定版虚拟环境没有 DESC，而且 JAX/Scipy 版本与 DESC 0.16 不兼容。
   已建立独立环境，通过 SHA-256 核对的离线 wheelhouse 安装，不修改稳定 score 环境。
4. Slurm `sbatch --test-only` 曾错误预估 P107 要等待一天；真实探针和后续 P107 作业
   均在提交当秒启动。后续以真实 pending reason 为准，不再依据该失真预估等待。

## 9. 最终判断与下一步

### 已经证明

- 当前 score 具有可优化梯度，不只是 QUASR 样本上的离线排序指标；
- CEM 能从几乎全失败的随机邻域移动到大多数样本可稳定评分的区域；
- 随机优化得到的高分样本有真实嵌套磁面，而非依靠固定小圆柱或无磁面区域作弊；
- 中等大面 QS error 约 1%，最大面约 3%--5%；
- 最大边界可以得到 nested、低 force residual 的 DESC equilibrium。

### 尚未证明

- 单次 QA/QH 随机起点不足以证明大范围优化鲁棒性；
- QH 目标没有表现出很强的 QA/QH/QP 分离，不能据此宣称得到高纯度 QH；
- 最大面独立 Boozer $\iota$ 存在分支跳变，严格的连续面分支追踪仍需改进；
- DESC 达到最大迭代数，虽然结果低残差且 nested，但没有满足优化器正式成功条件。

### 最有价值的后续改进

1. 提高大体积 `volume_qs` 在 65--80 分区间的影响，避免 axis/psi/coordinate 高分掩盖
   目标 QS 较弱；
2. 在 score 中加入目标 helicity 相对非目标 helicity 的 margin，而不只看绝对 QS；
3. 用连续 $\psi$ level continuation 约束旧 Boozer 分支，消除独立 $a$ 扫描的 $\iota$
   跳变；
4. 将 CEM population 分发到 4 GPU、每卡复用已验证的最优并行度，缩短优化墙钟时间。

本轮目标是验证有效性而非全面扫参。以这个标准看，实验已经成功；更严格的结论应在
改进目标特异性后，用多个随机起点重复。

## 10. 产物

- [QH CEM 原始汇总](assets/native_score_cem_validation/raw/cem_qh_summary.json)
- [QA CEM 原始汇总](assets/native_score_cem_validation/raw/cem_qa_summary.json)
- [全面评估汇总](assets/native_score_cem_validation/raw/full_summary.json)
- [QUASR 1000 样本参考](assets/native_score_cem_validation/raw/quasr_1000_reference.json)
- [DESC 输入](assets/native_score_cem_validation/desc/input.check)
- [DESC equilibrium](assets/native_score_cem_validation/desc/equilibrium.h5)
- [DESC 原始汇总](assets/native_score_cem_validation/desc/summary.json)

## 11. 后续修正：三周期 HTML 拓扑

原交互 HTML 的磁面只含一个场周期。绘图代码先把这个周期首尾闭合，再复制旋转三份，
所以每个周期都被错误地封成独立曲面；视觉上表现为每一周期的首尾被强行连接，而不是
三个周期依次连接成完整装置。

现实现先在 Python 端把一个周期旋转拼接为完整三周期顶点，再对完整环向网格只做一次
周期闭合。实际产物包含 55296 个顶点和 110592 个三角形。索引级回归检查结果为：

- 三个相邻场周期的交界边全部存在；
- 三个旧的“单周期首尾封口边”全部不存在；
- 第三周期末端只与第一周期起点连接；
- 无头浏览器实际渲染非空，显示为一张连续的三周期闭合曲面。

[打开修正后的线圈与磁面 HTML](assets/native_score_cem_validation/coils_surface.html)

## 12. 微分 QS 分量的缩放复核

### 12.1 必须修正的 helicity 尺度

当前微分目标满足

$$
f_C(kM,kN)=k f_C(M,N).
$$

因此直接用同一软阈值比较 QA 的 $(M,N)=(1,0)$ 和 QH 的 $(1,N_{\mathrm{fp}})$，会仅因
helicity 向量更长而系统性压低 QH。现评分保持原始诊断量不变，只把软阈值改为

$$
h=\sqrt{M^2+N^2},\qquad
s_{\mathrm{global}}=0.05h,\qquad
s_{\mathrm{edge}}=0.07h.
$$

1000 个 QUASR 验证样本中有 798 个完整成功。修正前后的 `volume_qs` 分量如下：

| 模式 | 样本数 | 修正前 P10 / 中位数 / P90 | 修正后 P10 / 中位数 / P90 |
| --- | ---: | ---: | ---: |
| QA | 441 | 7.78 / 25.84 / 39.08 | 7.78 / 25.84 / 39.08 |
| QH | 357 | 4.81 / 9.05 / 17.72 | 15.53 / 22.37 / 29.25 |

QA 不受影响；QH 不再因为 $N_{\mathrm{fp}}$ 的纯代数尺度被额外惩罚。旧 CEM 保存的 QH
候选中，QS 分量的中位数从 1.28 重标为 3.49，最大值从 5.40 重标为 13.60；原全面
评估候选用新二进制重算后为 9.12。这仍明显低于 QUASR QH 中位数 22.37，说明 CEM
候选确实不够 QH，而不是仅仅被错误尺度压低。

![QS 分量缩放审计](assets/native_score_cem_validation/qs_score_scale_audit.png)

### 12.2 是否还应继续放宽阈值

结论是暂不继续放宽。微分 QS 与 QUASR 元数据中的旧面 QS 不是同一归一化：两者比值的
$\log_{10}$ 中位数为 4.65，不能共用数值阈值。当前 helicity 归一化后，QUASR 好样本和
CEM 差样本已有明确梯度；继续增大阈值只会把真实较差的 CEM 样本抬高，削弱区分度。

当前 CEM 总分约 72 而 QS 分量只有 9--14，暴露的是总目标中其他高分项可以补偿 QS，
不是 QS 软阈值过严。若下一轮优化仍忽略 QS，合理改法是提高该分量的总权重、设置最低
QS 门槛，或加入目标 helicity 相对非目标 helicity 的 margin，而不是继续放大软阈值。

[缩放审计原始 JSON](assets/native_score_cem_validation/qs_score_scale_audit.json)

## 13. 微分 QS 是否会被形状或点数“作弊”

### 13.1 先区分原始量和最终分量

原始 `qs_global_error` 是在选定物理体积上计算的微分 QS RMS。最终 `volume_qs` 分量为

$$
S_{\mathrm{volume\_qs}}
=100\,S_{\mathrm{residual}}
\left(0.35+0.65S_{\mathrm{size}}\right).
$$

所以最终分量本来就不是“纯 QS”，而是“较大有效体积上的 QS”。磁面尺寸影响最终分量
是设计目标，不是采样泄漏。需要防止的是点数变化、无效点丢弃或错误体积权重影响
$S_{\mathrm{residual}}$。

### 13.2 找到并修正的体积权重 bug

采样在每条星形射线上对 $u=\rho^2$、$\theta$、$\phi$ 均匀。令边界半径为
$r_b(\theta,\phi)$，柱坐标体积元给出

$$
dV=R\,r\,dr\,d\theta\,d\phi
=\frac{1}{2}R\,r_b^2\,du\,d\theta\,d\phi.
$$

因此每点物理体积权重应满足

$$
w_i\propto R_i r_{b,i}^2.
$$

旧实现只用了 $R_i$，漏掉 $r_{b,i}^2$，使不同形状磁面的隐式角向权重不正确。该 bug
已经在 C++/CUDA 和 CPU 参考实现中同时修正。修正前后 798 个共同成功样本的原始 QS
秩相关为 0.99915；相对变化中位数为 0.27%，P10/P90 为 $-7.12\%/+5.80\%$。这说明
旧结果的大体排序没有崩坏，但新结果才是严格的物理体积平均。

### 13.3 点数与有效点审计

定义加权有效样本比例

$$
\eta_{\mathrm{ESS}}
=\frac{\left(\sum_i w_i\right)^2}{N\sum_i w_i^2}.
$$

则有效点数为 $N_{\mathrm{eff}}=N\eta_{\mathrm{ESS}}$。修正后的 1000 样本结果为：

| 指标 | 最小值 | P10 | 中位数 | P90 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 实际 QS 点数 | 100000 | 100000 | 100000 | 100000 | 100000 |
| 候选有效比例 | 0.99870 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| $\eta_{\mathrm{ESS}}$ | 0.1085 | 0.3578 | 0.6114 | 0.8036 | 0.9423 |
| $N_{\mathrm{eff}}$ | 10847 | 35783 | 61140 | 80363 | 94230 |

所有成功样本都使用相同的 10 万点，因此不存在“少算点反而得高分”的通道。总分前 20
的有效比例全部为 1，最低 $\eta_{\mathrm{ESS}}=0.729$；QS 分量前 20 的有效比例也全部
为 1，最低 $\eta_{\mathrm{ESS}}=0.596$。高分样本没有依赖权重集中。

代码进一步收紧为：只有可用点足以填满固定预算，且候选有效比例至少 95% 时才允许
进入 QS 评分。这样优化器不能通过主动制造无效区域来丢弃最多 40% 的坏点。ESS 保留为
诊断量而不直接扣分，因为它描述物理体积权重的集中程度，盲目惩罚会把合法磁面形状
当作作弊。

### 13.4 最终分量主要由什么决定

在 QA、QH 内分别统计后：

| 相关量 | QA Spearman | QH Spearman |
| --- | ---: | ---: |
| `volume_qs` 与 $-\log_{10}$ 原始微分 QS | 0.956 | 0.937 |
| `volume_qs` 与残差软分数 | 0.956 | 0.939 |
| `volume_qs` 与磁面尺寸因子 | 0.223 | -0.490 |
| `volume_qs` 与 $\eta_{\mathrm{ESS}}$ | 0.153 | -0.557 |
| 原始微分 QS 与 QUASR 旧面 QS | 0.874 | 0.531 |

因此在各自模式内，最终分量首先由微分 QS 残差决定，不是由点数决定。QH 中 ESS 与
分量存在中等相关，但这是数据集中磁面形状、场梯度和真实 QS 品质共同变化造成的协方差；
ESS 不直接进入评分公式。更重要的是，最低 ESS 样本的加密复核没有显示数值漂移。

最后一行也给出必要限制：这个微分体积指标与旧的单面 Boozer/Fourier QS 有一致排序，
但并不等价，尤其 QH 只有中等相关。原因包括体积平均与单面指标的目标不同、选取磁面
不同，以及当前 $\alpha/\iota$ 仍是有限阶拟合。因此可以说它主要反映“当前定义下的
体微分 QS”，不能把它直接宣称为旧 `qs_error` 的高精度替代值。

![QS 采样与混杂因素审计](assets/native_score_cem_validation/qs_sampling_audit.png)

### 13.5 5 万、10 万、20 万点收敛复核

选择 QA/QH 中最低 ESS 样本和最高 QS 分量样本各一个，共 4 个极端样本，在同一原生
C++/CUDA 黑箱上分别使用 5 万、10 万和 20 万点。12 次评估全部成功。

以 20 万点为参考，10 万点的原始 QS 最大相对变化为 1.38%，QS 分量最大绝对变化仅
0.114 分。两个最低 ESS 样本的原始 QS 相对变化分别为 0.020% 和 0.020%。即使最坏的
1.38% 出现在非常低误差的高分 QA 样本上，也只改变最终 QS 分量 0.22%。默认 10 万点
已经足够稳定，没有必要把生产成本翻倍。

![QS 点数收敛](assets/native_score_cem_validation/qs_sampling_convergence.png)

[采样审计原始 JSON](assets/native_score_cem_validation/qs_sampling_audit.json)

## 14. 修正后 1000 样本速度与最终结论

修正后的完整验证仍有 798/1000 个样本成功，与修正前完全一致。4 张空闲 RTX 5090
并行时总墙钟为 882.43 秒，均摊 0.882 秒/样本。单样本内部计时均值为 3.42 秒，最大
7.75 秒，100% 小于 10 秒。开始预检时四张卡均为 0% 利用率、2 MiB 显存。

最终判断如下：

1. 原始微分 QS 确实主要反映选定物理体积上的微分 QS 误差，但不是旧单面 QS 数值的
   同义词；QH 上需要保留这项口径差异。
2. 原实现存在真实的形状权重 bug，现已用 $Rr_b^2$ 修正；旧排序大致有效，但新结果
   才有正确物理体积含义。
3. 当前成功评分固定使用 10 万点，并新增 95% 有效率门槛，点数与无效区域作弊通道
   已被关闭。
4. helicity 归一化是必要修正；当前不应继续放宽 QS 软阈值。若优化仍产出“总分高但
   QS 一般”的样本，应修改分量权重或目标特异性，而不是把差 QS 样本的分数整体抬高。
5. 修正后的 HTML、统计脚本、CPU/CUDA 实现和单测均已更新。最终原生库 smoke 返回
   `ok`，使用 100000 点且有效率为 1。

- [修正后 1000 样本汇总](assets/native_score_cem_validation/corrected_validation/summary.json)
- [修正后逐样本数据](assets/native_score_cem_validation/corrected_validation/rows.csv)
