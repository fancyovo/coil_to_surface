# Local Surface Evaluator 项目进度回顾

更新日期：2026-07-26  
当前分支：`desc-psi-volume-initial-guess`  
当前提交：`4fde178 Add toroidal Boozer coordinate correction`  
稳定版基线：`main` / `c5ac4d8`

> 这份报告用于中断一段时间后快速恢复上下文。它强调结论、因果关系和准确停点；
> 详细推导与完整数据仍以文末链接的专题报告和 JSON 为准。

## 1. 一分钟版本

稳定主流程已经可以从**线圈**出发，自动找到磁轴、拟合局部磁面标签 $s$、筛选可信
磁面、用 Simsopt 做 Boozer/QS 评估，并给出质量分数。`cem_1` 和 `cem_3` 都验证过
这条完整评估链路，也都曾用传统的“只给外边界”方式跑通 DESC。

当前分支做的是更难的一步：

> 对 `cem_qh03`，评估器能找到良好的嵌套磁面，但交给 DESC 后求解会发散。
> 我们希望用线圈场和体内磁面信息构造一个接近真实真空平衡的 DESC **体坐标初值**，
> 而不只给 DESC 一张外边界让它自己向轴缩放。

目前已经做到：

1. 证明 `cem_qh03` 的拟合 $s$ 面本身良好嵌套。早期 R/Z 不嵌套主要来自代码 bug
   和角坐标标签混用，不是磁岛或线圈位形从根本上无解。
2. 将无量纲 $s$ 标定为物理环向磁通 $\psi_T$。
3. 用 12 万训练点、6 万独立验证点和一次 GPU FP64 QR，联合求出
   $\alpha$、$\lambda$ 和 $\iota$；磁力线直线度提高约 18.9 倍。
4. 在直场线角下线性拟合 DESC 的 R/Z 体谱，测试的 `LMN=6,8,10,12` 都保持嵌套。
5. 发现“直场线坐标”还不是完整 Boozer 坐标，又用一次线性 Fourier 求解补上环向
   坐标修正 $\nu$。固定磁面上的完整 Simsopt Boozer residual 在中间磁面降到
   $3.5\times10^{-4}$ 至 $1.1\times10^{-3}$，边界为 $3.31\times10^{-3}$。

目前还没有做到：

1. 还没有把最新的 $\alpha+\nu$ 坐标完整、无歧义地写入 DESC 体初值。
2. 还没有在同一批物理点上闭合比较
   $\boldsymbol B_{\rm coil}$、$\boldsymbol B_{\alpha+\nu}$ 和
   $\boldsymbol B_{\rm DESC,initial}$。
3. DESC solve 仍未跑通。上一版只使用 α 直场线角的 R/Z 初值时，初始体嵌套，
   但 solve 后 Jacobian 变号，力残差暴涨。这是明确的物理失败。

因此当前准确停点是：

> **目标磁面分支上的近 Boozer 坐标已经基本构造出来；下一步是完成 DESC 坐标接口和
> 初始磁场闭合诊断，而不是继续盲目调 DESC 求解器。**

---

## 2. 项目其实有两条线

### 2.1 稳定版：快速评估器

稳定版回答：**给定一组线圈，它附近有没有值得继续研究的嵌套磁面，质量如何？**

```text
线圈 Fourier 系数、电流、nfp
        |
        v
Biot-Savart 磁场
        |
        v
GPU 搜索周期磁轴 + 椭圆/双曲拓扑判别
        |
        v
稠密线性 LS 拟合局部不变量 s，使 B·grad(s) 约等于 0
        |
        v
候选 s level 的一周期场线漂移筛选 + iota 初估
        |
        v
GPU 提取等值面并投影为 Simsopt 曲面
        |
        v
Simsopt Boozer LS/Newton + iota/G/volume/QS
        |
        v
质量分数、失败原因、诊断图和结构化结果
```

这条链路已经相对成熟，典型 GPU 总耗时约为数秒。它的定位是快速筛选和诊断，
**不是**高精度 MHD 平衡求解器。

### 2.2 当前分支：DESC 体初值研究

当前分支回答：**已经知道线圈真空场附近有一族磁面，怎样把这族磁面和正确的场线坐标
交给 DESC，让 DESC 从接近真实解的位置开始？**

DESC 固定边界求解并不是“拟合外部线圈磁场”。给定边界、总环向磁通、压力和电流
profile 后，DESC 求的是一套自洽的理想 MHD 平衡。外部 Biot-Savart 场只提供目标几何
和构造初值的信息，不会自动成为 DESC 内部的磁场。

所以，一张几何上很好的边界，甚至一套嵌套的 R/Z 初值，仍然不保证 DESC solve
会收敛到正确平衡。

---

## 3. 三个例子分别说明什么

| 例子 | 已知结论 | 在当前研究中的角色 |
|---|---|---|
| `cem_1` | 稳定评估器分数约 92.92；传统边界方式 DESC 可完成，但最大面外层 Poincare 鲁棒性一般 | 可跑通参考例，但不是最干净的回归基准 |
| `cem_3` | 稳定评估器分数约 92.41；传统边界方式 DESC 可完成，最大归一化 force error 约 $3.38\times10^{-5}$ | 最可信的成功回归基准 |
| `cem_qh03` | 能找到局部嵌套磁面，但旧 Simsopt 面发生分支跳转，直接 DESC 路径发散 | 当前分支的困难样本 |

重要边界：`cem_1`、`cem_3` 的 DESC 成功使用的是稳定版传统
“Boozer 外边界 $\rightarrow$ DESC 默认体初值”路径。它们**不能证明**当前新增的
“$s$ 体面 $\rightarrow\alpha\rightarrow\nu\rightarrow$ DESC 体初值”路径已通过回归。

早期新体初值烟测中，`cem_1` 和 `cem_3` 在默认外层都出现过小比例的射线层反序或
Jacobian 少数符号错误。因此最终新方法仍必须重新回归这两个例子。

另一个容易混淆的点是边界选择：

- 早期 `cem_qh03` 的 $s$/RZ 嵌套诊断使用较外层的 $s_{\rm edge}=0.3$；
- 后续正式 α、ν 实验使用更保守的 $s_{\rm edge}=0.16$。

两组数字描述不同边界，不能直接拼成同一个实验。

---

## 4. 我们是怎样一步步走到现在的

### 阶段 A：最初尝试给 DESC 多层 R/Z 点云

最初从拟合 $s$ 提取多层等值面，得到

$$
(\rho,\theta,\zeta)\mapsto(R,Z),
$$

再把这些点交给 DESC 的 `set_initial_guess`，第一版取 $\lambda=0$。

这一步发现：只把“物理点”给 DESC 不够。点属于哪个计算角 $\theta$，以及边界和内部
是否使用同一套角标签，会直接决定谱体映射是否折叠。

### 阶段 B：外部联合 R/Z/L/$\iota$ 最小二乘与代码审计

随后实现外部线性最小二乘，直接控制 R、Z、L、每条场线截距 $\beta$ 和
$\iota(\rho)$。这轮最有价值的结果是找出了多个确定性 bug：

| 问题 | 后果 | 修正结果 |
|---|---|---|
| `A_beta` 使用 NumPy 高级索引选成子矩阵 | 每个 phase 样本错误连接多条场线 | phase RMS 从约 1.570 降到 0.377 |
| RK4 的 Z 更新漏掉 `+k4z` | Poincare 和 trace 数据被污染 | 修正并加入解析测试 |
| 轴位置与轴导数使用不一致插值 | $\nabla s$ 与实际拟合模型不一致 | 改为周期 cubic Hermite，CPU/GPU 同步 |
| `lambda_weight` 传错接口 | phase 权重未真正生效 | 修正参数传递 |
| 内层几何角与 Boozer 边界原生角混用 | R/Z 体坐标扭曲、折叠 | 统一使用 $s$-ray 参数化 |
| Simsopt/VMEC/DESC 往返后边界标签改变 | fixed boundary 与体末层同物异参 | 从拟合体的 $\rho=1$ 重取 DESC 边界 |

修正后，对 $s_{\rm edge}=0.3$ 的 `cem_qh03` 得到：

- 13 个径向层严格按射线嵌套；
- 最小相邻半径间隔为 $2.98\times10^{-3}\,\mathrm m$；
- 最小 $\partial s/\partial r=0.614>0$；
- 修正后的 Poincare 点贴合 $s$ 等值线；
- 没有证据支持“磁岛使 R/Z 根本不存在”。

![修正后嵌套的 s 等值面](assets/desc_rz_nesting_cem_qh03/psi_nested_sections.png)

统一角坐标后，R/Z 在 4 阶和 6 阶都可严格嵌套。旧的 L/$\iota$ phase 拟合也能通过
较强 ridge 保持坐标不折叠，但短场线点云路线仍然病态，而且没有解决 DESC 自身磁场
与线圈磁场不一致的问题。

### 阶段 C：改走稠密 Clebsch $\alpha$ 线性拟合

我们随后采用更物理、也更适合 GPU 的路线。先把 $s$ 标定成物理环向磁通
$\psi_T(s)$，再定义

$$
\rho=\sqrt{\frac{\psi_T}{\psi_{T,\rm edge}}},
$$

以及 Clebsch 场线标签

$$
\alpha=\theta+\lambda(\rho,\theta,\phi)-\iota(\rho)\phi,
$$

目标为

$$
\boldsymbol B\simeq\nabla\psi_T\times\nabla\alpha.
$$

$\lambda$ 使用 Fourier-Zernike 展开。径向函数 $\mathcal R_l^m(\rho)$ 是标准 Zernike
径向多项式，**不是**柱坐标大半径 $R$。

由于 $\rho$ 是 $\psi_T$ 的函数，径向导数项在叉乘中消失，所有系数仍线性进入方程。
因此可以像拟合 $s$ 一样，用十万级均匀点云和一次 QR 求解，不需要非线性初值。

正式实验配置与结果：

| 项目 | 配置或结果 |
|---|---:|
| 训练点 / 独立验证点 | 120,000 / 60,000 |
| 求解器 | 单卡 FP64 GPU QR |
| 最高模型 | $L=12,M=12,N=16$，3001 个未知量 |
| 非线性优化 / 约束 / ridge | 均未使用 |
| 边界总环向磁通幅值 | $1.385027\times10^{-3}\,\mathrm{Wb}$ |
| 磁通标定相对 RMS | $1.41\times10^{-8}$ |
| 三维磁场重构验证 residual | $5.79\%$ |
| 坐标可逆性最小值 | $\min(1+\partial_\theta\lambda)=0.382$ |
| α 拟合的 $\iota$ | $-0.565228$ |
| 32 周期真实场线的 $\iota$ | $-0.565327\pm0.000261$ |

5.79% 的残差主要集中在 $\rho<0.2$，不是由
$\boldsymbol B\cdot\nabla\psi_T$ 的法向误差主导。α 的 $\iota$ 与独立长轨迹符合到约
$10^{-4}$；旧 Boozer 文件的 $\iota=-0.486970$ 不能继续作为真值。

![alpha 修正前后的磁力线直线度](alpha_clebsch_experiment/fieldline_straightening.png)

八个场周期、三层磁面、每层四条线的平均直线 residual 从约
$0.218\,\mathrm{rad}$ 降到 $0.01154\,\mathrm{rad}$，改善 18.9 倍。

### 阶段 D：把直场线坐标投影到 DESC R/Z 体谱

令

$$
\vartheta=\theta+\lambda,
$$

然后拟合

$$
(\rho,\vartheta,\phi)\mapsto(R,Z).
$$

R 和 Z 仍分别用 GPU QR 线性投影到 DESC Fourier-Zernike 基底：

| DESC 分辨率 | R RMS | Z RMS | 边界 RMS | 初始是否嵌套 | 初始力 p95 |
|---|---:|---:|---:|---|---:|
| `LMN=6` | 0.163 mm | 0.187 mm | 0.252 mm | 是 | 2.115 |
| `LMN=8` | 0.152 mm | 0.176 mm | 0.215 mm | 是 | 3.001 |
| `LMN=10` | 0.137 mm | 0.155 mm | 0.203 mm | 是 | 2.605 |
| `LMN=12` | 0.133 mm | 0.149 mm | 0.197 mm | 是 | 2.839 |

几何高阶更准，但 DESC 初始力并非最低。这说明几何点位 RMS 不能单独作为 DESC 初值
选阶标准，高阶导数误差会进入力平衡 residual。

![直场线坐标下的 RZ 体谱投影](alpha_clebsch_experiment/rz_straight_fit_sections.png)

对初始力最低的 `LMN=6` 直接运行 DESC solve，得到：

| 指标 | solve 前 | solve 后 |
|---|---:|---:|
| 嵌套 | 是 | 否 |
| 归一化力均值 | 0.754 | $7.85\times10^5$ |
| 归一化力 p95 | 2.115 | 552.0 |
| 归一化力最大值 | 12.59 | $2.94\times10^9$ |
| objective | 约 $2.49\times10^3$ | $1.00\times10^{14}$ |

优化器因 `xtol` 报告 success，但解已不嵌套且 residual 暴涨，所以必须判为失败。

### 阶段 E：发现还缺环向 Boozer 坐标

α 让磁力线变直，只保证方向条件。完整 Boozer 条件还要求沿场参数速度正确：

$$
G\boldsymbol B
=B^2\left(\boldsymbol x_\phi+\iota\boldsymbol x_\vartheta\right),
$$

并且同一磁面上的 $G$ 为常数。

冻结目标 $s$ 磁面、完全不优化几何时，α 后的场线方向误差通常只有
$0.04^\circ$ 到 $0.1^\circ$，但完整 Simsopt Boozer residual 仍约为 14%。诊断发现
它几乎完全来自

$$
G_{\rm local}=\boldsymbol B\cdot
\left(\boldsymbol x_\phi+\iota\boldsymbol x_\vartheta\right)
$$

在磁面上的约 14% 起伏，而不是磁面或场线方向仍然很差。

这轮还发现旧 `boozer_surface.npz` 虽然 residual 很小，但把它的物理点代回当前
$s(R,Z,\phi)$ 后只有 $\langle s\rangle=0.03234$，而不是目标 $s=0.16$。平均法向距离
约 17.1 mm。旧 Simsopt 固定体积优化跳到了另一条几何分支；它的低 residual 和错误
$\iota$ 不能作为目标磁面的参考。

### 阶段 F：用线性方程补上环向修正 $\nu$

引入周期修正

$$
\phi_B=\phi+\nu,
\qquad
\theta_B=\vartheta+\iota\nu.
$$

它保持

$$
\theta_B-\iota\phi_B
=\vartheta-\iota\phi
=\alpha,
$$

所以不会破坏已经得到的直场线标签。令

$$
D=\partial_\phi+\iota\partial_\vartheta,
$$

只需在线性 Fourier 空间求解

$$
D\nu=\frac{G_{\rm local}}{G}-1.
$$

这仍然是稳定线性问题，不需要移动磁面或调用 Simsopt 非线性曲面优化。

![环向修正前后的完整 Boozer residual](alpha_clebsch_experiment/toroidal_correction_vs_rho.png)

12 阶 $\nu$ 的关键结果：

| $\rho$ | 仅 α residual | $\alpha+\nu$ residual | 改善倍数 |
|---:|---:|---:|---:|
| 0.12 | 0.14093 | 0.001100 | 128 |
| 0.20 | 0.14098 | 0.000999 | 141 |
| 0.30 | 0.14110 | 0.000698 | 202 |
| 0.50 | 0.14149 | 0.000473 | 299 |
| 0.80 | 0.14240 | 0.000352 | 404 |
| 1.00 | 0.14326 | 0.003311 | 43 |

映射满足

$$
0.7435\le 1+D\nu\le1.2054,
$$

没有接近折叠；$|\nu|$ 最大约 $4.00^\circ$ 到 $5.02^\circ$。对 $\rho\le0.9$，
$G_{\rm local}$ 相对起伏已降到约 $1.7\times10^{-5}$ 至 $4.6\times10^{-5}$。

**这就是当前最新结果。**

---

## 5. 当前状态表

| 层次 | 状态 | 已有证据 | 剩余问题 |
|---|---|---|---|
| 线圈场和磁轴 | 已完成 | 稳定版主链路，多例验证 | 无当前阻塞 |
| 局部不变量 $s$ | 已完成 | $\boldsymbol B\cdot\nabla s$ 很小；qh03 层严格嵌套 | 它不是物理磁通，需标定 |
| $s\rightarrow\psi_T$ 标定 | 已完成 | 多截面一致；相对拟合 RMS $1.41\times10^{-8}$ | 外层仍是近似磁面 |
| $\alpha,\lambda,\iota$ | 基本完成 | 稠密 QR；长场线独立验证；无坐标折叠 | 近轴三维重构残差较高，边界有方向误差尖峰 |
| 环向修正 $\nu$ | 已完成 | 固定磁面 Boozer residual 降低 43-404 倍 | 尚未接入 DESC 表示 |
| R/Z 体谱 | 已完成一版 | 4 个 LMN 档都嵌套，误差约 0.13-0.19 mm | 当前版本使用 α 角，还没纳入最终 $\nu$ |
| DESC 初始磁场闭合 | 未完成 | 已知仅几何好不够 | 要比较 coil / α+$\nu$ / DESC initial 三种场 |
| DESC solve | 未完成 | 上次 solve 明确发散 | 必须先完成上一步，再做受限 continuation |
| 新路线回归 | 未完成 | `cem_1/cem_3` 仅传统路径成功 | 新体初值方法需重新验证 |

一句话概括当前成熟度：

> **几何磁面和 Boozer 类坐标已经达到“值得接入 DESC”的精度，但 DESC 自身变量、
> profile 与该坐标之间的物理闭合还没有验证。**

---

## 6. 现在离 DESC 真正想要的解还有什么差别

我们现在构造的是线圈真空场的近似磁面坐标：

$$
(\rho,\theta_B,\phi_B)\mapsto(R,Z),
$$

并且在固定磁面上已经近似满足 Boozer 条件。

DESC 真正求的是：在指定固定边界、总磁通、压力和电流 profile 下，一组满足理想 MHD
平衡且 Jacobian 全域合法的谱系数：

$$
\boldsymbol J\times\boldsymbol B=\nabla p.
$$

当前例子取 $p=0$、等离子体环向电流为零。若线圈真空场确有精确嵌套磁面，理论上它应
对应一个真空平衡；但数值接口上还有三层差别：

1. **坐标表示差别。** 我们显式有环向修正 $\nu$；DESC 侧必须确认它应通过 toroidal
   coordinate、`omega` 或等价 R/Z 重参数化进入，不能默认 $\phi_B=\phi$。
2. **磁场构造差别。** DESC 根据边界、$\Psi$、current/iota profile 和谱几何重建自己的
   $\boldsymbol B$；它不会直接采用 $\boldsymbol B_{\rm coil}$。
3. **有限谱与近似磁面差别。** $s$、α、R/Z 都有截断误差，近轴和最外层最明显；
   高阶虽然减小点位误差，也可能放大导数与力 residual。

所以“离解有多远”不能只用 R/Z 的 0.15 mm RMS 回答。真正需要的下一项指标是同一点上

$$
\frac{\|\boldsymbol B_{\rm DESC,initial}-\boldsymbol B_{\rm coil}\|}
{\|\boldsymbol B_{\rm coil}\|},
$$

并按 $\rho$ 分层报告方向、幅值和 force residual。这个诊断尚未完成，正是当前最重要的
缺口。

---

## 7. 下一步应该怎样继续

### P0：完成 $\alpha+\nu$ 到 DESC 的坐标映射

先审计 DESC 0.16 实际运行版本中 toroidal coordinate / `omega` / RZ 参数化接口，明确
$\nu$ 应写到哪里。然后对同一个物理点族拟合最终

$$
(\rho,\theta_B,\phi_B)\mapsto(R,Z).
$$

验收门槛：边界与体末层同物同参；R/Z 投影与导数误差可控；$\sqrt g$ 和
$\sqrt g_{\rm PEST}$ 全域单符号；坐标逆映射无折叠；暂不调用 DESC solve。

难度：中等。数学关系已经明确，主要风险是 DESC 坐标约定和版本接口。

### P1：进入 solve 前闭合三种磁场

在一套错位独立网格上比较：

$$
\boldsymbol B_{\rm coil},
\qquad
\boldsymbol B_{\alpha+\nu},
\qquad
\boldsymbol B_{\rm DESC,initial}.
$$

按径向层报告三分量相对 L2、方向夹角分位数、$B$ 幅值误差、
$\boldsymbol B\cdot\nabla\rho$、DESC force 分位数和 Jacobian 最小安全裕度。

若前两者接近而 DESC initial 明显偏离，问题就在 DESC 的 profile/接口映射，而不是继续
提高 α 或 R/Z 阶数。

难度：中等，是最有诊断价值的一步。

### P2：通过 P0/P1 后才重新求解 DESC

从低阶开始，使用限步长或逐级 continuation。每一步拒绝 objective 大幅上升或 Jacobian
安全裕度下降，再逐步提高分辨率。同时检查 optimizer 状态、嵌套性和 force 分位数；
`xtol success` 不能再作为成功标准。

难度：中到高。前两步若不闭合，调求解器没有意义。

### P3：按诊断结果处理剩余拟合误差

若误差主要来自 $\rho<0.2$，继续保持线性 QR，只对近轴分层补点、提高径向权重或改进
轴正则基底。若主要来自 $\rho=1$，优先改进边界附近的 α 和 R/Z 谱投影。当前没有坐标
折叠证据，不应优先切换到复杂非线性或带不等式约束的优化。

难度：低到中等。

### P4：回归并加固稳定主流程

在 `cem_1`、`cem_3` 上回归新体初值路径。同时给 Simsopt Boozer 阶段增加“磁面身份”
检查：优化前后把物理点代回 $s$，检查 $s$ 均值、法向距离、iota 连续性、体积和拓扑，
防止再次接受固定体积下的分支跳转。

难度：中等，但这是把研究成果安全并回主流程的必要条件。

---

## 8. 恢复工作时最需要记住的概念

| 符号 | 含义 | 不要混淆为 |
|---|---|---|
| $s(R,Z,\phi)$ | 由 $\boldsymbol B\cdot\nabla s\simeq0$ 拟合的无量纲局部不变量 | 物理磁通 |
| $\psi_T(s)$ | 通过截面磁通积分标定的物理环向磁通 | 原始拟合标签 $s$ |
| $\rho$ | $\sqrt{\psi_T/\psi_{T,\rm edge}}$ | 物理距离半径 |
| $\theta$ | 相对磁轴的几何极向角 | Boozer 极向角 |
| $\lambda$ | 把几何角修正为直场线角的周期函数 | R/Z 几何修正本身 |
| $\vartheta$ | $\theta+\lambda$，直场线极向角 | 完整 Boozer 坐标的全部 |
| $\alpha$ | $\vartheta-\iota\phi$，场线标签 | 单独的极向角 |
| $\nu$ | 修正沿场环向参数速度的周期函数 | 移动磁面几何的优化 |
| $(\theta_B,\phi_B)$ | 加入 $\nu$ 后的近 Boozer 坐标 | 旧 Simsopt 文件参数必然正确 |
| $\mathcal R_l^m(\rho)$ | Zernike 径向多项式 | 柱坐标大半径 $R$ |
| $\boldsymbol B_{\rm coil}$ | 线圈 Biot-Savart 真空场 | DESC 自动采用的内部场 |
| $\boldsymbol B_{\rm DESC}$ | DESC 根据平衡变量重建的场 | 外部线圈场的直接拟合 |

---

## 9. Git 时间线

当前分支相对 `main` 的 6 个提交：

| 提交 | 内容 |
|---|---|
| `d5e8e5a` | 存档最初的 DESC 体初值、联合 LS、bug 修复和 R/Z 嵌套诊断 |
| `e8654ae` | 加入稠密 Clebsch α、物理磁通标定、R/Z 投影和 DESC 实验 |
| `3754ddc` | 统一报告公式为 `$` / `$$` LaTeX 分隔符 |
| `03adc43` | 明确 $\mathcal R_l^m$ 是 Zernike 径向基，不是大半径 $R$ |
| `3404ae7` | 固定目标磁面诊断 Simsopt Boozer residual，识别旧曲面分支跳转 |
| `4fde178` | 加入环向修正 $\nu$，完整 Boozer residual 降低 43-404 倍 |

当前工作区在本报告写入前是干净的，没有正在运行或未归档的实验。

---

## 10. 从哪些文件继续看

### 建议阅读顺序

1. [`alpha_clebsch_experiment_report.md`](alpha_clebsch_experiment_report.md)：最新、最完整的
   $\psi_T\rightarrow\alpha\rightarrow R/Z\rightarrow\nu$ 实验和原始数字。
2. [`cem_qh03_psi_rz_nesting_diagnosis.md`](cem_qh03_psi_rz_nesting_diagnosis.md)：代码 bug、
   $s$ 面为何良好、R/Z 早期为何折叠。
3. [`desc_joint_linear_ls_audit.md`](desc_joint_linear_ls_audit.md)：外部联合 LS 和 DESC
   0.16 接口审计。
4. [`desc_problem_initial_guess_analysis.md`](desc_problem_initial_guess_analysis.md)：较长的理论与
   DESC 初值分析，适合查推导，不适合快速回忆。
5. [`cem_1_report/report.md`](cem_1_report/report.md) 和
   [`cem_3_report/report.md`](cem_3_report/report.md)：稳定主流程成功参考。

### 当前核心实现

- `stellarator_eval/alpha_clebsch.py`：磁通标定、α/$\lambda$/$\iota$ 线性系统。
- `stellarator_eval/toroidal_correction.py`：$D\nu$ 的 Fourier 线性解和坐标映射。
- `stellarator_eval/desc_joint_ls.py`：早期 DESC R/Z/L 联合 LS 与 phase 约束。
- `scripts/alpha_clebsch_ls_experiment.py`：稠密 GPU QR 主实验。
- `scripts/desc_alpha_rz_projection_experiment.py`：直场线坐标到 DESC R/Z 体谱及 solve。
- `scripts/diagnose_alpha_boozer_residual.py`：固定磁面 residual 与旧曲面分支诊断。
- `scripts/diagnose_alpha_toroidal_correction.py`：最新 $\nu$ 修正实验。

### 原始结构化结果

- `reports/alpha_clebsch_experiment/alpha_summary.json`
- `reports/alpha_clebsch_experiment/rz_projection_summary.json`
- `reports/alpha_clebsch_experiment/desc_solve_summary.json`
- `reports/alpha_clebsch_experiment/alpha_boozer_residual_summary.json`
- `reports/alpha_clebsch_experiment/alpha_toroidal_correction_summary.json`

---

## 11. 最终结论

过去这轮工作不是“反复给 DESC 调一个更好的点云初值但仍然失败”。更准确的总结是：

1. 先证明并修复了早期 R/Z 非嵌套中的代码和坐标标签问题；
2. 放弃病态的短场线 phase 点云拟合，改成十万级稠密、一次线性 GPU QR 的
   Clebsch α 路线；
3. 得到与独立长场线一致的 $\iota$，并把磁力线明显拉直；
4. 识别出直场线坐标与完整 Boozer 坐标之间缺失的环向速度条件；
5. 用另一个稳定线性问题求出 $\nu$，在不移动目标磁面的情况下把完整 Boozer residual
   降到 $10^{-4}$ 至 $10^{-3}$ 量级；
6. 剩余核心问题已经收敛为 DESC 接口与物理闭合，而不是“磁面是否存在”或
   “最小二乘是否会发散”。

下一次恢复开发时，第一件事不是重新跑大规模阶数扫描，也不是直接再调用 `eq.solve()`；
而是把 $\alpha+\nu$ 最终坐标明确映射到 DESC，随后完成三种磁场的同点比较。

---

## 12. 补充：α 路线是否已经给出完整 Boozer 坐标

本节回答四个容易混在一起的问题：现在是否已经得到完整 Boozer 坐标、它是单面还是
体坐标族、从 $\psi$ 到 Boozer 坐标经过什么步骤、最新结果是否已经放进 DESC。

### 12.1 先给直接结论

**对已经测试的固定磁面，可以认为完整 Boozer 坐标的已知机制已经补齐。**

这里的“完整”是相对于当前零压力、无等离子体环向电流的线圈真空场，以及 Simsopt
实际使用的 Boozer surface residual：

$$
\boldsymbol r_B
=G\boldsymbol B
-B^2\left(
\boldsymbol x_{\phi_B}
+\iota\boldsymbol x_{\theta_B}
\right).
$$

α 步骤先解决磁力线方向：

$$
\alpha=\theta_B-\iota\phi_B,
$$

使磁力线在角坐标中近似为直线。随后 $\nu$ 步骤解决沿磁力线的参数速度，使
$G_{\rm local}$ 在磁面上近似成为常数。修正后我们没有只检查某个自定义代理量，而是把
修正后的 `SurfaceXYZTensorFourier` 直接送入 Simsopt 的 `boozer_surface_residual`，并在
与拟合网格错开的独立网格上计算。结果为：

- $0.12\le\rho\le0.9$：完整相对 residual 为
  $3.52\times10^{-4}$ 至 $1.10\times10^{-3}$；
- $\rho=1$：完整相对 residual 为 $3.31\times10^{-3}$；
- 相比 α-only 的约 14%，降低了 43 至 404 倍；
- 坐标映射满足 $0.7435\le1+D\nu\le1.2054$，没有折叠。

因此，如果把**修正后且参数化正确的同一张曲面**作为原 Simsopt Boozer LS/Newton 的
初值，它进入的就是原求解器所使用的同一个 residual，而且初始 residual 已经是小量。
从当前证据看，Boozer 阶段不再缺少一个类似“环向角还没修正”的已知机制。

但要保留三个严格限定：

1. 我们实际调用了原 residual 函数做独立验证，**还没有真的从该初值再运行一次原
   LS/Newton**。所以“小残差初值”已经验证，“Newton 必然不跳分支”尚未实测。
2. 这些仍是近似磁面。$s$、α 和有限阶曲面投影都有误差，边界的方向误差约
   $0.40^\circ$，所以 residual 不为零。
3. “Boozer 坐标机制已补齐”不等于“DESC 平衡机制已闭合”。DESC 会根据固定边界、
   总磁通和 profile 重建自己的磁场，不会直接采用线圈场；二者之间仍有尚未核对的
   物理接口。

所以最准确的表述是：

> **对测试过的固定磁面，α+$\nu$ 已经给出了数值上接近完整 Boozer 的坐标；剩余是
> 近似磁面、有限谱和拟合误差，而不是另一个已知缺失的 Boozer 坐标条件。这个结论
> 不能直接外推成 DESC 初值已经正确。**

### 12.2 得到的是单个磁面，还是通用体坐标

答案介于二者之间，必须区分 α 和 $\nu$。

#### α 部分已经是体模型

α 拟合使用整个 $0.06\lesssim\rho\le1$ 体积中的 12 万个点，得到带径向 Zernike
展开的

$$
\lambda(\rho,\theta,\phi),
\qquad
\iota(\rho).
$$

因此 α 不是只在一个磁面上拟合的。给定一个新的 $\rho$，可以直接：

1. 取 $s=s_{\rm edge}\rho^2$；
2. 提取该 $s$ 等值面；
3. 计算该半径上的 $\lambda$ 和 $\iota$；
4. 得到直场线角 $\vartheta=\theta+\lambda$。

最终选用的拟合把 $\iota$ 固定为常数 $-0.565228$，因为允许低阶径向变化并没有降低
总体 residual，反而在采样不足的内层产生不可信摆动。这意味着当前例子中使用的是
“全体积常数 $\iota$ 近似”，不是已经高精度解析出任意径向的真实 $\iota(\rho)$ profile。

#### $\nu$ 部分目前仍是逐磁面模型

当前 $\nu$ 实验在

$$
\rho=0.12,0.2,0.3,\ldots,1.0
$$

共 10 个面上分别求解二维 Fourier 系数：

$$
\nu_\rho(\theta,\phi).
$$

代码尚未把这些系数再沿 $\rho$ 投影成一个连续的

$$
\nu(\rho,\theta,\phi)
$$

Fourier-Zernike 体模型，也没有验证相邻面之间的径向导数光滑性。

因此当前能力可以准确描述为：

- 给任意指定的 $\rho$，现有算法可以提取对应磁面，并通过一次独立的线性 Fourier
  求解现场得到该面的 $\nu$ 和近 Boozer 参数化；
- 但还不能只读取一套已经保存好的三维 $\nu$ 系数，就对任意 $\rho$ 立即求值；
- 也还没有证明这 10 个独立二维修正拼成的体坐标在径向上足够光滑，能直接作为 DESC
  的谱体坐标。

所以它已经是**通用的逐面生成算法**，但还不是最终封装完成的**连续 Boozer 体坐标
模型**。把 $\nu$ 系数沿径向做正则的 Zernike/多项式投影，并检查三维 Jacobian，是
进入 DESC 前还缺的一步。

### 12.3 从已有 $\psi$ 到近 Boozer 坐标的准确流程

这里的起点是 evaluator 已经拟合好的无量纲局部不变量 $s(R,Z,\phi)$。项目历史中有时
把它简称为 `psi`，但它在标定前不是物理磁通。

#### 第 1 步：把 $s$ 标定成物理环向磁通

在多个固定环向截面上，对 $s=s_k$ 内部区域积分：

$$
\Phi_T(s_k,\phi_j)
=\int_{D(s_k,\phi_j)}B_\phi\,dR\,dZ.
$$

对截面取平均并定义

$$
\psi_T(s)=\frac{\langle\Phi_T(s,\phi_j)\rangle_j}{2\pi}.
$$

再用单调四次多项式拟合 $\psi_T(s)$。本例边界总磁通为
$1.385027\times10^{-3}\,\mathrm{Wb}$，标定相对 RMS 为
$1.41\times10^{-8}$。

#### 第 2 步：建立物理径向坐标和几何角

定义

$$
\rho=\sqrt{\frac{\psi_T(s)}{\psi_{T,\rm edge}}},
$$

并以磁轴为中心定义几何极向角 $\theta$。同时由 $s$ 模型与标定函数计算
$\nabla\psi_T$。

#### 第 3 步：稠密线性最小二乘拟合 α

在体积内均匀取 12 万训练点和 6 万独立验证点，采样线圈 Biot-Savart 场。令

$$
\alpha
=\theta+\lambda(\rho,\theta,\phi)
-\iota(\rho)\phi,
$$

并最小化

$$
\boldsymbol B
-\nabla\psi_T\times\nabla\alpha.
$$

$\lambda$ 使用 Fourier-Zernike 基，$\iota$ 与所有 $\lambda$ 系数一起在线性系统中求出。
最终选用常数 $\iota=-0.565228$。这一步得到全体积的直场线角

$$
\vartheta=\theta+\lambda.
$$

#### 第 4 步：对指定 $\rho$ 提取固定物理磁面

在每个 $(\theta,\phi)$ 射线上解

$$
s(R,Z,\phi)=s_{\rm edge}\rho^2,
$$

得到同一个物理磁面的点。然后用 α 模型把它从几何角重新参数化到
$(\vartheta,\phi)$，并投影成有限阶 `SurfaceXYZTensorFourier`。

#### 第 5 步：求环向 Boozer 修正 $\nu$

在该固定磁面上计算

$$
G_{\rm local}
=\boldsymbol B\cdot
\left(\boldsymbol x_\phi+\iota\boldsymbol x_\vartheta\right),
\qquad
G=\langle G_{\rm local}\rangle,
$$

然后解周期磁微分方程

$$
\left(\partial_\phi+\iota\partial_\vartheta\right)\nu
=\frac{G_{\rm local}}{G}-1.
$$

在均匀周期网格上，这一步是正交 Fourier 线性求解。最后定义

$$
\phi_B=\phi+\nu,
\qquad
\theta_B=\vartheta+\iota\nu.
$$

它严格保持 α 不变，只修正沿磁力线的参数速度。

#### 第 6 步：反解坐标、重投影和独立验证

在规则 $(\theta_B,\phi_B)$ 网格上反解旧坐标，把**同一个物理磁面**重新参数化并投影，
随后在错位独立网格上直接调用 Simsopt `boozer_surface_residual`。这里没有移动磁面，也
没有运行 Boozer LS/Newton。

### 12.4 当前实测耗时

现有脚本主要为研究诊断编写，一次运行中包含阶数扫描、独立验证、画图和长场线追踪，
所以“实验总耗时”不等于未来单次生产路径耗时。下表只列保存结果中实际记录的数字：

| 步骤 | 实测耗时 | 说明 |
|---|---:|---|
| 原稳定 evaluator 默认全流程 | 2.809 s | `cem_qh03` 历史默认评估；包含磁轴、$s$、候选面和旧 Boozer，不是单独 $s$ 拟合耗时 |
| 物理磁通标定 | 24.811 s | 8 个环向截面、11 个 $s$ 层、256 个极向角、24 点径向积分 |
| 最终 α 线性系统组装 | 0.328 s | 120,000 点，360,000 行，2997 列 |
| 最终 α GPU FP64 QR | 18.925 s | `L12_M12_N16` |
| 最终 α 单模型组装加求解 | 19.290 s | 不含全部采样、验证和画图 |
| α 正式研究脚本总计 | 105.358 s | 含磁通标定、两个阶数模型、采样、验证、绘图和 8 周期场线诊断 |
| 单个 $s$ 面 GPU 提取 | 0.001-0.014 s | 10 个测试面的平均为 0.00965 s |
| α+$\nu$ 径向诊断总计 | 206.269 s | 10 个 $\rho$、3 个 $\nu$ 阶数，共 30 个完整投影/磁场/独立 residual 诊断 |
| 21 个 R/Z 内层面提取 | 0.130 s | 28,749 个点；这是较早 α-only DESC 投影路径 |
| R/Z 边界面提取 | 0.0014 s | 1369 个点 |
| `LMN=6` 的 R/Z 两个 QR 核心 | 0.111 s + 0.038 s | 仅 `torch.linalg.lstsq` 核心 |
| `LMN=6` R/Z 两个完整拟合 | 1.115 s + 0.317 s | 含矩阵和诊断；整个变体为 7.765 s，其中初始 force 计算 4.665 s |
| 四档 R/Z/DESC 初值诊断 | 84.222 s | `LMN=6,8,10,12`，不含最新 $\nu$ |
| α-only DESC solve | 20.000 s | 该 solve 物理发散，仅作失败记录 |

$\nu$ 的 312 模 Fourier 线性求解目前没有单独计时；206.269 s 的大头还包括反复的 CPU
Biot-Savart 场计算、Simsopt 曲面投影、坐标反解、三档阶数扫描和独立网格 residual。
因此不能把 206 s 理解成“求一个 $\nu$ 需要 206 s”。下一次正式接入时应给磁场采样、
$\nu$ 求解、坐标反解和重投影分别加 timer，并只跑选定的 12 阶模型。

也不能把表中各行简单相加：例如 24.811 s 的磁通标定已经包含在 105.358 s 的 α 脚本
总耗时中。当前能给出的可靠结论是，真正的大线性 QR 约 19 s，而面提取和 R/Z QR
本身都很快；研究脚本的大量时间花在重复诊断、磁场采样和 DESC/Simsopt 后处理。

### 12.5 最新近 Boozer 坐标是否已经放进 DESC

**没有。**

时间顺序是：

1. 先完成 α 拟合；
2. 用 $\vartheta=\theta+\lambda$ 把 21 个磁面投影到 DESC R/Z 体谱；
3. 把这套 **α-only、尚无 $\nu$** 的初值放入 DESC，并运行 solve；
4. solve 从嵌套初值发散成非嵌套解；
5. 之后才通过固定磁面 Simsopt residual 诊断发现缺少环向速度条件；
6. 最后才实现 $\nu$，把完整 Boozer residual 降低 43 至 404 倍。

当前 $\nu$ 只存在于独立诊断脚本生成的逐面修正结果中。它尚未：

- 拟合成连续径向体模型；
- 写入 DESC 的 toroidal coordinate / `omega` 或等价表示；
- 用于新的 DESC R/Z 体谱；
- 计算 $\boldsymbol B_{\rm DESC,initial}$ 与线圈场的同点差异；
- 运行新的 DESC solve。

因此旧的 DESC 发散结果不能用来判断 α+$\nu$ 初值是否有效。下一步应先完成连续
$\nu(\rho,\theta,\phi)$ 或 DESC 等价表示，并在 solve 前做坐标、Jacobian 和三种磁场
闭合检查。只有通过这些门槛后，才值得重新运行 DESC。

---

## 13. 补充：DESC 接口、微分 QS 指标与计算瓶颈

本节只做理论与源码接口分析，没有运行新实验。接口判断基于项目旁的 DESC 源码和此前
实际使用的 DESC 0.16 接口；正式实现前仍应在远端安装版本上再核对一次。

### 13.1 “把完整 Boozer 面塞进 DESC”究竟意味着什么

首先要修正一个容易产生误导的说法：**当前 DESC 并不把 Boozer 环向角当作平衡求解的
计算环向坐标。**

DESC 的基本体坐标是

$$
(\rho,\theta,\zeta),
$$

几何变量是

$$
R(\rho,\theta,\zeta),
\qquad
Z(\rho,\theta,\zeta),
$$

直场线极向角由

$$
\theta_{\rm PEST}=\theta+\lambda
$$

给出。源码形式上写有

$$
\phi=\zeta+\omega,
$$

但当前 `Equilibrium` 中 `omega` 及其所有导数都被硬编码为零，参数列表中注释掉了
尚未实现的 `W_lmn`。因此实际是

$$
\phi=\zeta,
$$

也就是 DESC 计算环向角等于实验室柱坐标环向角。

DESC 自己在平衡结果上做 Boozer 后处理时，才计算

$$
\nu=\zeta_B-\zeta,
$$

以及

$$
\theta_B=\theta_{\rm PEST}+\iota\nu,
\qquad
\zeta_B=\phi+\nu.
$$

这与我们从线圈场求出的 α+$\nu$ 关系完全同型，但在 DESC 中它是**由平衡磁场计算出的
后处理坐标变换**，不是 `Equilibrium` 可输入的独立自由度。

#### 不能做的接口映射

不能简单把我们的 $\phi_B$ 当成 DESC 的 `zeta`，再把对应物理点的 R/Z 传进去。
原因是 DESC 会用

$$
x=R\cos\zeta,
\qquad
y=R\sin\zeta
$$

重建物理点；而同一个原始物理点实际位于

$$
\phi=\phi_B-\nu.
$$

若把 `zeta=phi_B`，R/Z 无法吸收缺失的环向旋转，得到的是另一个物理点和另一张曲面。
这不是参数化误差，而是几何被移动了。

也不能指望直接设置 `eq.omega` 或 `W_lmn`，因为当前接口没有这些自由度。若修改 DESC
使其支持非零 $\omega$，需要贯通基函数、基矢、Jacobian、约束和求解变量，是一项较大的
DESC 功能开发，而且对当前目标未必必要。

#### 可以做的三种接口

**接口 A：只把最外层物理几何作为 fixed boundary。**

丢弃 Boozer 参数标签，只按实验室柱坐标角 $\phi$ 重新采样同一物理边界，拟合为
`FourierRZToroidalSurface`，然后构造 `Equilibrium(surface=..., axis=...)`。这就是传统
边界接口。它只能利用磁面形状，不能利用体内磁面或 α 坐标。

因为 $\nu$ 只重新参数化同一物理面，不移动几何，所以 α+$\nu$ 边界通过这种接口进入
DESC 后，与 α-only 的同一物理边界没有本质区别。不能预期仅靠 $\nu$ 让 boundary-only
DESC solve 突然收敛。

**接口 B：通过点值接口初始化完整 R/Z/L 体谱。**

DESC 0.16 支持

```python
eq.set_initial_guess(grid, R, Z, lambda_values, ensure_nested=False)
```

其中 `grid` 节点必须使用 DESC 的 $(\rho,\theta,\zeta)$，并保持
$\zeta=\phi_{\rm lab}$。α 体模型可用两种等价 gauge：

1. 取 DESC $\theta$ 为原几何角，传入我们拟合的
   $\lambda(\rho,\theta,\phi)$；
2. 取 DESC $\theta=\vartheta=\theta_{\rm geom}+\lambda$，传入
   `lambda_values=0`。

第二种就是已经做过的 α-only R/Z 体投影。它把直场线极向坐标直接吸收到 R/Z 的参数
标签中，使 DESC 初始 `L_lmn=0`。两种方式在连续精确表示下等价；有限谱下应比较哪一种
R/Z/L 总体阶数更低、Jacobian 裕度更大。

点值拟合后必须：

- 从拟合体的 $\rho=1$ 重新取 `eq.surface`，确保 fixed boundary 与体末层同物同参；
- 统一 handedness，避免 DESC 构造时自动翻转后又被外部系数覆盖；
- 使用 `ensure_nested=False`，避免 `GoodCoordinates` 在输入后自动大幅改变已验证坐标；
- 自己检查 $\sqrt g$、$\sqrt g_{\rm PEST}$ 和边界误差。

**接口 C：直接写 R/Z/L 谱系数或从另一个 Equilibrium 初始化。**

如果外部 QR 已经使用 DESC 的同一套 `FourierZernikeBasis`，可直接设置
`R_lmn`、`Z_lmn`、`L_lmn`，或先构造一个只保存这些系数的 `Equilibrium`，再调用
`set_initial_guess(eq_source)`。这比让 DESC 再拟合一次点云更可控，但仍必须满足接口 B
的边界、方向和 profile 约定。

#### profile 是比 $\nu$ 更重要的接口问题

DESC 不允许同时指定 `current` 和 `iota` profile。当前目标使用

$$
p=0,
\qquad
\mathrm{current}=0,
$$

于是 DESC 会在求解中自行得到 $\iota(\rho)$。α 拟合得到的 $\iota$ 目前只用于构造和
诊断直场线坐标，不能在不改变物理问题的情况下同时强制写入。

可以另外做一个**诊断变体**：固定 α 得到的 `iota` profile，而不指定 current，看看
DESC 初始场是否更接近线圈场。若它收敛但产生明显非零电流，只能说明 current/iota
profile 是发散来源之一，不能当作最终无电流真空解。最终目标仍应回到 `current=0`。

#### 预期效果

把完整 Boozer **物理面族**用于接口 B/C，预期仍然是有价值的：它提供嵌套 R/Z 和
正确的 PEST 直场线极向标签，比仅给边界更接近目标体坐标。

但最新 $\nu$ 本身不应直接进入当前 DESC，也不太可能单独修复力平衡发散。原因是：

1. $\nu$ 不移动物理磁面；
2. α-only 初值已经把正确的直场线极向坐标放入 DESC；
3. DESC 发散更可能来自它按 `current=0`、总磁通和有限谱重建出的
   $\boldsymbol B_{\rm DESC}$ 与 $\boldsymbol B_{\rm coil}$ 不一致。

因此本次源码审计**修正并取代**前文“下一步把连续 $\nu$ 写入 DESC”的表述。更准确的
下一步是：继续使用实验室 $\phi$ 和 α/PEST 体初值，在 solve 前比较

$$
\boldsymbol B_{\rm coil}
\quad\text{与}\quad
\boldsymbol B_{\rm DESC,initial},
$$

并隔离误差来自总磁通、current/iota profile、R/Z/L 有限谱还是 DESC 接口约定。

### 13.2 是否存在适合稠密点云统计的微分 QS error

**存在，而且 DESC 已经实现了两种。**

#### 指定 QH/QA/QP 类型的 two-term residual

给定螺旋度 $(M,N)$，定义

$$
A=(\boldsymbol B\times\nabla\psi)\cdot\nabla B,
\qquad
C=\boldsymbol B\cdot\nabla B.
$$

完美准对称要求存在通量函数

$$
F_{M,N}(\psi)
=\frac{M G(\psi)+N I(\psi)}{M\iota(\psi)-N},
$$

使

$$
A-F_{M,N}C=0.
$$

DESC 的 `QuasisymmetryTwoTerm` 使用等价的点值残差

$$
f_C
=(M\iota-N)A
-(M G+N I)C.
$$

它直接支持 `helicity=(M,N)`。按 DESC 当前 convention，QA 常用 $(1,0)$，QH 示例使用
$(1,N_{\rm FP})$；本项目角度和 handedness 可能要求 $N$ 取相反符号，因此必须用已知
QH 样本做一次符号回归，不能只凭名称硬编码。

这个公式完全在物理空间中成立，不要求先做 Boozer 变换。只要有

$$
\boldsymbol B,
\quad
\nabla B,
\quad
\nabla\psi,
\quad
\iota(\psi),
\quad
I(\psi),G(\psi),
$$

就能在十万级甚至百万级点上逐点计算。

#### 不指定对称类型的 triple-product residual

DESC 的 `QuasisymmetryTripleProduct` 使用

$$
f_T
=\nabla\psi\times\nabla B
\cdot\nabla(\boldsymbol B\cdot\nabla B).
$$

它不需要事先给 $(M,N)$，但需要更高一阶空间导数，而且只能判断“是否接近某种局部
QS”，不能直接回答“是否接近指定 QH”。它对数值噪声更敏感，不适合作为第一版稠密
筛选指标。

因此当前最合适的是 helicity-specific two-term $f_C$。

#### 如何定义可比较的平均误差

原始 $f_C$ 有量纲，不能直接把不同磁场强度、尺寸或磁通范围的样本做均值比较。
讲义中的

$$
\left\langle
\left(\frac{A-FC}{B^2}\right)^2
\right\rangle_\psi
$$

仍然依赖磁通和长度归一化。对批量评分，更稳健的无量纲定义是

$$
\epsilon_C^2(\rho)
=
\frac{
\left\langle f_C^2\right\rangle_\rho
}{
\left\langle
\left[(M\iota-N)A\right]^2
+\left[(M G+N I)C\right]^2
\right\rangle_\rho
+\epsilon
}.
$$

它把“两个本来应相等的项之差”除以两项自身的 RMS 尺度，不会在 $\nabla B=0$ 的局部
极值点逐点除以接近零的数。还应同时保存 DESC 风格的
$\sqrt{\langle f_C^2\rangle}/B_{\rm ref}^3$，便于与 DESC 输出交叉验证。

统计时必须先明确“平均”的含义：

- 若点在物理体积中均匀采样，简单平均近似体积平均；
- 若每个 $\rho$ bin 放相同点数，得到的是各径向层等权平均，不是体积平均；
- 若要严格磁面平均，需使用磁面面积或通量 Jacobian 权重；
- 最好同时输出 $\epsilon_C(\rho)$ 径向曲线和一个明确权重的总体分数，不要只留单个数。

当前 $s$ 近似误差会污染 $\nabla\psi$，所以还应同步报告

$$
\frac{|\boldsymbol B\cdot\nabla\psi|}
{|\boldsymbol B|\,|\nabla\psi|}
$$

的径向分位数。否则无法区分 QS 差和磁面标签本身不准。

### 13.3 计算微分 QS error 是否只需要 α，需不需要全空间 $\nu$

**不需要全空间 $\nu$。严格地说，连完整 α 函数也不是微分 $f_C$ 的必要输入。**

$f_C$ 是坐标不变的物理空间标量。逐点计算时不使用
$\theta_B$、$\phi_B$ 或 $\nu$。α 路线在这里提供的主要附加信息只有
$\iota(\rho)$；$\lambda$ 的全部高阶系数并不直接进入 $f_C$。

对当前线圈真空场：

- $I(\psi)$ 对应包围的等离子体环向电流，目标情况下应为零；
- $G(\psi)$ 可由已知线圈链接电流、安培环路或 α 面上的 $G_{\rm local}$ 面平均得到；
- $\iota$ 可由 α 拟合或分层长场线追踪得到。

因此若已经有可靠的 $\iota,I,G$，计算微分 QH error 只需

$$
\psi,\nabla\psi,
\boldsymbol B,\nabla B,
$$

无需 α 的角坐标，也无需 $\nu$。

如果暂时不信任 $I,G,\iota$，还可以在每个径向 bin 对

$$
A-FC
$$

直接拟合一个最优常数

$$
F_*(\rho)
=\frac{\langle AC\rangle_\rho}
{\langle C^2\rangle_\rho}.
$$

这能回答“该面是否接近某种微分 QS”，但不能单独证明它是目标 QH。应同时比较
$F_*$ 与目标 $F_{M,N}$；前者 residual 小而两者差异大，说明场接近另一种 helicity
或输入的 $I,G,\iota$ 有问题。

只有在下列目标中才需要 $\nu$：

1. 把 $B$ 写成 $B(\rho,\theta_B,\phi_B)$；
2. 计算 Boozer Fourier 非对称模能量；
3. 画 Boozer $|B|$ 等高线；
4. 与 Simsopt/DESC 的 Boozer 谐波 QS 指标逐项对照。

所以建议把评估器拆成两层：

| 模式 | 目标 | 是否需要 α | 是否需要 $\nu$ |
|---|---|---|---|
| 快速微分 QS | 大体积稠密点云上的 $f_C$ 径向统计与总体分数 | 只需要其中可靠的 $\iota$，也可由追线替代 | 不需要 |
| 完整 Boozer 审计 | Fourier QS、等高线、坐标和分支验证 | 需要 | 需要 |

### 13.4 若目标改成稠密微分 QS，真正的耗时路径

前一节的计时之所以难读，是因为它记录的是研究脚本，混合了阶数扫描、画图、
Simsopt 投影、独立验证和失败的 DESC solve。若目标只变成“从已有 $s$ 计算稠密微分
QH error”，真正需要的路径是：

```text
已有 s 模型
  |
  +-- 物理磁通标定 psi_T(s)
  |
  +-- 获得 iota(rho), I(rho), G(rho)
  |
  +-- 在物理体积中生成稠密点并计算 psi、grad(psi)
  |
  +-- 计算 B、grad(|B|)
  |
  +-- GPU 上逐点计算 f_C，按 rho 分箱归约
  |
  +-- 输出径向曲线、总体分数和 psi 法向误差
```

以下步骤全部不再需要：逐面提取、$\nu$ 求解、坐标反解、Simsopt 曲面重投影、Boozer
LS/Newton、R/Z DESC 投影和 DESC solve。因此前文的 206 s、84 s 和 20 s 都与快速
微分 QS 路径无关。

现有实测中仍相关的只有：

| 相关步骤 | 当前数字 | 在新路径中的地位 |
|---|---:|---|
| 物理磁通标定 | 24.811 s | 当前明显偏慢，但包含大量重复小批次磁场调用 |
| 完整 α 单模型 QR | 19.290 s | 只有选择继续用 α 获得 $\iota$ 时才需要 |
| 稳定 evaluator 全流程 | 2.809 s | 历史参考，包含其它阶段，不能与上两项直接相加 |
| 稠密 $\boldsymbol B,\nabla B$ | 尚未单独计时 | 预计将成为主要运行瓶颈 |
| $f_C$ 点运算和分箱归约 | 尚未实现 | 仅 $O(N)$，放在 GPU 后预计很小 |

如果已有 $s$ 和 α 文件，增量计算中不需要重新做 24.8 s 标定和 19.3 s QR；只需加载
标定、$\iota$，再做稠密 $\boldsymbol B,\nabla B$ 和归约。

若对每个新线圈都从头评分，最优先的架构改动不是优化 $\nu$，而是：

1. 批量化物理磁通标定；
2. 决定是否真的需要完整 α QR，还是用便宜的分层追线获得 $\iota$；
3. 实现融合的 GPU $\boldsymbol B+\nabla B+f_C$ kernel。

### 13.5 GPU 与 FP32 优化判断

#### 第一优先级：融合解析的 GPU $\boldsymbol B+\nabla B$

对每个线段，Biot-Savart 贡献可写成

$$
\boldsymbol B_s
=c\frac{\boldsymbol w\times\boldsymbol r}{r^3}.
$$

其解析导数为

$$
\frac{\partial B_{s,i}}{\partial x_j}
=c\left[
\frac{\epsilon_{ikj}w_k}{r^3}
-3\frac{(\boldsymbol w\times\boldsymbol r)_i r_j}{r^5}
\right].
$$

因此一次遍历线圈线段时可以复用 $\boldsymbol r$、$r^{-3}$、$r^{-5}$ 和叉积，同时累计
$\boldsymbol B$ 与导数。最后直接计算

$$
\nabla B
=\frac{(\nabla\boldsymbol B)^T\boldsymbol B}{|\boldsymbol B|}.
$$

如果只为 $f_C$，无需把完整 $3\times3$ Jacobian 复制回 CPU；kernel 可直接输出
$\boldsymbol B$、$\nabla B$，甚至继续融合 $\nabla\psi$、$f_C$ 和径向 bin reduction。

这比有限差分优越：有限差分至少需要 6 次额外磁场评估，并引入步长选择和相消误差；
解析融合只增加每个线段的局部算术，具有高并行度和较高运算密度，最适合 GPU。

当前 GPU 后端的批量 `sgpu_eval_B` 仍是 FP64，且没有 $\nabla B$ API；已有 FP32/mixed
kernel 主要用于场线追踪和 $s$ 拟合。因此这里存在明确的优化空间。

#### 第二优先级：点值 FP32，关键归约 FP64

RTX 5090 的消费级 FP64 吞吐远低于 FP32。对远离线圈、磁场光滑的体内筛选点，第一版
可采用：

- 线圈线段、采样点、Biot-Savart 与解析导数使用 FP32；
- 线段贡献使用树形/分块归约，避免完全串行累加误差；
- $f_C^2$、分母和 bin 计数的最终统计使用 FP64 或补偿求和；
- 在各 $\rho$ bin 抽取约 1% 点用 FP64 重算，持续报告 FP32/FP64 的 p50、p95 和最大差；
- 对近轴、$|\nabla\psi|$ 很小或两大项强相消的点自动回退 FP64。

不能仅以 $\boldsymbol B$ 的相对误差判断 FP32 是否够用，因为 $f_C$ 是两个大项之差，
且依赖导数。验收标准应直接看最终 $\epsilon_C(\rho)$ 和 helicity 排名在 FP32/FP64
之间是否稳定。

#### 第三优先级：批量化磁通标定

24.811 s 的标定目前按多个截面和层反复构造积分点并调用磁场。应把所有
$(s_k,\phi_j,\theta_q,r_q)$ 积分点一次性或分大块送入 GPU，并在 GPU 上完成径向、极向
和截面的分层归约，只把几十个 $\Phi_T(s_k,\phi_j)$ 标量传回 CPU。

磁场点值可先用 FP32，积分归约用 FP64。标定结果已有跨截面相对一致性
$1.61\times10^{-4}$，所以精度验收可以直接要求 mixed 结果相对 FP64 的偏差显著低于
这一水平。

#### 第四优先级：若保留 α，尝试 mixed-precision QR

当前 α 的 360,000 行、2997 列 FP64 QR 单次约 18.925 s，是保留完整 α 路线时最明确的
线性代数瓶颈。设计矩阵约有 $1.08\times10^9$ 个元素，FP64 存储约 8.6 GB，FP32 可减半
内存并显著提高 5090 吞吐。

推荐顺序是：

1. 保留列归一化；
2. 用 FP32 QR 求初解；
3. 在 FP64 或混合精度中计算原方程 residual；
4. 必要时做一到两次 iterative refinement；
5. 与 FP64 QR 比较 $\iota$、场线直线 residual、$1+\lambda_\theta$ 最小值和最终
   微分 QS 排名。

不建议第一步就改用 FP32 normal equations，因为 $A^TA$ 会平方条件数，违背当前选择 QR
的稳定性目的。

更根本的问题是：若快速 QS 只需要 $\iota$，完整 2997 列 α QR 可能本来就不应位于
快速评分路径。可以先比较分层 GPU 场线追踪得到的 $\iota(\rho)$ 与 α 结果；若精度足够，
快速模式可完全省掉这 19 s，把完整 α 留给 Boozer 审计模式。

#### 第五优先级：数据常驻 GPU 与流式分块

从点生成开始就让 R/Z/$\phi$、$s$、$\nabla\psi$、$\boldsymbol B$ 和 $\nabla B$ 常驻 GPU。
按显存选择例如 64k 至 256k 点一块，直接累加每个 $\rho$ bin 的计数、分子、分母和误差
分位数所需直方图，避免把十万级导数数组来回复制。

### 13.6 推荐的新评估架构

综合物理需求和性能，建议把后续工作拆成两条明确路径。

**快速评分路径：**

```text
axis + s
  -> flux calibration
  -> cheap iota / I / G
  -> dense GPU B + grad(B) + grad(psi)
  -> QH f_C radial profile + volume score
```

特点：不做 $\nu$、不做 Simsopt Boozer Newton、不做 DESC；适合大批量线圈候选。

**物理审计路径：**

```text
fast score passed
  -> dense alpha QR
  -> selected rho surfaces
  -> per-surface nu and Boozer Fourier QS
  -> branch identity checks
  -> DESC PEST volume initial guess and field closure
  -> constrained DESC continuation
```

特点：保留完整坐标、图和交叉验证，只对少量高质量候选运行。

这两个指标应相互校验，但不应强迫快速评分每次都支付完整 Boozer 和 DESC 的成本。

## 14. 新目标：从稳定版 $\psi$ 到 GPU 体积 QS

### 14.1 任务边界

本阶段明确把下面这段视为**冻结且已经验证的稳定版上游**：

$$
\text{线圈}
\longrightarrow
\text{磁轴}
\longrightarrow
\text{拟合磁通函数}
$$

磁轴怎样搜索、$s$ 怎样拟合、上游使用多少点、上游 GPU 是否还能更快，都不属于本阶段工作。新任务从稳定版已经接受的磁通函数开始，只研究

$$
\boxed{
\text{稳定版磁通函数}
\longrightarrow
\text{指定 helicity 的体积 QS error}
}
$$

后文所说的性能、稳定性和算法设计，均只指这个下游区间。上游耗时也不计入下游优化账目。

唯一需要检查上游接口的情况是：微分 QS 公式需要某个物理量，而稳定版对象尚未提供它。此时只补齐接口或在下游推导该量，不改变稳定版算法。

### 14.2 稳定版实际交付了什么

当前代码中的 **PsiModel** 实际给出的是无量纲近似不变量

$$
s=s(R,Z,\phi)
$$

及其解析梯度 $\nabla s$。项目中一直把这一步简称为“拟合 $\psi$”，但从物理量纲看，**PsiModel** 的 $s$ 还不是以 Wb/rad 表示的物理环向磁通

$$
\psi_T=\frac{\Phi_T}{2\pi}.
$$

这只是接口命名需要澄清，并不表示稳定版路线有问题。对新下游而言，稳定版应被当作提供以下只读输入：

- 已通过稳定版质量门槛的 $s(R,Z,\phi)$ 和 $\nabla s$；
- 稳定版找到的磁轴几何，用于定义极向角 $\theta$，不重新求磁轴；
- 稳定版确认的可用体积范围 $0\le s\le s_{\rm edge}$；
- 原线圈磁场对象、线圈电流和 $N_{\rm FP}$，供下游计算 $\boldsymbol B$ 与导数；
- 稳定版已有的 $s$ residual、嵌套性和失败状态，供最终 QS 结果附带引用。

若将来稳定版直接增加了物理映射 $\psi_T(s)$，下游直接使用即可。按当前代码，尚需在下游得到

$$
F(s)=\frac{d\psi_T}{ds}.
$$

下面给出的首选路线可以在线性 LS 中同时求出 $F(s)$，因此不要求修改稳定版，也不要求先做逐磁面磁通标定。

### 14.3 从 $\psi$ 到微分 QS 真正缺少的量

给定目标对称模式 $(M,N)$，采用前文已经核对过的两项微分 QS 条件：

$$
A=(\boldsymbol B\times\nabla\psi_T)\cdot\nabla B,
\qquad
C=\boldsymbol B\cdot\nabla B,
$$

$$
f_C=(M\iota-N)A-(MG+NI)C.
$$

因此，从稳定版 $s$ 到 $f_C$ 还需要：

1. 物理梯度 $\nabla\psi_T=F(s)\nabla s$；
2. 旋转变换 $\iota(s)$；
3. 真空电流函数 $I$ 和 $G$；
4. $B=|\boldsymbol B|$ 及 $\nabla B$；
5. 在目标体积中的均匀物理空间取点和归约。

不需要完整 Boozer 环向角 $\zeta_B$、全空间 $\nu$、显式的逐个 $R(\theta,\zeta)$ 和 $Z(\theta,\zeta)$ 磁面、Simsopt Boozer LS--Newton 或 DESC solve。

这里的 $\lambda$ 主要是求准 $\iota$ 和物理磁通尺度时必须保留的周期 nuisance function，而不是最终 QS 输出。

### 14.4 首选路线：直接线性拟合 Clebsch 势 $\beta$

这一部分可以把“物理磁通标定”和“$\alpha/\iota$ 拟合”合并为同一个线性问题。

对稳定版给出的任意良好磁面标签 $s$，写

$$
\boldsymbol B
=
\nabla\psi_T\times\nabla\alpha,
$$

$$
\alpha
=
\theta+\lambda(s,\theta,\phi)-\iota(s)\phi.
$$

定义

$$
F(s)=\frac{d\psi_T}{ds},
\qquad
H(s)=F(s)\iota(s),
\qquad
\widetilde{\lambda}=F(s)\lambda.
$$

再定义允许含 $\theta$、$\phi$ 世俗项的 Clebsch 势

$$
\beta
=
F(s)\theta-H(s)\phi+\widetilde{\lambda}(s,\theta,\phi).
$$

由于所有纯径向导数项都平行于 $\nabla s$，与 $\nabla s$ 叉乘后消失，所以有

$$
\boldsymbol B
=
\nabla s\times\nabla\beta.
$$

令

$$
\boldsymbol C_\theta=\nabla s\times\nabla\theta,
\qquad
\boldsymbol C_\phi=\nabla s\times\nabla\phi.
$$

把两个径向函数展开为

$$
F(s)=\sum_p a_p q_p(s),
\qquad
H(s)=\sum_p h_p q_p(s),
$$

并把周期函数展开为

$$
\widetilde{\lambda}
=
\sum_j c_j f_j(s,\theta,\phi).
$$

每个物理点上的磁场方程变成

$$
\boldsymbol B
\approx
\sum_p a_p q_p(s)\boldsymbol C_\theta
-
\sum_p h_p q_p(s)\boldsymbol C_\phi
+
\sum_j c_j
\left[
\frac{\partial f_j}{\partial\theta}\boldsymbol C_\theta
+
\frac{\partial f_j}{\partial\phi}\boldsymbol C_\phi
\right].
$$

未知量只有 $\{a_p\}$、$\{h_p\}$ 和 $\{c_j\}$，并且全部线性进入设计矩阵。用一次 GPU QR/TSQR 即可求解，没有非线性方程，也没有不确定次数的迭代。

求解后直接得到

$$
\frac{d\psi_T}{ds}=F(s),
\qquad
\psi_T(s)=\int_0^s F(u)\,du,
$$

$$
\iota(s)=\frac{H(s)}{F(s)},
\qquad
\lambda=\frac{\widetilde{\lambda}}{F}.
$$

积分和除法只是显式后处理，不是非线性求解器。$F(s)$ 应在有效区间内保持同号；若变号或接近零，直接把该模型判为失败，不追加非线性约束优化。

这个构造的意义是：即使稳定版只给无量纲 $s$，也可以从 $\boldsymbol B$ 与 $s$ 的几何关系中线性恢复物理磁通尺度和 $\iota$。它比“先抽取许多 $s$ 等值面，再逐面做磁通积分和 $\alpha$ 拟合”更符合本任务的稳定性与 GPU 目标。

### 14.5 为什么仍要保留周期基底

不能只拟合两个径向函数 $F$ 和 $H$，然后删除 $\widetilde{\lambda}$。几何极向角 $\theta$ 一般不是直场线角，周期角修正承担了吸收这一差别的作用。若删除它，LS 会被迫把角度误差投影到 $F$ 和 $H$，从而污染物理磁通尺度与 $\iota$。

但这里也不需要沿用“构造完整高精度 Boozer 坐标”所需的最高阶基底。正确目标是

$$
\text{寻找使 }F(s),\ \iota(s),\ \epsilon_C(\rho)
\text{ 收敛的最小周期基底},
$$

而不是追求 $\lambda$ 点值本身的极限精度。

建议采用当前已经实现并验证过的 Fourier--Zernike 类基底，并做固定列表的线性阶数扫描：

- 固定低阶 $F(s)$；
- 从常数 $\iota$ 开始，再逐步增加 $H(s)$ 的径向阶数；
- 逐步增加 $\widetilde{\lambda}$ 的极向、环向和径向阶数；
- 以独立点上的磁场切向 residual、$\iota$ 稳定性和最终 QS 分数稳定性决定停止；
- 不以训练 residual 单独决定阶数。

此前约 36 万行、2997 列、18.9 s 的 FP64 QR 是“完整 $\alpha$ 坐标拟合”的参考，不是新 QS 路线必须支付的固定成本。新路线首先要实测：为了稳定 $F$、$\iota$ 和 QS 排名，周期 nuisance basis 最少需要多少列。

### 14.6 联合 $\beta$ 拟合的风险与备用路线

联合 $\beta$ 方案在线性结构上成立，但有两个条件问题需要实验确认。

第一，原始三分量磁场 residual 容易被较大的环向磁场主导，而 $\iota$ 主要由较小的极向结构约束。建议采用：

- 设计矩阵列归一化；
- 按 $\rho$ bin 平衡行权重，避免外层体积点淹没近轴信息；
- 将 $\boldsymbol B$ 投影到 $\boldsymbol C_\theta,\boldsymbol C_\phi$ 的局部切向双基后做尺度均衡；
- 对 $H(s)$ 的高阶径向系数加入 Tikhonov 平滑行，仍保持普通线性 LS；
- 使用独立 shifted grid 交叉验证。

第二，稳定版 $s$ 是近似不变量，因此 $\boldsymbol B$ 可能含有不能由 $\nabla s\times\nabla\beta$ 表示的法向小量。定义

$$
\boldsymbol B_{\perp s}
=
\frac{\boldsymbol B\cdot\nabla s}{|\nabla s|^2}\nabla s.
$$

$\beta$ LS 只可能拟合切向部分。最终必须把

$$
\frac{\|\boldsymbol B_{\perp s}\|_2}{\|\boldsymbol B\|_2}
$$

作为不可消除下限单独列出，不能把它误判为基底不足。

若联合 $\beta$ 的 $H/F$ 对条件数过于敏感，备用路线仍然完全固定预算：

1. 使用稳定版已有的物理磁通输出；若没有，则用无求根的截面网格积分得到 $F(s)$；
2. 单独用标量直场线方程
   $$
   \boldsymbol B\cdot\nabla
   \left[
   \theta+\lambda-\iota(s)\phi
   \right]
   \approx0
   $$
   做一次线性 LS；
3. 用该 LS 的 $\iota(s)$ 计算 $f_C$。

标量方程每个点只有一行，矩阵比三分量 $\beta$ 拟合更小，而且直接针对场线直线性。它适合作为联合方案的交叉验证，也可能在实测后成为更快的正式方案。两条路线都不包含 LS--Newton 或开放式优化。

### 14.7 $I$ 和 $G$

当前目标是线圈在真空区域产生的磁场。在该前提下，

$$
I(\psi)=0,
$$

而 $G$ 是由链接电流决定的常数，可由原线圈电流和拓扑直接计算。这个步骤只是电流 bookkeeping，不需要拟合 $\nu$ 或构造 Boozer 面，计算成本可以忽略。

实现前必须锁定 $G$ 中的 $\mu_0/(2\pi)$ 因子、电流单位、角度和 $\psi_T$ 符号、$(M,N)$ 与 $N_{\rm FP}$ 定义，并与 DESC 的 **QuasisymmetryTwoTerm** 使用同一约定。应在 **cem_1**、**cem_3** 等可信例子上逐项对照。

若未来输入包含体电流，则 $I=0$ 不再成立。那属于新的物理范围，不在本阶段静默兼容。

### 14.8 体积取点与 QS 归约

得到 $F(s)$ 和 $\iota(s)$ 后，不需要抽取任何单独磁面。下游在稳定版确认的包围体积中建立确定性的 shifted Cartesian lattice，评估 $s$ 后保留

$$
\rho_{\min}
\le
\rho=
\sqrt{\frac{\psi_T(s)}{\psi_{T,\rm edge}}}
\le
\rho_{\max}.
$$

这只是使用稳定版 $s$ 筛点，不是重新拟合 $\psi$。在真实物理体积中均匀取点时，普通点平均已经代表物理体积权重，无需 Boozer Jacobian。

输出应同时包含径向剖面

$$
\epsilon_C^2(\rho_k)
=
\frac{
\left\langle f_C^2\right\rangle_{\rho_k}
}{
\left\langle
[(M\iota-N)A]^2+[(MG+NI)C]^2
\right\rangle_{\rho_k}
+\epsilon
}
$$

和全体积指标

$$
\epsilon_{C,V}^2
=
\frac{
\sum_{p\in V}f_{C,p}^2
}{
\sum_{p\in V}
\left(
[(M\iota_p-N)A_p]^2
+
[(MG_p+NI_p)C_p]^2
\right)
+\epsilon
}.
$$

还应报告未归一化 $f_C$ RMS、各径向 bin 的点数、近轴和边界排除比例、稳定版 $s$ residual、$\beta$ 验证 residual 以及 FP32/FP64 审计差异。

近轴处 $|\nabla s|$ 很小，$F$ 和 $\iota$ 更难辨识。第一版应明确设置小的 $\rho_{\min}$，单独报告未评分体积，之后再研究径向收敛。

### 14.9 下游 GPU 数据流

只考虑从稳定版磁通函数开始，建议数据流为：

    accepted PsiModel + stable axis + coil field
      -> dense deterministic volume points
      -> evaluate s, grad(s), theta geometry
      -> one fused GPU B + grad(B)
      -> linear beta QR/TSQR
      -> F(s), iota(s), grad(psi)
      -> f_C at the same points
      -> radial bins + volume reduction

主要复用关系是：

1. $\beta$ LS 和最终 QS 使用同一批体积点；
2. $\beta$ LS 直接复用最终融合 kernel 产生的 $\boldsymbol B$；
3. QS 复用同一 kernel 的 $\nabla\boldsymbol B$，不做有限差分；
4. $s$ 和 $\nabla s$ 只调用稳定版解析评估，不重新拟合；
5. $F(s)$、$\iota(s)$ 和基底值留在 GPU，直接生成 $\nabla\psi_T$ 和 $f_C$；
6. 只把小型 QR 因子、径向统计量和诊断传回 CPU。

对离散电流段，Biot--Savart 项及解析导数为

$$
B_i
=
c\frac{(\boldsymbol w\times\boldsymbol r)_i}{r^3},
$$

$$
\frac{\partial B_i}{\partial x_j}
=
c
\left[
\frac{\epsilon_{ikj}w_k}{r^3}
-
3\frac{(\boldsymbol w\times\boldsymbol r)_i r_j}{r^5}
\right].
$$

应在一个融合 kernel 中复用 $\boldsymbol r$、$r^{-3}$、$r^{-5}$ 和叉乘，同时得到 $\boldsymbol B$ 与 $\nabla\boldsymbol B$，再计算

$$
\nabla B
=
\frac{(\nabla\boldsymbol B)^T\boldsymbol B}{|\boldsymbol B|}.
$$

这是新下游最值得优化的算子，因为 $\nabla B$ 是体积 QS 新增且稳定版 $\psi$ 路线没有提供的物理量。

### 14.10 精度与稳定性

下游的计算契约应为：

- $\beta$ 或标量 $\alpha$ 只用直接 QR/TSQR；
- 不使用 normal equations；
- 不调用逐面 Newton；
- 不运行 DESC 或 Boozer nonlinear solve；
- 每个阶数只运行一次固定规模分解；
- mixed-precision iterative refinement 最多固定 1--2 次；
- 阶数扫描列表预先给定，不能根据 residual 无限追加；
- 未通过验收时返回结构化失败，不换初值反复重跑。

精度建议为：

- 点坐标、基底生成、Biot--Savart 与解析导数默认 FP32；
- 大矩阵第一遍 QR 使用 FP32；
- 列缩放、径向小函数系数、积分和全局归约使用 FP64；
- 每个 $\rho$ bin 固定抽样少量点，用 FP64 重算 $\boldsymbol B$、$\nabla B$ 和 $f_C$；
- 在 FP64 中计算独立验证 residual；
- 若 $f_C$ 两个大项严重消减，则该 bin 回退 FP64 或标为低可信度。

能否大部分使用 FP32，最终必须看 $\epsilon_C(\rho)$ 和候选排序的 FP32/FP64 差异，不能只看 $\boldsymbol B$ 点值误差。

### 14.11 瓶颈与实施顺序

冻结稳定版上游后，下游瓶颈只剩：

1. 十万级体积点上的融合 $\boldsymbol B+\nabla\boldsymbol B$；
2. $\beta$ 或标量 $\alpha$ 的 GPU QR/TSQR。

$f_C$ 点值计算、径向分 bin 和归约都只是线性复杂度，预计不是主瓶颈。优化优先级是：

1. 解析融合 $\boldsymbol B+\nabla\boldsymbol B$，杜绝有限差分和重复磁场调用；
2. 找到只保证 $F$、$\iota$ 与 QS 收敛所需的最小 nuisance basis；
3. FP32 QR 加 FP64 验证；
4. 矩阵过大时使用流式 TSQR；
5. 批量样本复用 kernel、工作区和基底逻辑。

这里不再把稳定版磁轴或 $s$ 拟合列为瓶颈，也不提出优化它们。目前还不能给出可信的下游秒数，因为没有融合 GPU $\nabla\boldsymbol B$ 的实测吞吐，也没有最小 $\beta$ 基底扫描。此前 18.9 s 的 2997 列 FP64 完整 $\alpha$ QR 只是保守上界参考。

后续建议按以下顺序实施：

1. 锁定 $f_C$、$(M,N)$、$I/G$ 和 $\psi_T$ 的约定；
2. 在少量点上建立 FP64 的 $f_C$ 逐点参考；
3. 实现并验证融合 GPU $\boldsymbol B+\nabla\boldsymbol B$；
4. 实现联合线性 $\beta$ 拟合，检查 $F(s)$、$\iota(s)$ 和法向 residual 下限；
5. 用“物理磁通标定加标量 $\alpha$ LS”作为独立参考；
6. 做基底阶数、点数、FP32/FP64 和径向收敛测试；
7. 跑通稳定版输出到体积 QS 的单样本与批量接口。

### 14.12 最终判断

从稳定版 $s$ 出发，体积微分 QS 路线在理论上闭合：

$$
s,\nabla s,\boldsymbol B
\overset{\text{linear LS}}{\longrightarrow}
F=\frac{d\psi_T}{ds},\ \iota,
$$

$$
F\nabla s,\ \iota,\ I,\ G,\ \boldsymbol B,\nabla B
\overset{\text{pointwise}}{\longrightarrow}
f_C,
$$

$$
f_C
\overset{\text{GPU reduction}}{\longrightarrow}
\epsilon_C(\rho),\ \epsilon_{C,V}.
$$

验收重点是：

- 不修改或重新评估稳定版线圈到 $\psi$ 路线；
- 下游所有高维求解均为线性 QR/TSQR；
- 不出现运行时间未知的非线性迭代；
- $F(s)$ 单调且不变号；
- $\iota$ 在独立网格和基底加阶后稳定；
- 目标 helicity 与错误 helicity 能正确区分；
- 最终 QS 分数和排序通过 FP64 参考与可信磁面结果的交叉验证。

**修正后的结论是：本任务不是重新设计“线圈到 QS”全链路，而是在稳定版“线圈到 $\psi$”之后增加一个独立、固定预算、GPU 化的体积 QS 后端。联合 Clebsch $\beta$ 线性 LS 有望直接补齐当前 PsiModel 缺少的物理磁通尺度和 $\iota$，随后无需完整 Boozer 坐标即可计算指定 helicity 的体积微分 QS。**
