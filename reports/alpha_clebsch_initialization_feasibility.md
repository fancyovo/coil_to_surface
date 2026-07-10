# 基于物理磁通标定与 Clebsch 势 alpha 的 DESC 初值方案可行性报告

日期：2026-07-10

说明：本报告只做理论与数值方案分析，没有为本报告新增代码实验。

## 1. 结论

设想的主方向是可行的：

1. 先得到一个描述嵌套磁面的近似不变量 $s(\boldsymbol x)$；
2. 将其标定为有物理量纲和正确归一化的环向磁通
   $\psi_T(\boldsymbol x)$；
3. 使用

   $$
   \boldsymbol B=\nabla\psi_T\times\nabla\alpha
   $$

   在稠密体网格上对 Clebsch 势 $\alpha$ 做一次线性拟合；
4. 从 $\alpha$ 的拓扑分解中直接得到 $\iota(\rho)$、
   $\lambda(\rho,\theta,\zeta)$ 和直线场角；
5. 最后把同一组磁面按选定角度参数化，线性拟合成 DESC 的 R/Z 体谱。

与当前“追踪少量场线点，再同时拟合每条线截距、lambda 和 iota”的方案相比，
alpha 路线具有明显优势：

- 可以使用与 psi 拟合同量级的稠密 Eulerian 点；
- 不需要逐条场线的 angle unwrap；
- 不需要为每条场线引入截距 $\beta_j$；
- 直接拟合磁场三个分量，而不是只拟合场线上角度值；
- iota 是 alpha 的拓扑绕数，不再只是短轨迹上的回归斜率；
- lambda 的导数直接进入目标，更容易控制坐标可逆性；
- 整个 alpha 阶段在 psi 和基底固定后仍是线性的。

但“标定 psi 后做一次普通线性最小二乘，就自动得到后面所有内容”仍漏了几项：

1. alpha 不是普通的单值双周期标量，必须显式保留其多值拓扑部分；
2. psi 标定只改变磁面标签的尺度，不会修复错误的磁面形状；
3. alpha 不能单独决定 R、Z，DESC 的几何映射仍需由 psi level sets 构造；
4. 为保证坐标不折叠，最好使用带线性不等式的凸二次规划，而不是完全无约束
   的最小二乘；
5. alpha 给出的是 Clebsch/直线场坐标，不自动等于完整 Boozer 坐标；
6. 它能大幅改善 DESC 初值，但不能替代最终的非线性 MHD 平衡求解。

因此，最准确的预期是：**后处理可以变成一个稳定的磁通标定、一个稠密线性或
凸 alpha 拟合、一个结构化 R/Z 谱投影；不再需要当前复杂且病态的场线点云联合
拟合，但仍保留稳定的一维 level-set 求根和最终 DESC solve。**

## 2. 需要先区分的三个 psi

当前项目中容易把三个概念都称为 psi：

### 2.1 近似不变量

当前多项式拟合得到的是无量纲函数

$$
s(\boldsymbol x),
$$

其目标主要是

$$
\boldsymbol B\cdot\nabla s\simeq0.
$$

它的等值面提供几何磁面候选，但 $s$ 的数值尺度由固定二次项等规范决定，没有
直接的 Weber 量纲，也不保证 $s$ 与真实环向磁通成正比。

### 2.2 物理环向磁通

记某个磁面包围的总环向磁通为

$$
\Psi_T(s),
$$

单位为 Weber。Clebsch 表示通常使用

$$
\psi_T(s)=\frac{\Psi_T(s)}{2\pi}.
$$

这一步就是“真实 psi 标定”：寻找单调函数 $f$，使得

$$
\psi_T(\boldsymbol x)=f\bigl(s(\boldsymbol x)\bigr).
$$

### 2.3 DESC 径向坐标

进入 DESC 时通常再归一化为

$$
\rho=\sqrt{\frac{\psi_T}{\psi_{T,\mathrm{edge}}}}.
$$

所以完整关系为

$$
s(\boldsymbol x)
\longrightarrow
\psi_T=f(s)
\longrightarrow
\rho=\sqrt{\psi_T/\psi_{T,\mathrm{edge}}}.
$$

## 3. 如何标定真实的物理磁通

### 3.1 截面通量积分

对于 $s=s_0$ 的闭合磁面，在固定几何环向角 $\phi=\phi_0$ 的极向截面内，
总环向磁通可写成

$$
\Psi_T(s_0)
=\int_{A(s_0,\phi_0)}
\boldsymbol B\cdot d\boldsymbol S.
$$

若截面是固定 $\phi$ 平面，按柱坐标定向可写成

$$
\Psi_T(s_0)
=\int_{A(s_0,\phi_0)} B_\phi(R,Z,\phi_0)\,dR\,dZ,
$$

符号由截面定向和磁场方向确定。

对多个 $s_0$ 做二维数值积分，即得到离散关系

$$
s_k\mapsto\Psi_T(s_k).
$$

随后使用满足

$$
\Psi_T(0)=0,
\qquad
\frac{d\Psi_T}{ds}>0
$$

或固定统一负号方向的单调样条进行一维拟合。

### 3.2 必须做多个环向截面

若 $s$ 是精确磁通函数且 $\nabla\cdot\boldsymbol B=0$，同一磁面的通量积分应与
$\phi_0$ 无关。实际应在多个环向截面上计算

$$
\Psi_T(s_k,\phi_j),
$$

并检查

$$
\epsilon_\Psi(s_k)
=\frac{\operatorname{std}_{\phi}
\Psi_T(s_k,\phi)}
{\left|\operatorname{mean}_{\phi}
\Psi_T(s_k,\phi)\right|}.
$$

这个量不是普通数值误差；它直接衡量当前 $s$ 等值面是否足够接近真实磁面。

### 3.3 标定不能修复磁面形状

因为

$$
\nabla\psi_T=f'(s)\nabla s,
$$

所以

$$
\boldsymbol B\cdot\nabla\psi_T
=f'(s)\boldsymbol B\cdot\nabla s.
$$

单调重标定只能修正 psi 的物理尺度，不能改变等值面的几何位置，也不能把一个
原本非磁面的等值面变成精确磁面。

因此在 alpha 拟合前必须先通过：

- $\boldsymbol B\cdot\nabla s$；
- 多截面 $\Psi_T$ 一致性；
- Poincare return；
- 相邻 level-set 不交叉；

确认 $s$ 的几何质量。否则 alpha 最小二乘存在不可消除的 residual 下界。

## 4. alpha 的正确拓扑形式

### 4.1 alpha 不是普通周期函数

对于具有非零环向和极向绕数的磁场，alpha 是多值 Clebsch 势。选定几何极向角
$\theta$ 和环向角 $\zeta=\phi$ 后，应写成

$$
\alpha(\rho,\theta,\zeta)
=\theta
+\lambda(\rho,\theta,\zeta)
-\iota(\rho)\zeta.
$$

其中只有 lambda 是 $\theta,\zeta$ 双周期函数。

若直接把 alpha 展开成普通双周期 Fourier 级数并最小化

$$
\boldsymbol B\cdot\nabla\alpha,
$$

则常数 alpha 就是没有物理意义的零 residual 解。因此必须固定
$\theta$ 的拓扑系数为 1，并显式保留 $-\iota\zeta$ 的世俗项。

### 4.2 gauge

变换

$$
\alpha\longrightarrow\alpha+C(\psi_T)
$$

不改变

$$
\nabla\psi_T\times\nabla\alpha.
$$

因此必须删除 lambda 的纯 flux-function mode，或施加

$$
\left\langle\lambda\right\rangle_{\theta,\zeta}=0.
$$

还必须固定 theta 和 zeta 的方向以及整数绕数 convention，否则 iota 可以在
符号或整数角变换下改变表示。

## 5. 稠密线性最小二乘如何构造

### 5.1 alpha 展开

令 $f_k(\rho,\theta,\zeta)$ 为满足轴正则性和双周期性的基函数，令
$p_j(\rho)$ 为 iota 的径向基函数。写成

$$
\lambda
=\sum_k c_k f_k(\rho,\theta,\zeta),
$$

$$
\iota(\rho)
=\sum_j a_j p_j(\rho).
$$

则

$$
\alpha
=\theta
+\sum_k c_k f_k
-\zeta\sum_j a_j p_j.
$$

### 5.2 为什么对未知系数仍是线性的

代入 Clebsch 表示：

$$
\boldsymbol B
=\nabla\psi_T\times\nabla\theta
+\sum_k c_k
\nabla\psi_T\times\nabla f_k
-\sum_j a_j p_j(\rho)
\nabla\psi_T\times\nabla\zeta.
$$

梯度

$$
\nabla\left[zeta p_j(\rho)\right]
=p_j\nabla\zeta
+\zeta p_j'(\rho)\nabla\rho
$$

中的第二项与 $\nabla\psi_T$ 平行，所以在叉乘中自动消失。这正是 iota 可以与
lambda 一起线性拟合的原因。

对每个物理采样点 $q$，定义

$$
\boldsymbol b_q
=\boldsymbol B_q
-\nabla\psi_{T,q}\times\nabla\theta_q.
$$

lambda 列向量为

$$
\boldsymbol A_{qk}^{(\lambda)}
=\nabla\psi_{T,q}\times\nabla f_{k,q},
$$

iota 列向量为

$$
\boldsymbol A_{qj}^{(\iota)}
=-p_j(\rho_q)
\nabla\psi_{T,q}\times\nabla\zeta_q.
$$

于是三个磁场分量共同组成

$$
\boldsymbol A
\begin{bmatrix}
\boldsymbol c\\
\boldsymbol a
\end{bmatrix}
\simeq
\boldsymbol b.
$$

这就是一个标准线性最小二乘问题。

### 5.3 建议的目标归一化

不同半径的 $|\boldsymbol B|$ 和 $|\nabla\psi_T|$ 差异会造成权重偏置。建议
最小化相对磁场残差

$$
\mathcal L_B
=\sum_q w_q
\frac{
\left|
\boldsymbol B_q
-\nabla\psi_{T,q}\times\nabla\alpha_q
\right|^2
}{|\boldsymbol B_q|^2+B_{\mathrm{floor}}^2}.
$$

其中 $w_q$ 应对应物理体积 quadrature，而不是简单地让每个 rho 层权重相同。

### 5.4 普通 LS 还是凸约束 LS

若只做普通线性最小二乘，alpha 的场重构 residual 可能很小，但仍不能严格保证
角度映射可逆。需要检查

$$
\frac{\partial\vartheta}{\partial\theta}
=1+\frac{\partial\lambda}{\partial\theta}>0,
$$

其中

$$
\vartheta=\theta+\lambda.
$$

因为

$$
1+\sum_k c_k\frac{\partial f_k}{\partial\theta}
\ge\delta
$$

对 $c_k$ 是线性不等式，所以可以把同一个问题写成凸 quadratic programming。
它不是非线性方程，也没有局部极小值问题；数值上仍应比 DESC solve 简单、稳定。

建议将“一个线性最小二乘”的目标稍作修正为：**一个稠密、带线性约束的凸
最小二乘阶段。**

## 6. 是否必须先有 R、Z

### 6.1 拟合 alpha 本身不必先拟合 DESC R/Z

如果已经知道：

- 物理点 $(R,\phi,Z)$；
- 标定后的 $\psi_T(R,\phi,Z)$ 和梯度；
- 磁轴；
- 几何角

  $$
  \theta_g
  =\operatorname{atan2}
  \left(Z-Z_{\mathrm{axis}},R-R_{\mathrm{axis}}\right);
  $$

那么可以直接在物理空间计算
$\rho,\theta_g,\zeta$ 和所有 basis gradient，拟合 alpha。这个阶段不需要先把
R、Z 投影到 DESC 谱。

这反而能避免当前问题：先做有毫米误差的 R/Z 谱拟合，再用已经偏离原始磁面的
谱几何计算 lambda。

### 6.2 但进入 DESC 前仍需要 R、Z 体映射

alpha 只给出“一个物理点位于哪条场线、对应什么磁角”，并不单独决定该点的
空间位置。DESC 最终仍需要

$$
R(\rho,\theta,\zeta),
\qquad
Z(\rho,\theta,\zeta).
$$

这些几何信息来自 psi level sets，而不是来自 alpha 本身。因此仍需：

1. 对给定 $\rho,\theta_g,\zeta$，在相对磁轴的射线上解

   $$
   \psi_T(R,Z,\zeta)
   =\rho^2\psi_{T,\mathrm{edge}};
   $$

2. 得到结构化的 R/Z tensor grid；
3. 将 R、Z 线性投影到 DESC 的 Fourier-Zernike basis。

这里仍有一维 Newton 或 bisection，但它是每条射线上独立、单调的标量求根，
不是复杂耦合的非线性拟合。对 cem_qh03 已验证该求根稳定。

## 7. alpha 之后如何组织 DESC 坐标

存在两条可选路线。

### 7.1 路线 A：保留几何 theta，显式输出 lambda

使用

$$
R=R(\rho,\theta_g,\zeta),
\qquad
Z=Z(\rho,\theta_g,\zeta),
$$

并把 alpha 拟合得到的

$$
\lambda(\rho,\theta_g,\zeta)
$$

直接投影到 DESC `L_basis`。

优点：

- R/Z level-set 提取最简单；
- 与当前 DESC 数据结构直接对应；
- alpha LS 直接给出 `L_lmn`。

风险：

- lambda 可能较大；
- 必须严格检查
  $1+\partial_\theta\lambda>0$；
- DESC solve 仍需同时调整 R/Z/L。

### 7.2 路线 B：用 alpha 重参数化 R/Z，使初始 lambda 接近零

定义直线场极向角

$$
\vartheta
=\theta_g+\lambda.
$$

若对每个 $\rho,\zeta$ 都满足

$$
\frac{\partial\vartheta}{\partial\theta_g}>0,
$$

则可将同一物理磁面从 $\theta_g$ 重采样到均匀的 $\vartheta$ 网格，得到

$$
R(\rho,\vartheta,\zeta),
\qquad
Z(\rho,\vartheta,\zeta).
$$

然后把 DESC 的计算极向角直接选成 $\vartheta$，初始可取

$$
\lambda_{\mathrm{DESC}}\simeq0.
$$

这条路线的潜在优势很大：lambda 的复杂性被吸收到 R/Z 的角度参数化中，避免
把一个接近可逆性边界的大 lambda 直接交给 DESC。

它仍需要每个磁面上的单调一维插值，但不需要新的全局非线性方程。建议将路线 B
作为主要实验，路线 A 作为交叉验证；两者描述的物理曲面应一致。

## 8. 一次 alpha 拟合后可以直接得到什么

在 psi 标定和 alpha 拟合成功后，可以得到：

1. 物理磁通标签 $\psi_T$ 和 DESC 径向坐标 $\rho$；
2. Clebsch 场线标签 $\alpha$；
3. 旋转变换 profile $\iota(\rho)$；
4. 周期坐标修正 $\lambda(\rho,\theta,\zeta)$；
5. 直线场角 $\vartheta=\theta+\lambda$；
6. 每个物理点的磁坐标 $(\rho,\alpha,\zeta)$ 或
   $(\rho,\vartheta,\zeta)$；
7. 磁场的 Clebsch 重构

   $$
   \boldsymbol B_{\mathrm{fit}}
   =\nabla\psi_T\times\nabla\alpha;
   $$

8. iota、lambda 的独立局部 residual 和坐标可逆性诊断；
9. 用于 DESC 的 R/Z/L 初值，或重参数化后的 R/Z 与近零 L 初值。

但不能只靠 alpha 得到：

1. 完整 Boozer 坐标。Boozer 坐标还要求协变磁场分量为 flux functions，通常
   还需要环向角修正；
2. 压力或等离子体电流 profile。当前外部线圈场对应真空区域，但 DESC 的 profile
   约束仍需明确；
3. 已经满足离散 DESC 方程的 MHD 平衡；
4. 在磁岛或混沌区不存在的全局光滑磁坐标；
5. 不经任何几何投影就得到的 DESC R/Z 谱系数。

## 9. 该方案可能遗漏的关键问题

### 9.1 alpha 的全局存在性

在良好的 irrational 嵌套磁面上，alpha 的周期部分通常存在并在 gauge 后唯一。
在有理面上，需要闭合场线积分的可解性条件；靠近共振时会出现小除数，lambda
可能需要很高阶或变得不光滑。

因此 alpha LS residual 随 rho 的尖峰本身就是磁岛、共振或 psi 误差的诊断，
不能一律靠正则压掉。

### 9.2 轴附近奇异性

$\theta_g$ 在磁轴上没有定义，且
$|\nabla\psi_T|\to0$。alpha 的拓扑角结构在轴附近需要使用满足 Zernike/Cartesian
正则性的 basis，不能把大量不同 theta 的近轴点当作普通独立数据。

建议：

- 训练时排除极小的 $\rho<\rho_{\min}$；
- 用解析轴正则性补充约束；
- 使用按体积衰减的权重；
- 单独验证 $\rho\to0$ 的 mode scaling。

### 9.3 psi 误差造成不可约 residual

任何 Clebsch 重构

$$
\nabla\psi_T\times\nabla\alpha
$$

都严格切于 psi level set。因此原磁场的法向分量

$$
B_n
=\frac{\boldsymbol B\cdot\nabla\psi_T}
{|\nabla\psi_T|}
$$

不可能由 alpha 修复。alpha LS 的最小 residual 至少包含这部分误差。

应把总 residual 分解为：

$$
\boldsymbol B
=\boldsymbol B_\perp
+\boldsymbol B_\parallel,
$$

其中 $\boldsymbol B_\perp$ 是相对 psi 面的法向分量。先报告
$\|\boldsymbol B_\perp\|$，再判断 alpha 对切向分量的拟合质量，否则会把 psi
问题误判为 alpha 阶数不足。

### 9.4 基底与小除数

即使场和磁面都光滑，Fourier mode 仍可能包含

$$
\frac{1}{m\iota-n}
$$

形式的小除数放大。不能只看训练 residual 决定升阶；必须同时检查谱尾、导数
极值、独立验证 residual 和 rho 连续性。

### 9.5 稠密点并不等于正确权重

若在 rho 上均匀分层、每层角点相同，仍会过采样轴附近。建议令

$$
s_\rho=\rho^2
$$

均匀，或直接按

$$
w_q\propto|\sqrt g_q|
$$

使用物理体积 quadrature。训练点和验证点必须独立错位。

## 10. 建议的完整流程

### 阶段 A：几何不变量

1. 用现有稠密方法拟合 $s(R,Z,\phi)$；
2. 检查嵌套性、$\boldsymbol B\cdot\nabla s$ 和 Poincare；
3. 若磁面几何不合格，停止，不进入 alpha。

### 阶段 B：物理磁通标定

1. 在多个 $s_k$ 和多个 $\phi_j$ 上提取闭合截面；
2. 计算 $\Psi_T(s_k,\phi_j)$；
3. 检查环向截面一致性；
4. 拟合单调 $\Psi_T(s)$；
5. 定义

   $$
   \psi_T=\Psi_T/(2\pi),
   \qquad
   \rho=\sqrt{\psi_T/\psi_{T,\mathrm{edge}}}.
   $$

### 阶段 C：稠密 alpha 拟合

1. 在近似均匀物理体积的训练网格上采样
   $\boldsymbol B,\psi_T,\nabla\psi_T,\theta_g,\zeta$；
2. 使用固定拓扑项

   $$
   \alpha=\theta_g+\lambda-\iota\zeta;
   $$

3. 对 $L_{lmn}$ 和 iota profile 系数做批量线性 LS；
4. 加入 gauge 和

   $$
   1+\partial_\theta\lambda\ge\delta;
   $$

5. 在独立密网格上验证磁场重构、iota、谱尾和可逆性。

### 阶段 D：构造 DESC 几何

优先尝试：

1. 从 psi level sets 得到几何 theta 网格上的 R/Z；
2. 用 $\vartheta=\theta_g+\lambda$ 将每个面重参数化；
3. 在线性谱投影中得到

   $$
   R(\rho,\vartheta,\zeta),
   \qquad
   Z(\rho,\vartheta,\zeta);
   $$

4. 初始设置较小或零的 DESC lambda；
5. 将 fixed boundary 直接同步为拟合体的 $\rho=1$ surface。

同时保留“几何 theta R/Z + 显式 L”路线作为一致性对照。

### 阶段 E：受限 DESC continuation

即使 alpha 重构很好，仍建议从低分辨率和小 trust radius 开始，逐步提高
分辨率和边界半径，并在每一步检查 Jacobian margin。

## 11. 建议的采样与分辨率

第一版不必立即复制 psi 的 389,760 点，但应达到数万级并做收敛扫描：

| 用途 | 建议网格 | 点数 |
|---|---:|---:|
| alpha 初始训练 | $25\times32\times32$ | 25,600 |
| alpha 主训练 | $33\times48\times48$ | 76,032 |
| 独立验证 | $48\times64\times64$ | 196,608 |

径向点在 $\rho^2$ 上均匀，角向网格覆盖完整周期。若 GPU 批量法与 psi 拟合
一样成熟，可再提升到约 30--40 万点验证系数稳定性。

阶数建议分开扫描：

- iota profile：先用低阶、单调或弱剪切径向 basis；
- lambda 径向阶：$L=4,6,8,10$；
- lambda 角向阶：先固定 $M=N=6$，再测试 8 和 10；
- 每次升阶都检查训练/验证差距和最高 mode 能量。

## 12. 成功判据

只有同时满足以下条件，才认为 alpha 阶段可替代当前 fieldline phase LS：

1. 物理磁通标定在不同 $\phi$ 截面上相对一致；
2. psi 法向磁场误差单独可接受；
3. 切向磁场 Clebsch 重构 residual 在训练和验证网格上接近；
4. iota 对采样密度、阶数和网格偏移稳定；
5. iota 与长时间 Poincare rotation number 在统一 convention 后一致；
6. lambda 的最高 mode 能量衰减；
7. 在独立密网格上满足

   $$
   1+\partial_\theta\lambda\ge\delta>0;
   $$

8. 路线 A 与路线 B 得到相同物理 R/Z 曲面；
9. 构造出的 DESC R/Z 在密网格上有有限 Jacobian margin；
10. DESC 初始 force 和直接 solve 的第一步不再通过 Jacobian 过零降低 residual。

## 13. 最终判断

该方案不是不可行的“口胡”，而是比当前 phase 点云 LS 更接近磁坐标理论的方案。
其核心数学结构正确，并且在 psi 固定后确实允许将 alpha、lambda 和 iota 的
求解写成一个大规模线性问题。

真正需要补齐的是：

1. 物理磁通标定；
2. alpha 的多值拓扑项和 gauge；
3. psi 法向误差与 alpha 切向误差的分离；
4. 均匀体积采样和独立验证；
5. 坐标可逆性的线性硬约束；
6. alpha 坐标到 DESC R/Z 谱映射的最后一步。

如果这些环节按上述方式实现，后半段可以从当前的“场线追踪 + beta/iota/lambda
联合点云拟合 + 多次补丁”简化为：

$$
\boxed{
\text{psi 几何}
\rightarrow
\text{物理磁通标定}
\rightarrow
\text{稠密凸 alpha 拟合}
\rightarrow
\text{结构化 R/Z 投影}
\rightarrow
\text{DESC continuation}
}
$$

其中只有 level-set 提取和最终 DESC 平衡仍涉及非线性过程；alpha、iota、lambda
以及 R/Z 谱系数的主要恢复步骤都可以保持为线性或凸问题。
