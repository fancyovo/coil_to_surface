# DESC 求解问题与更好初值方案分析

## 1. 结论摘要

DESC 默认求的不是“给定线圈磁场中的一张磁面”，而是一个 **fixed-boundary ideal-MHD equilibrium**：

$$
\mathbf J\times \mathbf B=\nabla p
$$

在给定外边界、总环向通量和 profile 后，DESC 在整个体积内求一组嵌套通量坐标

$$
(\rho,\theta,\zeta)\mapsto (R(\rho,\theta,\zeta),Z(\rho,\theta,\zeta),\phi(\rho,\theta,\zeta))
$$

以及 straight-field-line 坐标修正 $\lambda$。我们的线圈场目前只用于：

- 找磁轴；
- 拟合 $\psi$；
- 提取候选磁面；
- 算 boundary toroidal flux；
- 给 DESC 一个边界和初值。

DESC 的 force objective 本身并不会调用外部线圈的 Biot-Savart 场。因此 DESC 失败不能直接说明线圈磁场中没有磁面；它也可能说明我们给 DESC 的全体积初值、profile、坐标或者 fixed-boundary 问题设置还不够好。

当前最值得做的主动初值改进是：不只用 $\psi$ 给出多层几何面，还要从磁力线追踪中构造近似 straight-field-line 坐标，也就是给 DESC 一个非零且合理的 $\lambda(\rho,\theta,\zeta)$ 初值，并给出更合理的 $\iota(\rho)$ 初值或 profile。

## 2. DESC 默认在求什么

基于远端已安装的 DESC 0.16.0 源码：

- `desc/equilibrium/equilibrium.py:2216`：`Equilibrium.solve(...)`
- `desc/objectives/getters.py:51`：`get_equilibrium_objective(...)`
- `desc/objectives/_equilibrium.py:13`：`ForceBalance`
- `desc/objectives/getters.py:128`：`get_fixed_boundary_constraints(...)`
- `desc/equilibrium/initial_guess.py:23`：`set_initial_guess(...)`

`eq.solve()` 的默认参数是：

```python
eq.solve(objective="force", constraints=None, optimizer="lsq-exact")
```

如果 `constraints=None`，DESC 会自动使用 fixed-boundary constraints：

```python
constraints = (
    FixBoundaryR,
    FixBoundaryZ,
    FixPsi,
    fixed profiles,
    FixSheetCurrent,
)
```

也就是说，默认固定：

- LCFS 边界的 $R_{mn}$ Fourier 系数；
- LCFS 边界的 $Z_{mn}$ Fourier 系数；
- 总环向磁通 `Psi`；
- 已指定的 pressure/current/iota 等 profiles；
- sheet current 相关自由度。

优化变量主要是 DESC equilibrium 的谱系数：

- `R_lmn`：体积内 $R(\rho,\theta,\zeta)$ 的 Fourier-Zernike 系数；
- `Z_lmn`：体积内 $Z(\rho,\theta,\zeta)$ 的 Fourier-Zernike 系数；
- `L_lmn`：$\lambda(\rho,\theta,\zeta)$ 的 Fourier-Zernike 系数；
- 若 profile 未固定，也可能包含 profile 系数。

## 3. `force` objective 的具体残差

`objective="force"` 会构造 `ForceBalance`。它不是直接最小化三维向量 $\mathbf J\times\mathbf B-\nabla p$ 的所有笛卡尔分量，而是使用两个等价的通量坐标残差：

$$
F_\rho
$$

和

$$
F_{\mathrm{helical}}.
$$

源码中的定义是：

$$
F_\rho=\frac{(\nabla\times\mathbf B\times\mathbf B)_\rho}{\mu_0}-p_\rho
$$

以及

$$
F_{\mathrm{helical}}=\frac{\partial_\theta B_\zeta-\partial_\zeta B_\theta}{\mu_0}.
$$

`ForceBalance.compute()` 最终返回的是加权后的残差向量：

$$
\left[
F_\rho |\nabla\rho|\sqrt g,\quad
F_{\mathrm{helical}} |e^{\mathrm{helical}}\sqrt g|
\right].
$$

其中 DESC 内部磁场不是外部线圈场，而是由 equilibrium 几何、通量、$\iota$ 或 current profile、以及 $\lambda$ 构造出来的通量坐标磁场。源码中有：

$$
B^\rho=0,
$$

$$
B^\theta=\frac{\psi_\rho}{\sqrt g}\left(\iota\,\phi_\zeta-\lambda_\zeta\right),
$$

$$
B^\zeta=\frac{\psi_\rho}{\sqrt g}\left(-\iota\,\omega_\theta+\theta_{\mathrm{PEST},\theta}\right).
$$

这说明 DESC 假定磁场已经在嵌套通量坐标中，且 $\lambda$ 负责把几何角修正到 straight-field-line/PEST 角。我们若传入 `lambda=0`，相当于假定当前几何角已经足够接近 straight-field-line 坐标；对复杂三维面这通常不成立。

## 4. 这和我们的 evaluator 在解的问题有什么不同

我们的主流程先解的是外部线圈磁场中的近似不变量：

$$
\mathbf B_{\mathrm{coil}}\cdot\nabla\psi\approx 0.
$$

这一步只要求存在一个函数 $\psi$，它的等值面近似被外部线圈场切向穿过。它是一个 **磁面识别/重构问题**。

DESC 求的是给定边界内的全体积 MHD 平衡。它要求更强：

1. 从磁轴到边界都能用一组光滑嵌套通量坐标表示；
2. 在这个体积中由 DESC 构造的 $\mathbf B$ 满足 force balance；
3. 指定的 pressure/current/iota profile 与几何和边界兼容；
4. $\lambda$ 能把坐标修正为 straight-field-line 坐标；
5. 边界、通量和 profile 共同定义的问题本身有可解的平衡分支。

因此：

- $\psi$ 磁面好，不必然意味着 DESC fixed-boundary equilibrium 好解；
- DESC 失败，也不必然意味着外部线圈磁场没有磁面；
- DESC 成功，也不保证它对应原外部线圈场，除非再检查 boundary normal field、Poincare、$|B|$、QS 等。

## 5. 为什么当前初值还不够接近

目前我们给 DESC 的增强初值主要是：

```python
eq.set_initial_guess(nodes, R, Z, lambda=0)
```

其中 `nodes` 是 $(\rho,\theta,\zeta)$，`R,Z` 来自多层 $\psi=\psi_0\rho^2$ 等值面。DESC 源码里 `_initial_guess_points(...)` 的行为是：

1. 用这些点建立一个 `Grid`；
2. 对 $R$、$Z$、$\lambda$ 分别做谱拟合；
3. 得到 `R_lmn/Z_lmn/L_lmn`。

这只是一个线性几何拟合，不含力平衡信息。它的问题是：

- 几何面可能嵌套，但 $\theta$ 标签未必是 straight-field-line 标签；
- $\lambda=0$ 通常是坏假设；
- $\iota(\rho)$ 没有被主动拟合到外部线圈磁力线结果；
- 内层和外层的角度参数化可能不一致；
- 若用多个 variant 复用同一个 DESC surface 对象，还会出现边界污染问题，这一点已经单独确认并需要修复。

对 `cem_qh03`，修正 surface 污染后，边界尺度正常，但 DESC 初始 force residual 仍很大，solve 后非嵌套。这更像是“初值/profile/坐标离 DESC 平衡分支还很远”，而不是“线圈场中完全没有磁面”的直接证据。

## 6. 能否主动构造更接近的初值

可以，而且应该分层做。核心思想是：像解 $\psi$ 一样，先在外部线圈磁场上解出 DESC 需要的几何和坐标信息，再交给 DESC。

### 6.1 几何初值：多层 $\psi$ 面

这是已经部分完成的方案。

给定最大磁面 $\psi_{\mathrm{edge}}$，取多层

$$
\psi_\ell=\rho_\ell^2\psi_{\mathrm{edge}},
$$

在每个 $\zeta$ 截面上用一维 Newton 解等值线，得到

$$
R(\rho_\ell,\alpha,\zeta),\quad Z(\rho_\ell,\alpha,\zeta).
$$

这里 $\alpha$ 是几何极角或射线角。然后拟合 DESC 的 `R_lmn/Z_lmn`。

这一步能让 DESC 初始体坐标几何上接近我们找到的磁面族，但还不够。

### 6.2 坐标初值：从磁力线构造 straight-field-line 角

更关键的是构造 $\lambda$。

对每一层 $\rho_\ell$：

1. 在 $\zeta=0$ 的 $\psi=\psi_\ell$ 曲线上选一批起点；
2. 同时追踪这些点到下一个场周期；
3. 得到 Poincare return map：

   $$
   F_\ell:\alpha\mapsto\alpha'
   $$

4. 解一个圆周共轭问题：

   $$
   h_\ell(F_\ell(\alpha))=h_\ell(\alpha)+\Delta_\ell,
   $$

   其中

   $$
   \Delta_\ell=\frac{2\pi\iota(\rho_\ell)}{\mathrm{nfp}}.
   $$

5. 用 $h_\ell$ 定义该层的近似 straight-field-line 角：

   $$
   \vartheta=h_\ell(\alpha).
   $$

DESC 的 $\lambda$ 关系可理解为：

$$
\vartheta=\theta+\lambda.
$$

如果我们把 DESC 节点角 $\theta$ 暂时取为几何角 $\alpha$，那么可以给出初值：

$$
\lambda(\rho_\ell,\alpha,\zeta)\approx \vartheta(\rho_\ell,\alpha,\zeta)-\alpha.
$$

更完整时，还需要沿每条磁力线推进相位：

$$
\vartheta(\zeta)\approx \vartheta(0)+\iota(\rho_\ell)\zeta
$$

再把实际追踪点插回各 $\zeta$ 截面，得到全三维的 $\lambda$ 点云。

这一步的意义很大：它不仅给出几何面，还给 DESC 一个接近 straight-field-line 的角坐标。相比 `lambda=0`，这更接近 DESC 内部磁场表示。

### 6.3 主动拟合 $\iota(\rho)$

同一批磁力线追踪会自然给出每层的旋转变换：

$$
\iota(\rho_\ell)\approx \frac{\mathrm{nfp}}{2\pi}\Delta_\ell.
$$

可以把这些值拟合成一个低阶 radial profile：

$$
\iota(\rho)=\iota_0+\iota_2\rho^2+\iota_4\rho^4+\cdots
$$

然后有两种策略：

- 继续给 DESC `current=0`，只把 $\iota(\rho)$ 当诊断和初值参考；
- 或者改为给 DESC 指定 `iota` profile，而不是 current profile。

第二种会让 DESC 的磁力线 pitch 更接近外部线圈磁场，但物理解释要谨慎：它不再是严格的“零电流 profile”问题，而是指定旋转变换的 fixed-boundary equilibrium 问题。是否适合最终评估，需要单独比较。

### 6.4 continuation 初值

另一个稳健方向是从小面开始做 continuation：

1. 选一个很小的 $\psi_{\mathrm{edge}}$，DESC 较容易求解；
2. 得到一个 nested 且 force residual 小的 equilibrium；
3. 把它作为下一层更大边界的 initial guess；
4. 逐步扩大到目标边界。

这类似我们在 Boozer 面里用上一层的 iota 作为下一层初值。它的优点是 DESC 每一步只走小变形，不容易从好坐标跳到坏坐标。缺点是耗时更长，并且如果某个半径外真实拓扑已经变差，它会自然失败。

### 6.5 用 GoodCoordinates 做预处理

DESC 自带 `GoodCoordinates` 目标，可以尝试修正坐标嵌套性。它适合作为 force solve 前的预处理，但不能代替物理初值：

- 它主要改善坐标映射；
- 不保证 force residual 小；
- 不保证对应外部线圈磁场。

所以它应该放在多层 $\psi$ 几何和 $\lambda$ 初值之后，作为清理步骤，而不是第一主力。

## 7. 推荐的 DESC 初值生成路线

建议优先级如下。

### A. 必须先修的工程问题

1. 每个 DESC variant 都重新构造 fresh surface，禁止复用会被原地修改的 surface 对象。
2. DESC `L/M/N` 不低于输入边界实际 `mpol/ntor`。
3. 每次 `Equilibrium(...)` 后立即检查：

   $$
   \max |R_{\mathrm{eq.surface}}-R_{\mathrm{input}}|,
   \quad
   \max |Z_{\mathrm{eq.surface}}-Z_{\mathrm{input}}|.
   $$

4. force residual、boundary fidelity、`is_nested()` 必须同时报告。

### B. 第一版主动初值

1. 用 $\psi$ 提取 8 到 16 层嵌套面；
2. 用这些点拟合 `R_lmn/Z_lmn`；
3. 对每层追踪磁力线，估计 $\iota(\rho)$；
4. 用 Poincare map 共轭构造 $\lambda$；
5. 调用：

   ```python
   eq.set_initial_guess(nodes, R, Z, lambda_values, ensure_nested=False)
   ```

6. 先只评估初始 force residual 和 nested，不立刻 solve；
7. 若初始 residual 明显下降，再跑 DESC solve。

### C. 第二版 continuation

1. 从小 $\psi_{\mathrm{edge}}$ 开始；
2. 每层都用上一层 equilibrium 作为初值；
3. 每次只增加一点边界大小；
4. 记录第一层失败位置。

这个能区分两类问题：

- 若小半径也失败，多半是 DESC 初值/坐标/输入设置问题；
- 若小半径成功、大半径失败，可能是边界太大、拓扑变差或 fixed-boundary profile 不兼容。

## 8. 对“位型坏还是 DESC 不鲁棒”的当前判断

目前不能把 DESC 失败直接归因于磁场位型坏。

更准确的判断是：

1. evaluator 已经在外部线圈场中找到候选磁面，并且 Boozer 面能在一定半径上收敛；这说明至少局部磁面结构不是完全坏的。
2. DESC fresh run 修正边界污染后，边界尺度正常，但初始 force residual 很大；说明当前传入的体初值仍不接近 DESC 的 MHD 平衡。
3. `lambda=0` 是明显薄弱环节；对复杂三维磁面，几何角通常不是 straight-field-line 角。
4. DESC fixed-boundary solve 对初值和 profile 很敏感；它不是“任意线圈边界都能自动找到平衡”的鲁棒黑盒。
5. 如果后续用主动 $\lambda/\iota$ 初值和 continuation 仍失败，再结合 Poincare、$B\cdot n$ 分布和半径扫描，才更有依据说该边界对应的体积内没有合适嵌套平衡。

因此当前优先假设应是：

> DESC 失败主要来自初值坐标和 fixed-boundary equilibrium 设置不够接近，而不是已经证明线圈位型本身没有磁面。

## 9. 下一步实验建议

建议做一个小规模验证，不直接大改主流程：

1. 选 `cem_qh03` 当前最大边界和一个更小边界；
2. 对每个边界生成两类初值：
   - `psi_geometry_lambda0`：多层 $\psi$ 几何，$\lambda=0$；
   - `psi_geometry_sfl_lambda`：多层 $\psi$ 几何 + 磁力线共轭得到的 $\lambda$；
3. 对比 solve 前：
   - `is_nested()`;
   - mean/max normalized force；
   - boundary fidelity；
   - `sqrt(g)_PEST` 符号分布；
4. 如果 `sfl_lambda` 明显降低初始 force，再跑 DESC solve；
5. 若小边界能成功，继续做 boundary continuation。

这组实验能直接回答：我们主动构造更接近的初值是否真的有收益。

## 10. 直接拟合 $R,Z,\lambda$ 的可行性

这一节从“能不能像拟合 $\psi$ 一样，直接用线性最小二乘拟合出 DESC 的 $R,Z,\lambda$ 初值”这个角度重新分析。

结论是：

- $R,Z$ 的几何拟合可以直接写成线性最小二乘；
- $\lambda$ 不能仅由几何面唯一确定，因为它本质上是 poloidal label 的坐标修正；
- 但如果额外要求近似 straight-field-line，那么在 $R,Z$ 几何已经固定后，$\lambda$ 和 $\iota(\rho)$ 可以写成一个线性最小二乘问题；
- 因此可行方案不是“一步线性解完整 MHD equilibrium”，而是“先线性拟合几何，再线性拟合坐标修正，最后把这个更好的初值交给 DESC”。

### 10.1 DESC 中 $R,Z,\lambda$ 的展开形式

DESC 的体坐标可以抽象写成

$$
R(\rho,\theta,\zeta)=\sum_j c^R_j \Phi^R_j(\rho,\theta,\zeta),
$$

$$
Z(\rho,\theta,\zeta)=\sum_j c^Z_j \Phi^Z_j(\rho,\theta,\zeta),
$$

$$
\lambda(\rho,\theta,\zeta)=\sum_j c^\lambda_j \Phi^\lambda_j(\rho,\theta,\zeta).
$$

这里 $\Phi_j$ 是 DESC 的 Fourier-Zernike 基函数。若我们已经有点云

$$
(\rho_i,\theta_i,\zeta_i)\mapsto (R_i,Z_i),
$$

那么 $R,Z$ 初值就是标准线性问题：

$$
\sum_j c^R_j \Phi^R_j(\rho_i,\theta_i,\zeta_i)=R_i,
$$

$$
\sum_j c^Z_j \Phi^Z_j(\rho_i,\theta_i,\zeta_i)=Z_i.
$$

矩阵形式为

$$
A_R c^R \simeq R,\qquad A_Z c^Z \simeq Z.
$$

这正是 DESC 的 `set_initial_guess(nodes, R, Z, lambda)` 对点云做的事情：它分别对 $R,Z,\lambda$ 做谱拟合。区别在于，我们可以更主动地选择点云、权重、边界约束和角度标签，而不是把几何角直接塞进去。

### 10.2 只靠几何不能唯一确定 $\lambda$

需要特别注意：同一个物理磁面族可以用很多不同的 poloidal angle 标记。若把

$$
\theta\mapsto \tilde\theta(\rho,\theta,\zeta)
$$

换成另一个光滑单调角标，$R,Z$ 表示会改变，$\lambda$ 也会改变，但物理几何面不变。

因此，仅从“这些点构成嵌套曲面”这个纯几何事实，不能唯一推出 $\lambda$。最多能给出一些坐标质量约束，例如：

- 曲面不自交；
- $\sqrt g$ 不变号；
- $\theta$ 网格尽量正交或均匀；
- $\lambda$ 平均值为零作为 gauge。

这些约束可以让坐标好看，但不能保证 straight-field-line。要让 $\lambda$ 接近 DESC 想要的含义，必须加入磁场方向信息。

DESC 源码中有注释：

```python
# Assumes theta = vartheta - lambda.
```

也就是

$$
\vartheta=\theta+\lambda,
$$

其中 $\vartheta$ 是近似 PEST/straight-field-line poloidal angle。因此 $\lambda$ 的物理任务是把当前计算角 $\theta$ 修正到 straight-field-line 角。

### 10.3 用磁力线直线化条件拟合 $\lambda$

如果 $R,Z$ 几何已经通过多层 $\psi$ 面拟合好，那么在这个几何映射上可以计算协变基和逆变基：

$$
\mathbf x_\rho,\quad \mathbf x_\theta,\quad \mathbf x_\zeta,
$$

以及

$$
\nabla\rho,\quad \nabla\theta,\quad \nabla\zeta.
$$

对外部线圈磁场 $\mathbf B_{\mathrm{coil}}$，straight-field-line 条件可以写为：

$$
\mathbf B_{\mathrm{coil}}\cdot\nabla\vartheta
=
\iota(\rho)\,\mathbf B_{\mathrm{coil}}\cdot\nabla\zeta.
$$

因为

$$
\vartheta=\theta+\lambda,
$$

所以

$$
\mathbf B_{\mathrm{coil}}\cdot\nabla\lambda
-
\iota(\rho)\,\mathbf B_{\mathrm{coil}}\cdot\nabla\zeta
=
-\mathbf B_{\mathrm{coil}}\cdot\nabla\theta.
$$

把

$$
\lambda=\sum_j c^\lambda_j\Phi^\lambda_j
$$

代入，得到

$$
\sum_j c^\lambda_j\,
\mathbf B_{\mathrm{coil}}\cdot\nabla\Phi^\lambda_j
-
\iota(\rho)\,\mathbf B_{\mathrm{coil}}\cdot\nabla\zeta
=
-\mathbf B_{\mathrm{coil}}\cdot\nabla\theta.
$$

如果先用磁力线追踪估计好 $\iota(\rho)$，这就是对 $c^\lambda_j$ 的线性方程。

更进一步，如果令

$$
\iota(\rho)=\sum_k t_k P_k(\rho),
$$

其中 $P_k$ 是低阶径向基，例如 $1,\rho^2,\rho^4,\ldots$，则

$$
\sum_j c^\lambda_j\,
\mathbf B_{\mathrm{coil}}\cdot\nabla\Phi^\lambda_j
-
\sum_k t_k P_k(\rho)\,
\mathbf B_{\mathrm{coil}}\cdot\nabla\zeta
=
-\mathbf B_{\mathrm{coil}}\cdot\nabla\theta.
$$

这对未知量

$$
(c^\lambda_1,\ldots,c^\lambda_n,t_1,\ldots,t_K)
$$

仍然是线性的。矩阵形式为

$$
A_\lambda c^\lambda + A_\iota t \simeq b.
$$

这很重要：**在固定 $R,Z$ 几何后，$\lambda$ 和 $\iota$ 的 straight-field-line 初值可以通过线性最小二乘求出。**

### 10.4 为什么不能把 $R,Z,\lambda$ 全部一次线性求出

如果 $R,Z$ 也未知，上面的方程不再线性。原因是

$$
\nabla\theta,\quad \nabla\zeta,\quad \nabla\Phi^\lambda_j
$$

都依赖几何映射

$$
\mathbf x(\rho,\theta,\zeta).
$$

而

$$
\mathbf x_\rho,\mathbf x_\theta,\mathbf x_\zeta
$$

又由 $R,Z$ 的导数决定。于是 straight-field-line 条件会包含 $R,Z$ 系数的非线性函数。

同理，若直接用“磁面条件”

$$
\mathbf B_{\mathrm{coil}}\cdot\nabla\rho=0
$$

来反推 $R,Z$，也会遇到非线性。因为

$$
\nabla\rho=\frac{\mathbf x_\theta\times\mathbf x_\zeta}{\sqrt g},
$$

所以

$$
\mathbf B_{\mathrm{coil}}\cdot(\mathbf x_\theta\times\mathbf x_\zeta)=0
$$

对 $R,Z$ 系数是双线性/非线性的。

因此，“全部一次线性解出真正最优的 $R,Z,\lambda$”不现实。可行的线性化路径是：

1. 用 $\psi$ 或磁力线点云先确定几何点；
2. 线性拟合 $R,Z$；
3. 在固定几何上计算梯度；
4. 线性拟合 $\lambda$ 和 $\iota(\rho)$；
5. 必要时迭代 2 到 4 步。

### 10.5 两种可实施方案

#### 方案 A：几何角 + $\lambda$ 修正

这是最接近当前流程的方案。

1. 用 $\psi$ 提取多层等值面；
2. 每层每个 $\zeta$ 截面用几何角 $\alpha$ 标点；
3. 令 DESC 节点角

   $$
   \theta=\alpha;
   $$

4. 线性拟合 $R,Z$；
5. 在拟合几何上用线性方程拟合 $\lambda,\iota(\rho)$：

   $$
   \mathbf B\cdot\nabla(\theta+\lambda)
   =
   \iota(\rho)\mathbf B\cdot\nabla\zeta.
   $$

优点：

- 改动小；
- $R,Z$ 点云仍来自当前 $\psi$ 提取；
- $\lambda$ 的意义清晰；
- 可以直接对比 `lambda=0` 和 `lambda=LS` 的初始 force residual。

缺点：

- 如果几何角 $\alpha$ 非常差，$\lambda$ 可能需要很大幅度修正；
- $\lambda$ 拟合可能出现 wrap/gauge 问题；
- 如果 $R,Z$ 几何本身拟合差，$\lambda$ 不能补救。

#### 方案 B：直接用 straight-field-line 角拟合 $R,Z$

另一种更激进的做法是先通过磁力线追踪构造近似 straight-field-line 角 $\vartheta$，然后直接把 DESC 节点角取为

$$
\theta=\vartheta.
$$

这样拟合点云变成

$$
(\rho_i,\vartheta_i,\zeta_i)\mapsto(R_i,Z_i).
$$

然后线性拟合 $R,Z$，并令初始

$$
\lambda\approx0.
$$

或者只拟合一个小的 residual $\lambda$。

优点：

- $R,Z$ 本身就被参数化到近似 straight-field-line 角上；
- DESC 可能更容易接受；
- $\lambda$ 幅度可能更小。

缺点：

- 要在每一层和每个 $\zeta$ 截面建立稳定插值；
- 若磁力线追踪或 Poincare 共轭不稳，点云角标会污染 $R,Z$；
- 对 debug/cem 这种极端形状，角标 unwrap 和单调性需要严格检查。

### 10.6 线性最小二乘系统的推荐形式

推荐先实现方案 A，因为它更容易和当前代码对接。

#### 第一步：$R,Z$ 几何拟合

点云来自多层 $\psi$ 面：

$$
q_i=(\rho_i,\theta_i,\zeta_i),
\quad
y^R_i=R_i,
\quad
y^Z_i=Z_i.
$$

构造矩阵：

$$
A^R_{ij}=\Phi^R_j(q_i),
\quad
A^Z_{ij}=\Phi^Z_j(q_i).
$$

求解：

$$
\min_{c^R}
\|W_R(A^Rc^R-y^R)\|^2+\eta_R\|\Gamma_Rc^R\|^2,
$$

$$
\min_{c^Z}
\|W_Z(A^Zc^Z-y^Z)\|^2+\eta_Z\|\Gamma_Zc^Z\|^2.
$$

这里 $\Gamma$ 是高阶模式惩罚，防止拟合振荡。

边界可以用强权重或等式约束固定：

$$
R(1,\theta,\zeta)=R_{\mathrm{LCFS}}(\theta,\zeta),
\quad
Z(1,\theta,\zeta)=Z_{\mathrm{LCFS}}(\theta,\zeta).
$$

#### 第二步：$\lambda,\iota$ 直线场坐标拟合

在固定 $R,Z$ 后，对每个 collocation 点计算：

$$
g_i=\mathbf B_i\cdot\nabla\theta_i,
\quad
h_i=\mathbf B_i\cdot\nabla\zeta_i,
\quad
D_{ij}=\mathbf B_i\cdot\nabla\Phi^\lambda_j(q_i).
$$

令

$$
\iota(\rho)=\sum_k t_kP_k(\rho),
$$

则线性方程为：

$$
\sum_j D_{ij}c^\lambda_j
-
\sum_k h_iP_k(\rho_i)t_k
=
-g_i.
$$

加 gauge：

$$
\int_0^{2\pi}\int_0^{2\pi/\mathrm{nfp}}
\lambda(\rho,\theta,\zeta)\,d\theta\,d\zeta=0
$$

或简单固定每个 $\rho$ 层的 $\lambda$ 平均值为零。

最终求：

$$
\min_{c^\lambda,t}
\|W_\lambda(A_\lambda c^\lambda+A_\iota t-b)\|^2
+\eta_\lambda\|\Gamma_\lambda c^\lambda\|^2.
$$

这个系统是线性的。

#### 第三步：诊断

拟合后先不要直接跑 DESC solve，而是先检查：

$$
\epsilon_{\mathrm{sfl}}
=
\operatorname{rms}
\left[
\frac{
\mathbf B\cdot\nabla(\theta+\lambda)
-\iota(\rho)\mathbf B\cdot\nabla\zeta
}{
|\mathbf B|\,|\nabla(\theta+\lambda)|
}
\right],
$$

以及

$$
\epsilon_\rho
=
\operatorname{rms}
\left[
\frac{
\mathbf B\cdot\nabla\rho
}{
|\mathbf B|\,|\nabla\rho|
}
\right].
$$

前者看坐标是否接近 straight-field-line，后者看几何面是否真接近磁面。

### 10.7 这是否会比当前方法更有效

大概率会更有效，理由是当前 `lambda=0` 明显太粗糙。DESC 的内部磁场表达式显式依赖 $\lambda_z$ 等导数；如果 $\lambda$ 初值错，初始 $B^\theta/B^\zeta$、current 和 force residual 都会偏离很大。

但它不能保证 DESC 一定成功。原因是：

- 这只让初值更接近外部线圈磁场的直线场坐标；
- DESC 仍然要解 fixed-boundary MHD force balance；
- 若给定边界、通量和 profile 本身不兼容，或者体积内拓扑已经变差，DESC 仍会失败；
- 如果 $R,Z$ 几何来自不够准确的 $\psi$，$\lambda$ LS 只能修正角标，不能修正错误磁面。

因此它的定位应是：**比 `lambda=0` 更物理、更接近 DESC 坐标假设的初值生成器**，不是完整替代 DESC solve 的算法。

### 10.8 建议的下一步实验

建议先做一个不进入主流程的小实验：

1. 选一个已知较好的样本，例如 `01`，以及一个困难样本，例如 `cem_qh03`；
2. 对同一个边界构造三种初值：
   - `surface_scaled`：DESC 默认缩放面；
   - `psi_RZ_lambda0`：多层 $\psi$ 拟合 $R,Z$，$\lambda=0$；
   - `psi_RZ_lambdaLS`：多层 $\psi$ 拟合 $R,Z$，再线性拟合 $\lambda,\iota$；
3. 不跑 solve，先比较初始：
   - boundary fidelity；
   - `is_nested()`;
   - $\epsilon_\rho$；
   - $\epsilon_{\mathrm{sfl}}$；
   - DESC initial force residual；
4. 如果 `psi_RZ_lambdaLS` 的 initial force residual 明显下降，再跑 DESC solve；
5. 若 `01` 有效但 `cem_qh03` 无效，说明困难样本可能需要 continuation 或更小边界；
6. 若两者都有效，再考虑把 `lambdaLS` 作为 DESC 默认增强初值。

这组实验能直接判断“主动拟合 $R,Z,\lambda$”是否真的比当前方法更接近 DESC 的解。

## 11. 真正联合拟合 $R,Z,\lambda$ 的方案

上一节把 $R,Z$ 和 $\lambda$ 分开讲，是为了说明非线性来自哪里。但如果目标是“工程上一次性给 DESC 一个联合初值”，完全可以把 $R,Z,\lambda$ 放进同一个最小二乘系统里。需要区分两种“联合”：

1. **数据型联合 LS**：先给每个采样点指定 $(\rho,\theta,\zeta)$、目标 $R,Z,\lambda$，然后一次性拟合 `R_lmn/Z_lmn/L_lmn`。这是线性的。
2. **方程型联合 solve**：不预先给 $\lambda$ target，而是要求 $R,Z,\lambda$ 同时满足磁面条件、straight-field-line 条件和坐标质量条件。这是非线性的，只能做 nonlinear LS 或 Gauss-Newton 线性化。

这两者都算联合，但复杂度和可靠性不同。建议先做第一种，因为它可控、快、容易和 DESC 的 `set_initial_guess(nodes,R,Z,lambda)` 对接。

### 11.1 数据型联合线性 LS

构造一个统一未知向量：

$$
x=
\begin{bmatrix}
c^R\\
c^Z\\
c^\lambda\\
t
\end{bmatrix},
$$

其中 $c^R,c^Z,c^\lambda$ 是 DESC 谱系数，$t$ 是可选的 $\iota(\rho)$ profile 系数：

$$
\iota(\rho)=\sum_k t_kP_k(\rho).
$$

对每个采样点

$$
q_i=(\rho_i,\theta_i,\zeta_i)
$$

写三类方程。

第一类是几何点方程：

$$
\sum_j c^R_j\Phi^R_j(q_i)=R_i,
$$

$$
\sum_j c^Z_j\Phi^Z_j(q_i)=Z_i.
$$

第二类是 $\lambda$ 方程。若已经通过磁力线追踪或 Poincare 共轭得到该点的 straight-field-line 角 $\vartheta_i$，并且当前节点角取为几何角 $\theta_i$，则

$$
\lambda_i^{\mathrm{target}}=\vartheta_i-\theta_i.
$$

于是

$$
\sum_j c^\lambda_j\Phi^\lambda_j(q_i)=\lambda_i^{\mathrm{target}}.
$$

三类方程合在一起就是一个 block linear LS：

$$
\begin{bmatrix}
A_R & 0 & 0 & 0\\
0 & A_Z & 0 & 0\\
0 & 0 & A_\lambda & 0
\end{bmatrix}
\begin{bmatrix}
c^R\\
c^Z\\
c^\lambda\\
t
\end{bmatrix}
\simeq
\begin{bmatrix}
R\\
Z\\
\lambda^{\mathrm{target}}
\end{bmatrix}.
$$

如果不想先显式求 $\vartheta_i$，也可以把每条磁力线的初始相位 $\beta_s$ 和 $\iota(\rho)$ 一起放进线性系统。对第 $s$ 条磁力线上的第 $i$ 个点，有

$$
\vartheta_i=\beta_s+\iota(\rho_i)\zeta_i.
$$

因此

$$
\lambda_i=\beta_s+\iota(\rho_i)\zeta_i-\theta_i.
$$

把 $\lambda$ 展开代入：

$$
\sum_j c^\lambda_j\Phi^\lambda_j(q_i)
-\beta_s
-\zeta_i\sum_k t_kP_k(\rho_i)
=
-\theta_i.
$$

这对未知量 $c^\lambda,\beta,t$ 仍然是线性的。于是联合系统变成：

$$
\begin{bmatrix}
A_R & 0 & 0 & 0 & 0\\
0 & A_Z & 0 & 0 & 0\\
0 & 0 & A_\lambda & A_\beta & A_\iota
\end{bmatrix}
\begin{bmatrix}
c^R\\
c^Z\\
c^\lambda\\
\beta\\
t
\end{bmatrix}
\simeq
\begin{bmatrix}
R\\
Z\\
-\theta
\end{bmatrix}.
$$

这里第三行中：

$$
(A_\beta)_{is}=-1
$$

若点 $i$ 属于磁力线 $s$，否则为 0；

$$
(A_\iota)_{ik}=-\zeta_iP_k(\rho_i).
$$

这个形式很适合批量磁力线数据，因为不需要先显式解完整共轭函数 $h$，只要求每条线在一段追踪内近似满足直线场推进。

### 11.2 这个联合线性 LS 如何生成数据

推荐的数据生成方式如下。

对每个半径层 $\rho_\ell$：

1. 在 $\zeta=0$ 的 $\psi=\rho_\ell^2\psi_{\mathrm{edge}}$ 曲线上取 $N_\theta$ 个起点；
2. 同时追踪这些起点到多个 $\zeta$ 截面；
3. 对每个追踪点，记录物理坐标

   $$
   (R_i,Z_i,\phi_i)
   $$

   以及所属层 $\rho_\ell$ 和所属磁力线编号 $s$；
4. 在当前截面内计算几何角

   $$
   \theta_i=\operatorname{atan2}
   \left(
   \frac{Z_i-Z_{\mathrm{axis}}(\zeta_i)}{a_Z(\zeta_i)},
   \frac{R_i-R_{\mathrm{axis}}(\zeta_i)}{a_R(\zeta_i)}
   \right),
   $$

   或者使用由 $\psi$ 等值线射线参数给出的角；
5. 设

   $$
   q_i=(\rho_\ell,\theta_i,\zeta_i).
   $$

然后一次性最小二乘求

$$
c^R,c^Z,c^\lambda,\beta,t.
$$

这种做法的直观意义是：

- $R,Z$ 要把整批磁力线采样点拟合成嵌套曲面族；
- $\lambda$ 要把几何角 $\theta$ 修正成沿磁力线线性推进的角；
- $\iota(\rho)$ 由同一个系统顺手拟合出来。

这就是你说的“直接拟合出 $R,Z,\lambda$ 的近似解”的联合版。

### 11.3 必须加的约束和 gauge

联合系统必须加 gauge，否则 $\lambda$、$\beta$、$\iota$ 有规范自由度。

建议加：

1. 每个 $\rho$ 层的 $\lambda$ 平均值为 0：

   $$
   \langle \lambda\rangle_{\theta,\zeta}(\rho_\ell)=0.
   $$

2. 固定一条参考磁力线的初始相位，例如

   $$
   \beta_0=0.
   $$

3. $\iota(\rho)$ 只用低阶径向基，例如

   $$
   \iota(\rho)=t_0+t_2\rho^2+t_4\rho^4.
   $$

4. 边界点强权重约束：

   $$
   R(1,\theta,\zeta)=R_{\mathrm{LCFS}},
   \quad
   Z(1,\theta,\zeta)=Z_{\mathrm{LCFS}}.
   $$

5. 高阶模式正则化：

   $$
   \eta_R\|\Gamma_Rc^R\|^2+
   \eta_Z\|\Gamma_Zc^Z\|^2+
   \eta_\lambda\|\Gamma_\lambda c^\lambda\|^2.
   $$

这些约束仍然是线性的或二次正则，因此仍是标准 least squares。

### 11.4 纯几何能不能给出 $\lambda$

严格说，不能给出 DESC 需要的物理 $\lambda$。

纯几何可以定义某种“漂亮坐标”，比如让网格更均匀、让 $\sqrt g$ 波动更小、让曲线更正交。但 DESC 中 $\lambda$ 的作用是把角坐标修正为 straight-field-line/PEST 角。这个角依赖磁场方向，不是曲面几何唯一决定的。

所以如果“纯几何”指只用嵌套曲面形状，不用 $\mathbf B$ 或磁力线追踪，那么 $\lambda$ 只能作为任意坐标 gauge，不能保证降低 DESC force residual。

如果“纯几何”允许使用磁力线在曲面上的几何轨迹，即点云来自外部线圈场追踪，那么可以通过上一节的相位推进方程得到 $\lambda$。这已经不再是纯曲面几何，而是“磁场轨迹几何”。

### 11.5 方程型联合拟合为什么不是线性的

更理想的联合方程是同时要求：

$$
\mathbf B\cdot\nabla\rho=0,
$$

$$
\mathbf B\cdot\nabla(\theta+\lambda)
=
\iota(\rho)\mathbf B\cdot\nabla\zeta,
$$

以及边界、轴、嵌套性和光滑性。

这里未知量是 $R,Z,\lambda,\iota$。问题在于

$$
\nabla\rho,\quad \nabla\theta,\quad \nabla\zeta
$$

都由几何映射

$$
\mathbf x(\rho,\theta,\zeta; c^R,c^Z)
$$

决定。因此这些方程对 $c^R,c^Z$ 是非线性的。

例如

$$
\nabla\rho=
\frac{\mathbf x_\theta\times\mathbf x_\zeta}{\sqrt g},
$$

所以

$$
\mathbf B\cdot\nabla\rho=0
$$

含有 $R,Z$ 导数的叉积和 Jacobian。它不是线性方程。

同理，straight-field-line 方程虽然对 $\lambda,\iota$ 是线性的，但对 $R,Z$ 不是线性的，因为梯度算子依赖 $R,Z$。

因此，严格的“方程型联合拟合”应该写成 nonlinear least squares：

$$
\min_{c^R,c^Z,c^\lambda,t}
\|r_{\mathrm{geom}}\|^2
+w_\rho\|r_{\rho}\|^2
+w_{\mathrm{sfl}}\|r_{\mathrm{sfl}}\|^2
+w_b\|r_{\mathrm{boundary}}\|^2
+w_{\mathrm{reg}}\|r_{\mathrm{reg}}\|^2.
$$

其中

$$
r_\rho=
\frac{\mathbf B\cdot\nabla\rho}{|\mathbf B||\nabla\rho|},
$$

$$
r_{\mathrm{sfl}}=
\frac{
\mathbf B\cdot\nabla(\theta+\lambda)
-\iota(\rho)\mathbf B\cdot\nabla\zeta
}{
|\mathbf B||\nabla(\theta+\lambda)|
}.
$$

这个问题可以用 Gauss-Newton 做。每一步在当前 $R,Z,\lambda$ 附近线性化，然后解一个联合线性 LS：

$$
J
\begin{bmatrix}
\delta c^R\\
\delta c^Z\\
\delta c^\lambda\\
\delta t
\end{bmatrix}
=
-r.
$$

这才是严格意义上的“联合方程拟合”。它比数据型联合 LS 更强，但实现成本也高很多。

### 11.6 推荐实现路线

建议分两阶段。

第一阶段做 **数据型联合线性 LS**：

1. 多层 $\psi$ 面给起点；
2. 批量磁力线追踪生成点云；
3. 点云直接提供 $R,Z$ target；
4. 磁力线相位推进方程提供 $\lambda,\beta,\iota$ 的线性约束；
5. 一次性求 $R,Z,\lambda,\iota$；
6. 输出 DESC 初值，检查初始 force residual。

这是最便宜、最容易验证的版本。

第二阶段再做 **Gauss-Newton 联合方程拟合**：

1. 以上一阶段结果为初值；
2. 加入 $r_\rho$ 和 $r_{\mathrm{sfl}}$；
3. 自动或手写 Jacobian；
4. 每步解联合 LS；
5. 只迭代少数几步，目标是给 DESC 更近初值，不是替代 DESC。

如果第一阶段已经显著降低 DESC 初始 force residual，第二阶段可以暂缓。

### 11.7 对当前项目的判断

你强调的“联合 $R,Z,\lambda$”是合理的，而且比单独补 $\lambda$ 更符合 DESC 的需求。最实际的线性版本是：

$$
\boxed{
\text{用磁力线点云同时拟合 }R,Z,\lambda,\iota
}
$$

它不是纯曲面几何，因为 $\lambda$ 需要磁力线相位信息；但它可以保持线性最小二乘，并且和 GPU 批量追踪天然匹配。

我建议下一步优先实现这个数据型联合 LS。判断标准很明确：和当前 `psi_RZ_lambda0` 相比，它是否显著降低 `initial_force_mean_abs_normalized`、是否保持 `is_nested=True`，以及 DESC solve 后是否更不容易翻折。

## 12. 坐标标签、$\lambda$ 唯一性和连续方程

这一节先不讨论谱展开和矩阵实现，只讨论连续层面的定义。这样可以把“采样点的 $(\rho,\theta,\zeta)$ 从哪里来”“$\lambda$ 到底是不是唯一”“完整方程是什么”分清楚。

### 12.1 $(\rho,\theta,\zeta)$ 不是物理点自带的唯一坐标

给定一个物理点

$$
\mathbf x=(R\cos\phi,R\sin\phi,Z),
$$

它本身并不会天然携带唯一的 $(\rho,\theta,\zeta)$。这些是我们为一族嵌套曲面选择的参数标签。也就是说，我们要构造一个映射

$$
\mathbf x(\rho,\theta,\zeta)
=
\left(
R(\rho,\theta,\zeta)\cos\phi(\rho,\theta,\zeta),
R(\rho,\theta,\zeta)\sin\phi(\rho,\theta,\zeta),
Z(\rho,\theta,\zeta)
\right).
$$

在当前最简单的接法里，我们固定

$$
\zeta=\phi,
$$

即 DESC/VMEC 的 toroidal coordinate 直接取物理柱坐标 toroidal angle。这样物理点的 $\zeta$ 可以直接由它所在的 toroidal 截面给出。

$\rho$ 是磁面标签，也需要人为指定。常用选择是：

$$
\rho=\sqrt{\frac{\psi}{\psi_{\mathrm{edge}}}},
$$

其中 $\psi_{\mathrm{edge}}$ 是目标最外层磁面。这样 $\rho=0$ 是磁轴，$\rho=1$ 是目标边界。更物理的选择是用归一化环向通量，但当前我们从 $\psi$ 面出发，先用 $\sqrt{\psi/\psi_{\mathrm{edge}}}$ 最方便。

$\theta$ 是 poloidal label。它最不唯一。可以选择：

1. 几何角，例如相对磁轴的截面极角；
2. 射线提取等值面时的射线角；
3. 由磁力线追踪构造的 near-straight-field-line 角；
4. Boozer/PEST 角的近似。

不同 $\theta$ 选择对应不同的 $R,Z,\lambda$ 表示，但可能表示同一族物理曲面。

因此，给采样点指定 $(\rho,\theta,\zeta)$ 是可以的，但它是坐标选择的一部分，不是测量结果。指定得好，DESC 初值就接近；指定得差，几何上可能仍是同一个面，但 DESC 的 $\lambda$ 和 force residual 会很差。

### 12.2 两种实际给点赋坐标的方法

#### 方法 A：$\psi$ 等值面射线法

对每一层

$$
\psi_\ell=\rho_\ell^2\psi_{\mathrm{edge}},
$$

在每个 toroidal 截面

$$
\zeta_j=\phi_j
$$

里，以磁轴点为中心取射线角 $\alpha_k$，解

$$
\psi(R_{\mathrm{axis}}+r\cos\alpha_k,\,
Z_{\mathrm{axis}}+r\sin\alpha_k,\,
\zeta_j)=\psi_\ell.
$$

解出的点赋予标签

$$
(\rho,\theta,\zeta)=(\rho_\ell,\alpha_k,\zeta_j).
$$

这给出的是几何角参数化。它通常适合拟合 $R,Z$，但 $\theta=\alpha$ 一般不是 straight-field-line 角，所以需要非零 $\lambda$ 修正。

#### 方法 B：磁力线点云法

对每一层 $\rho_\ell$，先在 $\zeta=0$ 的 $\psi=\rho_\ell^2\psi_{\mathrm{edge}}$ 曲线上取起点。每个起点有一个初始线号 $s$ 和初始角 $\alpha_s$。然后追踪外部线圈磁场，记录它穿过一系列 toroidal 截面的点：

$$
(R_i,Z_i,\phi_i).
$$

赋值时取

$$
\rho_i=\rho_\ell,
\quad
\zeta_i=\phi_i.
$$

$\theta_i$ 有两种选择：

1. 仍取该截面内的几何角；
2. 直接取磁力线相位角。

若取几何角，则 $\lambda$ 负责把几何角修正为直线场角。若取磁力线相位角，则 $R,Z$ 的参数化本身已经更接近直线场角，$\lambda$ 可以更小。

磁力线点云法的优点是它天然携带 field-line phase 信息，更适合联合拟合 $R,Z,\lambda,\iota$。缺点是如果磁力线漂移、岛化或混沌，点云不再来自一张光滑嵌套面，拟合会暴露为 residual 大或坐标不嵌套。

### 12.3 “纯几何但考虑磁力线”是什么意思

这里需要区分三件事。

第一，纯曲面几何：只知道一族嵌套曲面长什么样，不用磁场方向。这不能唯一确定 $\lambda$。

第二，磁力线轨迹几何：不用电流、不用 MHD 平衡，但使用外部线圈磁场的磁力线在曲面上的走向。这可以确定 straight-field-line 坐标，因此可以确定 $\lambda$ 到规范自由度。

第三，MHD equilibrium：进一步要求 $\mathbf J\times\mathbf B=\nabla p$。这是 DESC solve 真正在做的事。我们现在讨论的联合拟合 $R,Z,\lambda$ 属于第二类，不考虑电流和平衡，但考虑磁力线方向。

所以你说的“纯几何但磁力线要考虑”是合理的。更准确地说，它是 **field-line geometry**，不是纯 surface geometry。

### 12.4 连续未知量

先固定外部线圈磁场：

$$
\mathbf B=\mathbf B_{\mathrm{coil}}(\mathbf x).
$$

我们想构造的未知量是：

1. 一族曲面：

   $$
   \mathbf x(\rho,\theta,\zeta)
   $$

   或等价的 $R(\rho,\theta,\zeta),Z(\rho,\theta,\zeta)$，并先取 $\zeta=\phi$。

2. 一个 straight-field-line 角修正：

   $$
   \lambda(\rho,\theta,\zeta).
   $$

3. 一个旋转变换 profile：

   $$
   \iota(\rho).
   $$

定义

$$
\vartheta=\theta+\lambda.
$$

这里 $\vartheta$ 是希望接近 PEST/straight-field-line 的 poloidal angle。

### 12.5 完整的磁面方程

若 $\rho=\mathrm{const}$ 是外部线圈磁场的磁面，则磁场必须切向于该曲面：

$$
\mathbf B(\mathbf x)\cdot\nabla\rho=0.
$$

用参数曲面写，令

$$
\mathbf x_\rho=\partial_\rho\mathbf x,\quad
\mathbf x_\theta=\partial_\theta\mathbf x,\quad
\mathbf x_\zeta=\partial_\zeta\mathbf x.
$$

Jacobian 为

$$
\sqrt g=
\mathbf x_\rho\cdot(\mathbf x_\theta\times\mathbf x_\zeta).
$$

因为

$$
\nabla\rho=
\frac{\mathbf x_\theta\times\mathbf x_\zeta}{\sqrt g},
$$

所以磁面方程等价于

$$
\boxed{
\mathbf B(\mathbf x)\cdot
(\mathbf x_\theta\times\mathbf x_\zeta)=0
}
$$

这个方程不涉及电流和平衡，只表达“曲面是磁面”。它是几何上最根本的方程。

### 12.6 完整的 straight-field-line 方程

在坐标 $(\rho,\theta,\zeta)$ 中，磁场的逆变分量是

$$
B^\rho=\mathbf B\cdot\nabla\rho,
$$

$$
B^\theta=\mathbf B\cdot\nabla\theta,
$$

$$
B^\zeta=\mathbf B\cdot\nabla\zeta.
$$

磁面方程要求

$$
B^\rho=0.
$$

straight-field-line 条件要求磁力线在 $(\vartheta,\zeta)$ 平面中是直线：

$$
\frac{d\vartheta}{d\zeta}=\iota(\rho).
$$

沿磁力线有

$$
\frac{d\vartheta}{d\zeta}
=
\frac{\mathbf B\cdot\nabla\vartheta}
{\mathbf B\cdot\nabla\zeta}.
$$

因此

$$
\boxed{
\mathbf B\cdot\nabla(\theta+\lambda)
=
\iota(\rho)\,\mathbf B\cdot\nabla\zeta
}
$$

或者展开为

$$
\boxed{
\mathbf B\cdot\nabla\lambda
-
\iota(\rho)\mathbf B\cdot\nabla\zeta
=
-\mathbf B\cdot\nabla\theta
}
$$

若几何 $\mathbf x(\rho,\theta,\zeta)$ 已经固定，这个方程对 $\lambda$ 和 $\iota$ 是线性的。

还可以写成参数空间形式。因为在磁面上 $B^\rho=0$，有

$$
\mathbf B
=
B^\theta\mathbf x_\theta+B^\zeta\mathbf x_\zeta.
$$

并且

$$
\mathbf B\cdot\nabla\lambda
=
B^\theta\lambda_\theta+B^\zeta\lambda_\zeta.
$$

于是 straight-field-line 方程变成

$$
\boxed{
(1+\lambda_\theta)B^\theta
\lambda_\zeta B^\zeta
=
\iota(\rho)B^\zeta
}
$$

这更直观：$\lambda$ 调整 poloidal angle，使磁力线斜率变成只依赖 $\rho$ 的 $\iota(\rho)$。

### 12.7 边界、轴和嵌套性条件

除了磁面和 straight-field-line 方程，还需要几何合法性条件。

磁轴条件：

$$
\mathbf x(0,\theta,\zeta)=\mathbf x_{\mathrm{axis}}(\zeta),
$$

即 $\rho=0$ 时不依赖 $\theta$。

边界条件：

$$
\mathbf x(1,\theta,\zeta)=\mathbf x_{\mathrm{LCFS}}(\theta,\zeta).
$$

周期条件：

$$
\mathbf x(\rho,\theta+2\pi,\zeta)=\mathbf x(\rho,\theta,\zeta),
$$

$$
\mathbf x(\rho,\theta,\zeta+2\pi/\mathrm{nfp})
=
\mathcal S\,\mathbf x(\rho,\theta,\zeta),
$$

其中 $\mathcal S$ 表示一个 field period 的旋转对称。

嵌套性条件：

$$
\sqrt g
=
\mathbf x_\rho\cdot(\mathbf x_\theta\times\mathbf x_\zeta)
$$

不能变号。实际判据是：

$$
\operatorname{sign}(\sqrt g)
\text{ 在整个体积内一致。}
$$

这不是线性方程，但必须作为诊断或约束。

### 12.8 $\lambda$ 是否唯一

若以下内容都固定：

1. 曲面族 $\rho=\mathrm{const}$；
2. toroidal 坐标 $\zeta=\phi$；
3. 计算 poloidal 角 $\theta$ 的定义；
4. $\iota(\rho)$ 或其求解方式；
5. $\lambda$ 的平均值 gauge；

那么 $\lambda$ 基本唯一。

更准确地说，straight-field-line 方程

$$
\mathbf B\cdot\nabla\lambda
=
\iota(\rho)\mathbf B\cdot\nabla\zeta
-\mathbf B\cdot\nabla\theta
$$

沿每条磁力线是一阶微分方程。对一张 irrational surface，磁力线在面上稠密。若要求 $\lambda$ 是单值、周期、光滑的函数，则解只差一个 flux function：

$$
\lambda\mapsto \lambda+C(\rho).
$$

这个自由度可以用

$$
\langle\lambda\rangle_{\theta,\zeta}=0
$$

消掉。

同时 $\iota(\rho)$ 不是任意的。周期性/单值性要求会选出该磁面的旋转变换，也就是 field-line rotation number。若 $\iota$ 取错，$\lambda$ 通常无法成为全局单值周期函数，只能在最小二乘意义下近似。

若允许重新定义 poloidal label：

$$
\theta'=\theta+f(\rho,\theta,\zeta),
$$

则 $\lambda$ 会相应变化：

$$
\lambda'=\lambda-f.
$$

因此不固定 $\theta$ 的定义时，$\lambda$ 不唯一。

若允许改变 toroidal coordinate：

$$
\zeta'=\zeta+\omega(\rho,\theta,\zeta),
$$

还会引入额外 gauge。这就是之前讨论的 $G/\omega$ 修正相关问题。为了第一版实现清楚，建议先固定

$$
\zeta=\phi.
$$

### 12.9 有岛或混沌时会发生什么

如果外部线圈磁场在目标区域没有全局嵌套磁面，那么不存在光滑的

$$
\rho(\mathbf x)
$$

使得

$$
\mathbf B\cdot\nabla\rho=0
$$

在整个体积成立。

这时也不存在全局光滑的 $\lambda(\rho,\theta,\zeta)$ 和 $\iota(\rho)$ 让所有磁力线直线化。表现为：

- $R,Z$ 拟合 residual 大；
- $\mathbf B\cdot\nabla\rho$ residual 大；
- Poincare map 不可由单调圆映射描述；
- $\lambda$ 出现不连续、强振荡或多值；
- $\sqrt g$ 变号；
- 不同磁力线给出的 $\iota$ 不一致。

因此联合拟合本身也可以作为拓扑诊断。如果残差无法压下去，不一定是算法失败，也可能是目标区域确实没有全局嵌套结构。

### 12.10 不展开成系数时的完整问题

在不考虑 MHD force balance、只考虑外部线圈磁场的 field-line geometry 时，完整连续问题可以写成：

未知：

$$
\mathbf x(\rho,\theta,\zeta),\quad
\lambda(\rho,\theta,\zeta),\quad
\iota(\rho).
$$

方程：

$$
\mathbf B(\mathbf x)\cdot
(\mathbf x_\theta\times\mathbf x_\zeta)=0,
$$

$$
\mathbf B\cdot\nabla(\theta+\lambda)
=
\iota(\rho)\mathbf B\cdot\nabla\zeta.
$$

边界/正则条件：

$$
\mathbf x(0,\theta,\zeta)=\mathbf x_{\mathrm{axis}}(\zeta),
$$

$$
\mathbf x(1,\theta,\zeta)=\mathbf x_{\mathrm{LCFS}}(\theta,\zeta),
$$

$$
\sqrt g\text{ 不变号},
$$

$$
\lambda\text{ 周期且 }\langle\lambda\rangle_{\theta,\zeta}=0.
$$

这就是“不考虑电流和平衡，但考虑磁力线”的完备方程组。

它和 DESC 真正 solve 的 MHD 方程不同。它的目标是给 DESC 构造一个尽可能接近真实外部线圈磁面和 straight-field-line 坐标的初值。

### 12.11 对计算方法的启发

从这些连续方程可以看出：

- 若 $\mathbf x$ 未知，磁面方程对 $R,Z$ 是非线性的；
- 若 $\mathbf x$ 已知，straight-field-line 方程对 $\lambda,\iota$ 是线性的；
- 若用磁力线点云直接给 $R,Z$ 数据，并用相位推进给 $\lambda$ 数据，则可以构造数据型联合线性 LS；
- 若想严格同时满足磁面方程和 straight-field-line 方程，则需要 nonlinear LS 或 Gauss-Newton。

所以推荐路线仍然是：

1. 用磁力线点云构造联合数据；
2. 一次性线性拟合 $R,Z,\lambda,\iota$；
3. 检查上面两条连续方程的 residual；
4. 必要时再做少数步 Gauss-Newton。

## 13. 从连续方程到可计算算法

这一节继续把上一节的连续问题落到实际计算。目标不是立刻写代码，而是把“哪些数据要采样、每个点的坐标怎么赋、联合拟合怎么组织、失败时如何解释”讲清楚。

### 13.1 两种坐标策略

给同一批磁力线点云，可以有两种主要参数化。

#### 策略 A：几何角作为 $\theta$，用 $\lambda$ 修正

对每个点，先用截面几何角定义

$$
\theta_i=\alpha_i.
$$

然后用 $\lambda$ 把它修正成 straight-field-line 角：

$$
\vartheta_i=\theta_i+\lambda_i.
$$

磁力线约束写成：

$$
\vartheta_i=\beta_s+\iota(\rho_\ell)\zeta_i
$$

其中 $s$ 是该点所属的磁力线编号，$\beta_s$ 是这条磁力线在 $\zeta=0$ 的初始相位。

于是

$$
\lambda_i=\beta_s+\iota(\rho_\ell)\zeta_i-\theta_i.
$$

这个策略的优点是 $R,Z$ 的点云来自稳定的几何角，容易和当前 $\psi$ 等值线提取对接。缺点是如果几何角与直线场角差很多，$\lambda$ 可能很大、振荡强。

#### 策略 B：直接把 straight-field-line 角作为 $\theta$

也可以直接定义

$$
\theta_i=\vartheta_i=\beta_s+\iota(\rho_\ell)\zeta_i,
$$

然后拟合

$$
R(\rho_i,\theta_i,\zeta_i),\quad Z(\rho_i,\theta_i,\zeta_i),
$$

并取

$$
\lambda\approx0.
$$

这个策略的优点是 DESC 的 $\theta$ 本身就接近 straight-field-line 角，$\lambda$ 初值很小。缺点是如果磁力线点云在某些截面分布不均匀，或者 $\iota$ 估计不准，$R,Z$ 的参数化会被污染。

### 13.2 哪个策略更适合第一版

建议第一版采用策略 A：

$$
\theta=\alpha,\quad \lambda=\vartheta-\alpha.
$$

原因是：

1. 当前 $\psi$ 等值面提取已经天然给出几何角 $\alpha$；
2. $R,Z$ 拟合更稳定，不会直接受 $\iota$ 初估误差影响；
3. $\lambda$ residual 可以单独诊断；
4. 如果 $\lambda$ 拟合失败，可以明确知道是坐标直线化失败，而不是 $R,Z$ 几何也一起被污染；
5. 它和 DESC 的 `set_initial_guess(nodes,R,Z,lambda)` 最直接对应。

策略 B 可以作为第二阶段优化。如果策略 A 得到的 $\lambda$ 很大且 DESC 仍然不好，再尝试直接用 straight-field-line 角重参数化 $R,Z$。

### 13.3 采样点如何生成

对每个目标半径层

$$
\rho_\ell,\quad \ell=1,\ldots,N_\rho,
$$

执行以下步骤。

第一步，在 $\zeta=0$ 的 $\psi=\rho_\ell^2\psi_{\mathrm{edge}}$ 曲线上取 $N_\alpha$ 个起点：

$$
\mathbf x_{\ell s}(0),\quad s=1,\ldots,N_\alpha.
$$

每个起点有几何角

$$
\alpha_{\ell s}(0).
$$

第二步，批量追踪外部线圈磁场到一组 toroidal 截面：

$$
\zeta_j=j\Delta\zeta,\quad j=0,\ldots,N_\zeta-1.
$$

得到穿越点：

$$
\mathbf x_{\ell s j}.
$$

第三步，在每个截面内重新计算几何角：

$$
\alpha_{\ell s j}
=
\operatorname{atan2}
\left(
Z_{\ell s j}-Z_{\mathrm{axis}}(\zeta_j),
R_{\ell s j}-R_{\mathrm{axis}}(\zeta_j)
\right)
$$

实际实现中可以对 $R,Z$ 先用局部尺度归一化：

$$
\alpha=
\operatorname{atan2}
\left(
\frac{Z-Z_{\mathrm{axis}}}{a_Z(\zeta)},
\frac{R-R_{\mathrm{axis}}}{a_R(\zeta)}
\right).
$$

第四步，给每个点赋坐标：

$$
\rho_i=\rho_\ell,\quad
\theta_i=\alpha_{\ell s j},\quad
\zeta_i=\zeta_j.
$$

同时记录点的线号 $s$、层号 $\ell$，以及物理坐标 $R_i,Z_i$。

### 13.4 如何估计 $\iota(\rho)$ 和相位 $\beta_s$

对每个半径层，可以先从 Poincare return map 估计旋转数。

追踪一周期后，得到

$$
\alpha_{\ell s}'.
$$

unwrap 后估计平均相位推进：

$$
\Delta_\ell\approx
\operatorname{mean}_s
\left(
\alpha_{\ell s}'-\alpha_{\ell s}
\right).
$$

于是初估

$$
\iota_\ell\approx
\frac{\mathrm{nfp}}{2\pi}\Delta_\ell.
$$

这只是初值。在线性 LS 中还可以把 $\iota(\rho)$ 的 profile 系数 $t_k$ 作为未知量一起修正。

对每条线的初始相位 $\beta_s$，有两种方式：

1. 直接设

   $$
   \beta_s=\alpha_{\ell s}(0)
   $$

   然后只拟合 $\lambda$ 和 $\iota$；

2. 把 $\beta_s$ 也作为未知量进入 LS。

推荐第二种。因为几何角和直线场角在 $\zeta=0$ 也未必一致，$\beta_s$ 应该让 LS 自己调整。

### 13.5 联合线性系统的具体结构

对所有点 $i=(\ell,s,j)$，未知量取为

$$
x=
\begin{bmatrix}
c^R\\
c^Z\\
c^\lambda\\
\beta\\
t
\end{bmatrix}.
$$

几何方程：

$$
R(q_i)=R_i,
\quad
Z(q_i)=Z_i.
$$

直线场相位方程：

$$
\lambda(q_i)-\beta_{\ell s}
-\zeta_i\iota(\rho_i)
=
-\theta_i.
$$

其中

$$
\iota(\rho_i)=\sum_k t_kP_k(\rho_i).
$$

把 $R,Z,\lambda$ 展开后，三类方程都是线性的：

$$
A_Rc^R\simeq R,
$$

$$
A_Zc^Z\simeq Z,
$$

$$
A_\lambda c^\lambda
-A_\beta\beta
-A_\iota t
\simeq
-\theta.
$$

这就是严格意义上的联合线性 LS。它一次性求 $R,Z,\lambda,\beta,\iota$，而不是先后分开求。

### 13.6 为什么这个系统还不是完整方程

上面的联合 LS 是数据型方程，不是直接解连续磁面 PDE。它依赖于一个前提：磁力线点云确实近似落在一族嵌套磁面上。

换句话说，它把复杂的

$$
\mathbf B\cdot\nabla\rho=0
$$

通过“沿磁力线采样点云”间接吸收了。如果磁力线在一层内漂移很小，那么点云自然落在同一张面上，$R,Z$ 可以拟合得好。如果磁力线漂移很大，$R,Z$ 拟合 residual 会变大，或者拟合后嵌套性会变差。

因此第一版不直接把

$$
\mathbf B\cdot(\mathbf x_\theta\times\mathbf x_\zeta)=0
$$

放进 LS，而是用磁力线点云作为这个方程的离散证据。

如果要更严格，就进入 Gauss-Newton 阶段，把磁面 PDE residual 显式加入。

### 13.7 角度 unwrap 和周期性

相位方程涉及角度，必须处理 $2\pi$ 周期。不能直接用原始 `atan2` 输出。

对每条线，应该沿 $\zeta$ 方向 unwrap：

$$
\tilde\theta_{\ell s j}
=
\operatorname{unwrap}_j(\theta_{\ell s j}).
$$

相位方程使用 unwrap 后的 $\tilde\theta$：

$$
\lambda(q_i)-\beta_{\ell s}
-\zeta_i\iota(\rho_i)
=
-\tilde\theta_i.
$$

但是拟合出的 $\lambda$ 本身必须是周期函数。这里的兼容性由 $\iota$ 和 $\beta$ 吸收：$\tilde\theta$ 的线性增长部分由 $\zeta\iota$ 表示，剩下的周期部分由 $\lambda$ 表示。

如果某层存在岛或混沌，unwrap 后的相位推进会在不同线之间不一致，LS residual 会明显变大。

### 13.8 需要的规范约束

联合 LS 中至少有以下 gauge。

#### $\lambda$ 平均值

因为

$$
\theta+\lambda
$$

整体加一个只依赖 $\rho$ 的常数不改变直线场性质，所以有

$$
\lambda\mapsto\lambda+C(\rho),
\quad
\beta_{\ell s}\mapsto\beta_{\ell s}+C(\rho_\ell).
$$

需要固定：

$$
\langle\lambda\rangle_{\theta,\zeta}(\rho_\ell)=0.
$$

#### 每层相位原点

每一层的 $\vartheta$ 还可以整体平移：

$$
\vartheta\mapsto\vartheta+C_\ell.
$$

可以固定某条参考线：

$$
\beta_{\ell,0}=0
$$

或者固定所有 $\beta_{\ell s}$ 的平均：

$$
\operatorname{mean}_s \beta_{\ell s}=0.
$$

#### $\iota$ 和 $\beta$ 的耦合

如果只追踪很短的 toroidal 范围，$\iota$ 和 $\beta$ 可能相关性强。解决办法：

- 至少追踪一个场周期；
- 或追踪多个场周期但仍只取未漂移太大的点；
- 或给 $\iota$ 加弱先验，来自 return map 初估。

### 13.9 权重怎么选

联合 LS 中不同方程量纲不同，需要权重。

几何方程建议用长度归一化：

$$
w_R=w_Z=\frac{1}{a},
$$

其中 $a$ 是目标小半径尺度。

$\lambda$ 方程是角度，权重可以取

$$
w_\lambda\sim1.
$$

如果想让边界严格对齐，可以对 $\rho=1$ 点给更大权重：

$$
w_{\mathrm{boundary}}\gg w_{\mathrm{interior}}.
$$

高阶模式正则可以按模式阶数加权，例如

$$
\Gamma_j\sim (1+|m_j|+|n_j|+\ell_j)^p.
$$

这能避免为了追踪点噪声而出现高阶振荡。

### 13.10 成功判据

联合拟合完成后，不应直接相信结果，而要检查以下量。

几何拟合误差：

$$
\epsilon_{RZ}
=
\operatorname{rms}
\frac{\sqrt{(R_{\mathrm{fit}}-R_i)^2+(Z_{\mathrm{fit}}-Z_i)^2}}{a}.
$$

直线场角 residual：

$$
\epsilon_{\lambda}
=
\operatorname{rms}
\left[
\lambda(q_i)-\beta_{\ell s}
-\zeta_i\iota(\rho_i)
+\tilde\theta_i
\right].
$$

磁面 residual：

$$
\epsilon_{\rho}
=
\operatorname{rms}
\left|
\frac{\mathbf B\cdot\nabla\rho}{|\mathbf B||\nabla\rho|}
\right|.
$$

直线场 residual：

$$
\epsilon_{\mathrm{sfl}}
=
\operatorname{rms}
\left|
\frac{
\mathbf B\cdot\nabla(\theta+\lambda)
-\iota(\rho)\mathbf B\cdot\nabla\zeta
}{
|\mathbf B|\,|\nabla(\theta+\lambda)|
}
\right|.
$$

嵌套性：

$$
\sqrt g
\text{ 不变号。}
$$

最后再看 DESC initial force residual。真正有用的结果应该表现为：

- $\epsilon_{RZ}$ 小；
- $\epsilon_\lambda$ 小；
- $\epsilon_\rho$ 小；
- $\epsilon_{\mathrm{sfl}}$ 小；
- `is_nested=True`；
- DESC initial force residual 比 `lambda=0` 明显下降。

### 13.11 与 DESC 的关系

这个联合拟合并不等价于 DESC solve。它只是在外部线圈磁场中构造：

1. 近似嵌套磁面坐标；
2. 近似 straight-field-line 坐标；
3. 近似 $\iota(\rho)$。

DESC 接下来仍然要解：

$$
\mathbf J\times\mathbf B=\nabla p.
$$

所以它的目标是降低 DESC 初值难度，而不是保证最终一定有 MHD equilibrium。

如果联合拟合 residual 已经很差，说明目标区域可能没有好嵌套面，或者当前 $\psi$ 层/半径选错了。此时不应该硬跑 DESC。

如果联合拟合 residual 很好但 DESC 仍失败，说明问题更可能出在 fixed-boundary equilibrium/profile 兼容性或 DESC 求解鲁棒性，而不是外部线圈磁面识别。

### 13.12 第一版实验设计

建议第一版只做以下最小闭环。

输入：

- 目标边界 $\psi_{\mathrm{edge}}$；
- $N_\rho=6$ 到 10 层；
- 每层 $N_\alpha=32$ 到 64 条起点；
- 每条线采样 $N_\zeta=32$ 到 64 个截面；
- 追踪长度先取一个 field period。

输出三组初值比较：

1. `desc_default_scaled`；
2. `psi_surface_lambda0`；
3. `fieldline_joint_RZlambda`。

对每组记录：

- `RZ_fit_rms`;
- `lambda_phase_rms`;
- `rho_residual`;
- `sfl_residual`;
- `sqrtg_sign_flip_fraction`;
- `DESC initial force mean/max`;
- `DESC solve success/failure`。

如果第三组显著优于第二组，就说明联合 $R,Z,\lambda$ 的方向是对的。

## 13. 坐标是否唯一，以及和 Boozer 坐标的关系

这一节专门回答三个容易混淆的问题：

1. $(\rho,\theta,\zeta)$ 是否只是形式坐标？
2. 随意指定这些坐标会不会影响拟合结果？
3. 如果联合求出了 $R,Z,\lambda,\iota$，是否就唯一确定了每个面的 Boozer 坐标？

### 13.1 它们是形式坐标，但不是随便坐标

$(\rho,\theta,\zeta)$ 首先是一套参数坐标，也就是给三维区域贴标签的方式。它们不是物理空间点天然自带的唯一编号。

但它们也不是随便选都一样。我们希望它们满足一组物理和几何条件：

$$
\rho=\mathrm{const}
$$

应当是磁面；

$$
\zeta
$$

通常先固定为物理 toroidal angle $\phi$；

$$
\theta+\lambda
$$

应当是 straight-field-line poloidal angle。

所以更准确地说：

> $(\rho,\theta,\zeta)$ 是我们选择的磁坐标标签。它有规范自由度，但一旦选定规范，就会影响有限阶拟合的好坏。

在无限精度、无限模数下，同一个物理曲面族可以用很多不同的 $\theta$ 参数化表示。但在有限 Fourier-Zernike 阶数下，参数化会强烈影响：

- $R,Z,\lambda$ 是否光滑；
- 需要多少模数；
- 最小二乘是否病态；
- `is_nested()` 是否容易保持；
- DESC 初始 force residual 是否小。

因此，坐标不是唯一的，但绝不能随便指定。好的坐标应该让 $R,Z,\lambda$ 都尽量光滑、低模、非奇异，并接近 DESC 的 straight-field-line 假设。

### 13.2 三个坐标各自的规范自由度

#### $\rho$ 的自由度

$\rho$ 是磁面标签。只要 $f$ 单调，理论上

$$
\rho' = f(\rho)
$$

表示的是同一族磁面。为了固定这个自由度，我们通常选：

$$
\rho=\sqrt{\frac{\psi}{\psi_{\mathrm{edge}}}}
$$

或更物理地选归一化 toroidal flux：

$$
\rho=\sqrt{\frac{\Psi_t}{\Psi_{t,\mathrm{edge}}}}.
$$

第一版用 $\sqrt{\psi/\psi_{\mathrm{edge}}}$ 更方便，但它不是唯一真理。

#### $\zeta$ 的自由度

$\zeta$ 是 toroidal angle。第一版建议固定：

$$
\zeta=\phi.
$$

这样每个采样点的 $\zeta$ 由物理柱坐标直接给出，避免多一个 toroidal gauge。

但 Boozer 坐标中 toroidal angle 未必等于物理 $\phi$。更一般可以有

$$
\zeta_B=\phi+\omega(\rho,\theta,\phi).
$$

如果允许 $\omega$，坐标会更接近 Boozer，但问题更复杂。第一版先不引入 $\omega$，只拟合 $\lambda$。

#### $\theta$ 的自由度

$\theta$ 是最自由的。可以取几何角、射线角、磁力线初相位，或者某个已修正角。

若固定了 $\theta$，则 $\lambda$ 的意义是：

$$
\vartheta=\theta+\lambda,
$$

其中 $\vartheta$ 是 straight-field-line 角。

如果重新定义

$$
\theta'=\theta+f(\rho,\theta,\zeta),
$$

同一个 straight-field-line 角可以写成：

$$
\vartheta=\theta+\lambda=\theta'+\lambda',
$$

所以

$$
\lambda'=\lambda-f.
$$

这说明 $\lambda$ 本身不是绝对物理量；真正有意义的是组合

$$
\theta+\lambda.
$$

### 13.3 固定 $\theta,\zeta,\rho$ 后，$\lambda$ 是否唯一

固定以下内容：

1. $\rho$ 的定义；
2. $\zeta=\phi$；
3. 计算角 $\theta$ 的定义；
4. 外部线圈磁场 $\mathbf B$；
5. 要求 $\theta+\lambda$ 是 straight-field-line 角；

那么 $\lambda$ 基本由方程决定：

$$
\mathbf B\cdot\nabla(\theta+\lambda)
=
\iota(\rho)\mathbf B\cdot\nabla\zeta.
$$

等价于

$$
\mathbf B\cdot\nabla\lambda
=
\iota(\rho)\mathbf B\cdot\nabla\zeta
-\mathbf B\cdot\nabla\theta.
$$

这是一条沿磁力线的一阶微分方程。

在一张 irrational surface 上，如果存在光滑嵌套磁面，那么要求 $\lambda$ 是单值、周期、光滑函数，会选出唯一的 rotation number：

$$
\iota(\rho).
$$

选出 $\iota$ 后，$\lambda$ 只差一个 flux function：

$$
\lambda\mapsto \lambda+C(\rho).
$$

这个自由度可以用 gauge 消掉：

$$
\langle\lambda\rangle_{\theta,\zeta}=0.
$$

所以固定 gauge 后，$\lambda$ 可以认为唯一。

在 rational surface 上情况更微妙，因为磁力线闭合，可能有共振条件；若存在岛或混沌，则全局光滑 $\lambda$ 不存在。这时最小二乘 residual 无法压到零，反而可以作为拓扑诊断。

### 13.4 联合解出 $R,Z,\lambda,\iota$ 后得到的是什么

假设我们真的联合解出了

$$
R(\rho,\theta,\zeta),\quad
Z(\rho,\theta,\zeta),\quad
\lambda(\rho,\theta,\zeta),\quad
\iota(\rho),
$$

并且 residual 为 0，状态正常，即：

$$
\mathbf B\cdot\nabla\rho=0,
$$

$$
\mathbf B\cdot\nabla(\theta+\lambda)
=
\iota(\rho)\mathbf B\cdot\nabla\zeta,
$$

$$
\sqrt g
\text{ 不变号}.
$$

那么我们得到了什么？

得到的是一套 **嵌套磁坐标**，并且

$$
\vartheta=\theta+\lambda
$$

是一套 **straight-field-line poloidal angle**。

也就是说，磁力线在每个磁面上的 $(\vartheta,\zeta)$ 图中是直线：

$$
\frac{d\vartheta}{d\zeta}=\iota(\rho).
$$

这已经是非常强的结果。它说明我们找到了外部线圈磁场的一族嵌套磁面，并且找到了直线场坐标。

但这还不自动等于 Boozer 坐标。

### 13.5 为什么 straight-field-line 坐标不一定是 Boozer 坐标

Boozer 坐标比普通 straight-field-line 坐标多一个条件。

普通 straight-field-line 坐标要求磁力线是直线：

$$
\frac{d\vartheta}{d\zeta}=\iota(\rho).
$$

Boozer 坐标还要求磁场的协变角向分量是 flux functions。典型写法是：

$$
\mathbf B
=
\beta\nabla\rho
+I(\rho)\nabla\theta_B
+G(\rho)\nabla\zeta_B.
$$

也就是说，在 Boozer 坐标中：

$$
B_{\theta_B}=\mathbf B\cdot\frac{\partial\mathbf x}{\partial\theta_B}
=I(\rho),
$$

$$
B_{\zeta_B}=\mathbf B\cdot\frac{\partial\mathbf x}{\partial\zeta_B}
=G(\rho).
$$

它们不能依赖 $\theta_B,\zeta_B$。

等价地，Boozer Jacobian 满足类似

$$
\sqrt g_B
\propto
\frac{G(\rho)+\iota(\rho)I(\rho)}{B^2}.
$$

我们的方程只强制了磁力线直线化，没有强制

$$
B_{\theta}=I(\rho),
\quad
B_{\zeta}=G(\rho).
$$

因此联合求出的 $(R,Z,\lambda,\iota)$ 一般是 PEST-like 或 generic straight-field-line 坐标，不一定是 Boozer 坐标。

### 13.6 什么时候能唯一确定 Boozer 坐标

若外部线圈磁场存在良好的嵌套磁面，并且我们允许足够的角坐标变换，例如：

$$
\theta_B=\theta+\lambda_B(\rho,\theta,\zeta),
$$

$$
\zeta_B=\zeta+\omega_B(\rho,\theta,\zeta),
$$

再要求：

1. 磁力线直线：

   $$
   \mathbf B\cdot\nabla\theta_B
   =
   \iota(\rho)\mathbf B\cdot\nabla\zeta_B;
   $$

2. 协变分量为 flux functions：

   $$
   \mathbf B\cdot\partial_{\theta_B}\mathbf x=I(\rho),
   $$

   $$
   \mathbf B\cdot\partial_{\zeta_B}\mathbf x=G(\rho);
   $$

3. 角度周期性和 gauge，例如

   $$
   \langle\lambda_B\rangle=0,\quad
   \langle\omega_B\rangle=0;
   $$

那么 Boozer 坐标可以唯一到一些剩余规范自由度：

- flux label 重标记 $\rho\mapsto f(\rho)$；
- 每个磁面的角度原点平移；
- 符号和周期 convention；
- 少数整数线性角变换 convention。

如果我们固定 $\rho$、固定方向、固定角度平均值、固定 $\zeta_B$ 的 convention，那么 Boozer 坐标基本唯一。

### 13.7 只拟合 $R,Z,\lambda$ 能不能得到 Boozer

如果我们只固定

$$
\zeta=\phi
$$

并且只允许一个 poloidal 修正

$$
\vartheta=\theta+\lambda,
$$

那么通常只能得到 straight-field-line 坐标，不保证 Boozer。

要得到 Boozer，可能还需要引入 toroidal angle 修正：

$$
\zeta_B=\phi+\omega.
$$

这也是之前讨论 $G$ 修正时遇到的问题。只修正 poloidal angle，有时不足以同时满足 Boozer 的协变分量条件。

所以答案是：

> 联合解出 $R,Z,\lambda,\iota$ 且 residual 为 0，可以唯一确定一套带 gauge 的 straight-field-line 磁坐标；但不能自动唯一确定 Boozer 坐标。要 Boozer，还要额外求 $\omega$ 或等价的 Boozer Jacobian/covariant-field 条件。

### 13.8 这对拟合方案意味着什么

对 DESC 初值来说，我们不一定非要先得到完整 Boozer 坐标。DESC 的 `L_lmn` 本身就是 $\lambda$，它需要的是一套较好的 straight-field-line/PEST-like 初值。先做到：

$$
\mathbf B\cdot\nabla\rho\approx0,
$$

$$
\mathbf B\cdot\nabla(\theta+\lambda)
\approx
\iota(\rho)\mathbf B\cdot\nabla\zeta
$$

就已经比 `lambda=0` 强很多。

对最终物理评估，如果要准确 Boozer QS error，则还需要在得到可靠磁面后再做 Boozer 变换或 Boozer surface solve。不要把“DESC 初值坐标”和“最终 Boozer 坐标”混为一谈。

### 13.9 最清晰的层级关系

可以按以下层级理解：

1. **曲面坐标**：

   只是一套嵌套曲面参数化。

2. **磁坐标**：

   满足

   $$
   \mathbf B\cdot\nabla\rho=0.
   $$

3. **直线场坐标**：

   进一步满足

   $$
   \mathbf B\cdot\nabla(\theta+\lambda)
   =
   \iota(\rho)\mathbf B\cdot\nabla\zeta.
   $$

4. **Boozer 坐标**：

   进一步满足协变分量 flux-function 条件：

   $$
   B_{\theta_B}=I(\rho),
   \quad
   B_{\zeta_B}=G(\rho).
   $$

我们现在讨论的联合拟合 $R,Z,\lambda,\iota$，目标是到第 3 层。到第 4 层还需要额外 Boozer 条件。
