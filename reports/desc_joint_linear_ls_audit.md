# DESC 联合线性初值拟合审计报告

日期：2026-07-10

## 1. 结论摘要

当前 `cem_qh03` 联合线性拟合实验的 residual 不能直接解释为“物理拟合目标不对”。审计确认：

1. 当前 phase residual 首先被一个确定的 NumPy 索引 bug 严重污染。
   `A_beta[row0:row1, beta_ids] = -1` 会把每一行的许多 `beta` 列同时置为 `-1`，而不是每个采样点只关联所属磁力线的一个 `beta_s`。正确写法应使用成对索引：

   ```python
   rows = np.arange(row0, row1)
   A_beta[rows, global_beta_ids] = -1.0
   ```

   修正这一行后，`trace_phase_joint_iota` 的 phase RMS 从 `1.5703 rad` 降到 `0.37735 rad`，矩阵条件也从几乎完全依赖 `1e-8` 正则改善到最小奇异值约 `0.515`。因此原先的 `1.57 rad` 主要是代码错误，不是物理失败。

2. 修正 `A_beta` 后，初值仍然不嵌套，且 force 没有改善。主因是 `R/Z` 体几何自身已经翻折，而不是 `lambda` phase 方程先失效。

   在统一诊断网格上：

   - `sqrt(g)`：约 `81.54%` 为负、`18.46%` 为正，几何 Jacobian 已经换号；
   - `sqrt(g)_PEST`：约 `80.40%` 为负、`19.60%` 为正；
   - `theta_PEST_t = 1 + lambda_t <= 0`：约 `6.40%`；
   - force 中位数约 `1.72e-8`，但最大值约 `3.0e6`。

   这说明 force 均值主要由少量 Jacobian 奇异点放大。当前问题不是全体网格上都有很大的 MHD force mismatch，而是体坐标在近轴及部分中外层区域发生折叠，导致局部导数和 force 爆炸。

3. phase 方程本身与 DESC 定义一致，并非公式写错。DESC 使用

   $$
   \theta_{\mathrm{PEST}} = \theta_{\mathrm{DESC}} + \lambda,
   $$

   沿磁力线有

   $$
   \theta_{\mathrm{PEST}} = \beta_s + \iota(\rho)\zeta,
   $$

   因此

   $$
   \lambda(q_i)-\beta_s-\zeta_i\iota(\rho_i)=-\theta_i
   $$

   正是当前报告提出的数据型线性方程。问题是该方程只能拟合 straight-field-line 坐标，不能保证 `R/Z` 嵌套，也不能直接保证 MHD force balance。

4. 当前最优先的改进不应是继续扫描 `lambda` 符号或 LS 权重，而应先使 `R/Z` 初值成为严格可用的嵌套体坐标。`lambda/beta/iota` 应在这个几何基础上作为第二阶段拟合。

## 2. 审计范围与版本

检查了以下本项目代码：

- `stellarator_eval/desc_joint_ls.py`
- `scripts/desc_external_rzl_data_ls_experiment.py`
- `scripts/desc_joint_rzl_initial_guess_experiment.py`
- `scripts/desc_psi_volume_initial_guess_experiment.py`
- `reports/desc_problem_initial_guess_analysis.md`

检查了 DESC 中与当前流程直接相关的实现：

- `Equilibrium` 构造、basis 建立和 orientation 检查；
- `set_initial_guess` 及点云拟合接口；
- `ensure_positive_jacobian`；
- `FourierZernikeBasis` 和 stellarator symmetry；
- `is_nested`；
- `sqrt(g)_PEST`、`|F|_normalized` 和 iota/current profile 逻辑。

远端实际运行版本为 DESC `0.16.0`。本地 `../DESC` 当前源码提交为 `ab1c5a3`，关键文件哈希与远端安装版不一致。报告中的接口结论已用远端安装版源码再次核实；后续开发仍应消除这种源码漂移，否则本地阅读和远端行为可能继续分叉。

## 3. 当前一步的准确计算流程

### 3.1 输入和边界

1. 读取 `cem_qh03` case 和 evaluator 已有运行目录。
2. 从 `boozer_surface.npz` 重建 Simsopt 曲面。
3. 用实际 Biot-Savart 场计算边界环向磁通 `Psi`。
4. 将边界写成 VMEC/DESC 输入，再由 DESC 读为 `FourierRZToroidalSurface`。
5. 从 `axis_data.npz` 拟合 DESC 磁轴曲线。

本轮使用的典型设置为：

- `nfp = 3`
- DESC `L=M=N=6`
- `pressure=0`
- `current=0`
- `Psi=-0.0172506...`
- evaluator 边界 `psi_level=0.3`

### 3.2 DESC 初始对象构造

`build_equilibrium` 创建：

```python
Equilibrium(
    L=6, M=6, N=6,
    surface=desc_surface,
    axis=axis_curve,
    pressure=0,
    current=0,
    Psi=Psi,
    ensure_nested=False,
)
```

需要注意：即使 `ensure_nested=False`，`check_orientation` 仍默认为 `True`。DESC 会先从边界和轴生成默认 `R/Z` 初值，然后调用 `ensure_positive_jacobian`。如果坐标为左手系，它会整体执行 `theta -> -theta` 对应的系数变换，并同步翻转 iota/current profile 符号。

当前每次构造都出现 `Left handed coordinates detected` 警告。该警告本身不是失败，但说明后续外部数据必须明确使用哪一套 `theta` convention。

### 3.3 field-line 数据生成

对 8 个径向层：

```text
rho = 0.12, 0.22857, ..., 0.88
```

每层执行：

1. 在 `zeta=0` 的局部 `psi=rho^2*psi_edge` 等值线上取 24 个起点；
2. 用外部线圈 Biot-Savart 场追踪一个 field period；
3. 在 17 个 toroidal 截面保存 `R/Z`；
4. 用相对磁轴的

   $$
   \theta_{geom}=\operatorname{atan2}(Z-Z_{axis},R-R_{axis})
   $$

   作为 DESC 节点的几何角；
5. 对每条线沿 `zeta` 执行 `unwrap(theta_geom)`；
6. 记录每个点所属的磁力线 `beta_id`，以及所属 `rho` 层。

共得到：

- phase/interior 点：`8 * 24 * 17 = 3264`
- 独立 `beta_s`：`8 * 24 = 192`
- iota profile：`t0 + t2*rho^2 + t4*rho^4`

当前 phase 数据没有包含一个周期后的 endpoint，虽然 RK4 函数返回了 endpoint；因此联合 LS 没有直接使用 Poincare return map 约束 iota。

### 3.4 R/Z/L/beta/iota 线性系统

DESC stellarator-symmetric equilibrium 使用：

- `R_basis.sym = "cos"`
- `Z_basis.sym = "sin"`
- `L_basis.sym = "sin"`

所以 `L_basis` 已自动排除 flux-function 的 `m=n=0` 模式，DESC 的零 flux-surface-average lambda gauge 已由 basis 满足。

数据集为：

- `R/Z interior`：3264 个 field-line trace 点，权重 1；
- `R/Z boundary`：169 个边界点，权重 40；
- `R/Z axis`：32 个轴点，权重 20；
- phase：3264 行，权重固定为 1；
- beta gauge：每个 rho 层一行，默认权重 10。

未知量为：

$$
x=[c^R,c^Z,c^L,\beta,t].
$$

矩阵块为：

$$
A_Rc^R\simeq R,
$$

$$
A_Zc^Z\simeq Z,
$$

$$
A_Lc^L-A_\beta\beta-A_\iota t\simeq-\tilde\theta.
$$

这里所谓“联合”是一次求解同一个块矩阵，但 `R/Z` 和 phase 块实际上是解耦的。phase 方程不会修正 `R/Z`，`R/Z` 方程也没有显式的 nested/Jacobian 条件。

### 3.5 写回 DESC 和诊断

LS 求解后只写回：

```python
eq.R_lmn = fit.R_lmn
eq.Z_lmn = fit.Z_lmn
eq.L_lmn = fit.L_lmn
```

拟合出的 `beta` 和 `iota_coeffs` 只保存为诊断结果，不写入 DESC profile。

这对 fixed-current vacuum 问题并非简单 bug：DESC 不允许同时指定 `current` 和 `iota` profile。当前拟合 iota 的合理角色应是构造 lambda 的辅助变量或弱先验，而不是直接替代 `current=0` 物理约束。

随后检查：

- `eq.is_nested()`：实际检查 `sqrt(g)_PEST` 是否全域同号；
- boundary mismatch；
- `|F|_normalized` 和 `<|F|>_vol`；
- 可选 DESC solve。

目前所有联合变体在外部拟合后都不是 nested，因此不应进入正式 force solve。

## 4. 发现的问题

### 4.1 严重：`A_beta` 的高级索引写错

当前代码：

```python
A_beta[row0:row1, global_beta_ids] = -1.0
```

NumPy 会把 `global_beta_ids` 指定的整组列应用到整个 row slice。以 4 行、3 条线为例，当前写法得到：

```text
[-1 -1 -1]
[-1 -1 -1]
[-1 -1 -1]
[-1 -1 -1]
```

期望结果则是每行只有一个非零元素：

```text
[-1  0  0]
[-1  0  0]
[ 0 -1  0]
[ 0  0 -1]
```

这个 bug 直接破坏了“每个采样点属于一条磁力线”的物理含义。

隔离修正结果：

| 指标 | 当前代码 | 修正 beta 索引 |
|---|---:|---:|
| phase RMS | 1.57026 rad | 0.37735 rad |
| LS residual sum | 8048.57 | 465.92 |
| 最小奇异值 | 约 1e-8 | 0.515 |
| nested | false | false |
| 原脚本 force mean | 2.382 | 3.410 |
| 原脚本 force p95 | 4.08e-4 | 3.04e-4 |

因此修正 beta 索引明显修复了 phase 拟合，但没有修复 R/Z 嵌套性。

### 4.2 严重：R/Z 最小二乘没有任何 nested 约束

修正 beta 后，`R/Z` 拟合残差约为：

- `R RMS = 7.78 mm`
- `Z RMS = 7.75 mm`
- 最大误差约 `41 mm`

而各层平均小半径约为 `8 mm` 到 `62 mm`。尤其内层，几何拟合误差已经与磁面尺寸同量级。

更直接的证据是 `sqrt(g)` 本身换号：

| 指标 | direct lambda minus | 修正 beta 的 joint phase |
|---|---:|---:|
| `sqrt(g)` 负值比例 | 81.54% | 81.54% |
| `sqrt(g)` 正值比例 | 18.46% | 18.46% |
| `sqrt(g)_PEST` 正值比例 | 20.35% | 19.60% |
| `theta_PEST_t<=0` 比例 | 6.57% | 6.40% |

两种 lambda 方案拥有完全相同的 `sqrt(g)` 统计，因为它们的 `R/Z` 相同。这证明主问题是 R/Z 映射，而不是 lambda。

数据型 block LS 只能最小化点位置误差；它不能阻止两层曲面之间相交，也不能阻止 Zernike 体插值在没有采样的区域振荡。

### 4.3 严重：当前 force mean 在非嵌套状态下不具备稳定解释

`force_stats` 报告的是网格点上 `abs(|F|_normalized)` 的普通算术均值，不是体积加权平均。对于非嵌套坐标，少数 Jacobian 奇异点会产生极大值，且结果强烈依赖 radial grid。

在更密的统一审计网格上，修正 beta 的 joint phase 结果为：

| 分位数 | `|F|_normalized` |
|---|---:|
| p50 | 1.72e-8 |
| p90 | 1.74e-5 |
| p95 | 1.72e-4 |
| p99 | 5.64e-2 |
| p99.5 | 4.22e-1 |
| p99.9 | 1.81e2 |
| max | 3.01e6 |

最大峰值主要出现在近轴和 `rho≈0.75`。因此当前 `initial_force_mean_abs_normalized` 主要是在测“有没有局部坐标奇异点”，不是在稳定衡量全局 equilibrium 接近程度。

`<|F|>_vol` 在 Jacobian 换号时同样不可信，因为体积分和体积本身依赖有符号 Jacobian；项目已有结果中甚至出现负的 raw volume average。

### 4.4 高：orientation 处理没有作为统一数据约定

DESC 构造对象时会自动把左手系转换为右手系；外部 LS 随后直接覆盖 `R/Z/L`，但没有记录或应用这个 orientation 变换。

此前的 `theta_flip` variant 只翻了 phase 的 `theta` target，没有同步翻转：

- interior R/Z 节点；
- boundary 节点；
- phase basis 节点；
- 对应 iota convention。

这种测试本身是不一致的。隔离实验表明：当 interior、boundary 和 phase 全部一致执行 `theta -> -theta` 时，iota 符号从正变负，但 force 和 phase RMS 不变。这符合纯坐标变换，说明 iota 正负差异主要来自 convention，而不是物理拟合优劣。

### 4.5 高：boundary mismatch 目前混入了参数化差异

DESC 的 `ensure_positive_jacobian` 可能重新参数化 `eq.surface`。当前 `boundary_mismatch` 却在相同数值 `theta/zeta` 上逐点比较原始 `desc_surface` 和 `eq.surface`。

构造后、外部拟合前就已有约 `4.9 cm` 的 surface mismatch，这不可能解释为物理边界已经改变；它主要反映 `theta` 参数化发生翻转或相移。

边界保真应使用以下之一：

- 对齐 orientation 后再逐点比较；
- 使用最近点/Chamfer/Hausdorff 距离比较几何集合；
- 直接比较经过同一 convention 变换后的 Fourier 系数。

### 4.6 中：正则强度的实现语义不正确或至少极易误解

配置名为 `ridge=1e-8`，实际增广矩阵中的正则行幅度也是 `1e-8`，因此目标函数中的平方权重约为 `1e-16`。

如果配置希望表达标准 Tikhonov 系数

$$
\|Ax-b\|^2+\alpha\|Dx\|^2,
$$

则增广矩阵应使用 `sqrt(alpha)*D`。当前正则几乎没有抑制高阶导数。

修正 beta 后拟合出的 `lambda` 仍有：

- RMS 约 `0.92 rad`
- 最大绝对值约 `5.85 rad`
- `max|lambda_t|` 约 `8.35`
- `1+lambda_t<=0` 约占 `6.4%`

这说明 lambda 坐标映射本身也有局部不可逆区域，需要导数正则或显式 rejection 条件。

### 4.7 中：iota 在一个周期内辨识不足，且没有使用 return endpoint

当前只追踪一个 field period，并且 phase 数据不包含周期末 endpoint。每条磁力线又有自由 `beta_s`，周期 lambda 也能吸收部分线性趋势。因此 iota、beta 和 lambda 存在较强相关性。

报告中原本建议：

- 使用 return map 给 iota 初估；
- 追踪多个周期；
- 或给 iota 加弱先验。

当前实现没有落实这些约束。修正 beta 后拟合的 iota profile 与 Boozer surface iota 仍有显著差异；完整 orientation 翻转只改变符号，不消除幅值差异。

### 4.8 中：数据集权重依赖采样数量

当前目标中每个点直接使用固定权重，没有按数据集样本数归一化。因此增加 `trace-alpha` 或 `trace-zeta` 会自动提高 interior 相对于 axis/boundary 的总权重。

典型总权重约为：

- interior：`3264 * 1 = 3264`
- boundary：`169 * 40 = 6760`
- axis：`32 * 20 = 640`

axis 对近轴几何的约束明显偏弱，而审计中最大的 force 峰恰好出现在近轴。

此外，CLI 的 `--lambda-weight` 只作用于 direct lambda 数据；phase constraint 的权重在构造时固定为 1，所以该参数不能调节 joint phase 方程。

### 4.9 中：拟合出的 iota 不等于 DESC 的物理 profile

联合 LS 输出的 `iota_coeffs` 没有写入 `eq.iota`。对于当前 `current=0` vacuum equilibrium，这种做法可以成立，因为拟合 iota 只是外部线圈场的 field-line phase slope。

但代码和报告必须明确区分：

- `iota_phase_fit`：用于构造 straight-field-line lambda 的辅助量；
- `eq.iota`：DESC 在 fixed-iota 物理问题中的输入 profile；
- `eq.compute("iota")`：当前 fixed-current equilibrium 从几何和 current 推出的 iota。

三者目前没有一致性诊断，容易误把 LS iota 当成已传给 DESC 的物理初值。

### 4.10 中：本地 DESC 源码与远端运行版不同

本地 `../DESC` 和远端 DESC 0.16.0 的关键文件哈希不同。当前审计确认核心接口语义一致，但后续若依赖更细的 optimizer、constraint 或 coordinate mapping 行为，必须固定同一版本或让远端以 editable install 使用同一源码。

## 5. residual 到底来自哪里

可以把当前 residual 分成三层：

### 5.1 phase LS residual

原 `1.570 rad`：主要来自 `A_beta` 索引 bug。

修正后 `0.377 rad`：来自以下因素的组合：

- `L=M=N=6` 的有限谱表示；
- 只追踪一个 field period；
- 没有 return-map iota 先验；
- 简单 `atan2` 几何角在强变形截面上的参数化质量；
- field-line 点云可能存在层内漂移或局部非光滑。

`0.377 rad` 仍不算好，但它不是当前 force 爆炸的主因。

### 5.2 nested/Jacobian residual

主要来自 R/Z 体谱拟合。`sqrt(g)` 在约 18.5% 的点上换号，说明在加入 lambda 之前几何已经折叠。

这是当前最关键的失败。

### 5.3 DESC force residual

大多数网格点的 force 很小，但少量 Jacobian 奇异点产生巨大峰值。当前 mean/max 更接近“坐标奇异性指标”，不能据此判断外部 vacuum field 与 MHD equilibrium 在全局上相差多少。

因此准确结论是：

> 当前失败首先是实现 bug，其次是 R/Z 初值几何不嵌套；straight-field-line 物理目标本身没有写错，但数据型线性 LS 的目标不完整，不能单独保证可供 DESC 求解的体坐标。

## 6. 下一步最可能有效的改进方向

### P0：先修确定性错误和诊断

#### 6.1 修复 `A_beta` 索引并补单元测试

难度：低，约 0.5 天。

至少增加三个测试：

1. 每个 phase row 的 `A_beta` 恰好一个非零元素；
2. 合成 `lambda/beta/iota` 数据可恢复已知系数；
3. 打乱点顺序后结果不变。

#### 6.2 建立统一 orientation convention

难度：低到中，约 1 天。

建议让数据生成阶段显式返回 `theta_sign`，并对以下内容统一应用：

- interior nodes；
- phase theta；
- boundary nodes；
- iota 符号；
- boundary comparison。

不要再通过只翻 phase target 的 variant 判断符号。

#### 6.3 重写初值诊断门槛

难度：低，约 1 天。

在任何 force 指标之前报告：

- `sqrt(g)` min/max、换号比例；
- `sqrt(g)_PEST` min/max、换号比例；
- `min(1+lambda_t)` 和非正比例；
- force p50/p95/p99/p99.9/max；
- 按 rho 分层的异常位置。

只要 Jacobian 换号，就把 force mean 标记为“不适合作物理收敛指标”。

### P1：先得到 nested R/Z，再拟合 lambda

这是最可能真正推进 `cem_qh03` 的方向。

#### 6.4 将 boundary 和 axis 作为精确约束

难度：中，约 2 到 4 天。

不要再把 fixed boundary 和 axis 当普通软权重点。可以使用 constrained LS/KKT/null-space：

$$
\min_c \|A_{int}c-b_{int}\|^2+\alpha\|Dc\|^2,
$$

满足：

$$
A_{boundary}c=b_{boundary},
$$

$$
A_{axis}c=b_{axis}.
$$

这至少能消除近轴和 rho=1 的大偏差，并使后续 fixed-boundary solve 不需要突然修正边界。

#### 6.5 使用“嵌套优先”的 R/Z 构造，而不是全局无约束点拟合

难度：中到高，约 4 到 7 天。

候选实现按推荐顺序：

1. 每个 rho 层先独立拟合一个光滑 Fourier surface；
2. 对相同 `(m,n)` 系数沿 rho 做低阶、带端点约束的径向拟合；
3. 每加一层都检查 `sqrt(g)`，只接受保持同号的 continuation；
4. 必要时缩小高阶 radial/toroidal 模式，而不是继续增加 LS 权重；
5. 将 Jacobian sign 作为硬 rejection 条件。

线性 LS 不能直接施加全域 `sqrt(g)>0`，但分层 continuation 比一次性全体积拟合稳定得多。

#### 6.6 分阶段拟合

难度：中，约 2 到 3 天。

推荐流程：

```text
Stage A: 只拟合 R/Z，L=0
  -> boundary/axis 精确
  -> sqrt(g) 全域同号

Stage B: 固定 R/Z，拟合 L/beta/iota
  -> phase residual
  -> 1+lambda_t > 0
  -> sqrt(g)_PEST 全域同号

Stage C: 才计算 force，并进入 DESC solve
```

当前一次性输出虽然叫 joint LS，但 R/Z 和 phase 本来就是块解耦的。显式分阶段不会损失数学耦合，反而能建立正确的验收门槛。

### P2：改善 phase/iota 拟合

#### 6.7 加 return-map iota 先验和多周期追踪

难度：中，约 2 到 4 天。

建议：

- 把 RK4 endpoint 纳入相位推进诊断；
- 至少追踪 2 到 4 个 field periods；
- 对漂移过大的线/层降权或剔除；
- 加入

  $$
  \sqrt{w_\iota}(P(\rho)t-\iota_{return})\simeq0
  $$

  的弱先验；
- 对比每层各条线的 rotation number 离散度。

#### 6.8 正确实现正则和导数惩罚

难度：中，约 1 到 3 天。

需要：

- 使用 `sqrt(alpha)*D` 增广；
- 默认启用列缩放；
- 对 `L` 重点惩罚高 `m/n/l` 和 `lambda_t`；
- 对 R/Z 重点惩罚高 radial order；
- 将 `min(1+lambda_t)>0` 作为拒绝条件。

#### 6.9 改善几何角参数化

难度：中，约 2 到 4 天。

简单圆形 `atan2` 对强拉长截面会造成角点聚集。可比较：

- 用每截面的 `a_R/a_Z` 归一化后再 `atan2`；
- 用 psi 等值线的射线参数；
- 用 surface Fourier 参数反解几何角。

验收标准不是 phase RMS 单独最小，而是 R/Z residual、`sqrt(g)` 和 `1+lambda_t` 同时改善。

### P3：必要时进入非线性修正

#### 6.10 少量 Gauss-Newton 步骤

难度：高，约 1 到 2 周。

在线性初值已经 nested 后，再加入：

$$
r_\rho=\mathbf B_{coil}\cdot\nabla\rho,
$$

$$
r_{sfl}=\mathbf B_{coil}\cdot\nabla(\theta+\lambda)
-\iota(\rho)\mathbf B_{coil}\cdot\nabla\zeta.
$$

只迭代少数步，目标是把初值送入 DESC basin，而不是替代 DESC equilibrium solve。

在 R/Z 仍然非嵌套时直接做这一步意义不大，因为 Jacobian 奇异会使梯度和 Jacobian 本身不稳定。

## 7. 回归验证计划

修正后必须同时跑：

- `cem_qh03`：目标是从不可解向可解推进；
- `cem_1`、`cem_3`：确认救援初值不破坏原本可解 case。

每个 case 至少比较：

1. DESC 默认初值；
2. constrained/nested RZ + `lambda=0`；
3. constrained/nested RZ + joint phase lambda；
4. solve 前后的 nested、boundary、force quantiles 和 optimizer objective。

建议的硬门槛：

- `sqrt(g)` 全域同号；
- `sqrt(g)_PEST` 全域同号；
- `min(1+lambda_t)>0`；
- fixed boundary 几何误差在明确容差内；
- 再比较 force residual 和 DESC solve。

## 8. 推荐实施顺序

最短有效路径是：

1. 修 `A_beta` 索引并加合成测试；
2. 固定 orientation 和诊断定义；
3. 暂停调 lambda，先构造 boundary/axis 精确且 `sqrt(g)` 同号的 R/Z 体；
4. 固定 R/Z 后重新拟合 `L/beta/iota`；
5. 加 return-map iota 先验与正确正则；
6. 在 `cem_qh03` 通过 nested 门槛后才跑 DESC solve；
7. 用 `cem_1/cem_3` 做回归。

按这个顺序，前 1 到 2 天可以消除代码和诊断歧义；真正解决 R/Z 嵌套体初值预计需要约 1 周量级。若线性/continuation 方案仍无法稳定保持 Jacobian，再进入 Gauss-Newton，工作量会上升到 1 到 2 周。

