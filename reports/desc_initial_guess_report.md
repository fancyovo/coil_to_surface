# DESC 初值输入能力与本项目可提供初值报告

## 摘要

本报告基于远端已安装的 DESC `0.16.0` 源码检查，重点关注 DESC 能接受哪些输入、哪些输入真正会进入 equilibrium 初值，以及我们当前 `local_surface_evaluator` 默认流程能提供哪些高质量初值。

核心结论：

- DESC 不只接受 VMEC/DESC input 文件；Python API 里 `Equilibrium(...)` 和 `eq.set_initial_guess(...)` 能接受更丰富的几何初值。
- 我们之前给 DESC 的方式只用了“最外层边界面”，属于较弱初值。即使边界本身来自 Boozer 面，DESC 内部体坐标仍主要靠“边界缩放到轴”的默认构造。
- 对 `cem_qh03` 的实验表明，加入磁轴初值后 DESC residual 明显下降，但仍不够好。这说明“磁轴初值”有帮助，但还不等价于“好体坐标初值”。
- 我们当前最有价值的 DESC 初值来源其实是 $\psi$：它能生成多层嵌套等值面点云，最适合喂给 `eq.set_initial_guess(nodes, R, Z, lambda)` 或拟合成 `R_lmn/Z_lmn/L_lmn`。

## DESC 版本和源码依据

检查环境：

- DESC: `0.16.0`
- 源码来自当前 Python 环境中已安装的 `desc` 包。

关键源码位置：

- `desc/equilibrium/equilibrium.py`
- `desc/equilibrium/initial_guess.py`
- `desc/equilibrium/utils.py`
- `desc/input_reader.py`

`Equilibrium` 构造函数签名：

```python
Equilibrium(
    Psi=1.0,
    NFP=None,
    L=None, M=None, N=None,
    L_grid=None, M_grid=None, N_grid=None,
    pressure=None,
    iota=None,
    current=None,
    electron_temperature=None,
    electron_density=None,
    ion_temperature=None,
    atomic_number=None,
    anisotropy=None,
    surface=None,
    axis=None,
    sym=None,
    spectral_indexing=None,
    check_orientation=True,
    ensure_nested=True,
    **kwargs,
)
```

其中 `kwargs` 还可以直接传：

```python
R_lmn=...
Z_lmn=...
L_lmn=...
```

如果不传 `R_lmn/Z_lmn`，构造函数会自动调用：

```python
self.set_initial_guess(ensure_nested=ensure_nested)
```

这点非常关键：只传 `surface` 和 `axis` 时，DESC 仍会自己构造体内 flux surfaces；而直接传 `R_lmn/Z_lmn/L_lmn` 或调用 `set_initial_guess(nodes, R, Z, lambda)` 才能真正提供体坐标初值。

## DESC 可接受的输入类型

### 1. Python API: `Equilibrium(...)`

这是最直接、最灵活的入口。

几何输入：

- `surface`
  - `desc.geometry.Surface` 对象。
  - 常见是 `FourierRZToroidalSurface`，表示 LCFS 边界。
  - 也可以是 `ZernikeRZToroidalSection`，表示 Poincare section 类型边界。
  - 也可以是数组，形状类似 `(k, 5)`，每行 `[l, m, n, R, Z]`。
- `axis`
  - `FourierRZCurve` 对象。
  - 也可以是数组，形状类似 `(k, 3)`，每行 `[n, R, Z]`。
  - 如果不传，默认从 surface 的 centroid/m=0 模估计。
- `R_lmn/Z_lmn/L_lmn`
  - 通过 `kwargs` 直接传入。
  - 这是最强几何初值：直接指定 DESC 内部 Fourier-Zernike 体坐标系数。
  - `L_lmn` 是 DESC 的 angular correction，即 Boozer/straight-field-line 类似的角坐标修正量，不是我们的 $\psi$。

物理 profile 输入：

- `pressure`
- `iota`
- `current`
- `electron_temperature`
- `electron_density`
- `ion_temperature`
- `atomic_number`
- `anisotropy`

profile 可以是：

- DESC Profile 对象，例如 `PowerSeriesProfile`、`SplineProfile`、`HermiteSplineProfile`。
- 标量，作为常数 profile。
- 一维数组，作为 power series 系数。
- 二维数组 `(k, 2)`，每行 `[mode, coefficient]`。

限制：

- `iota` 和 `current` 不能同时指定。
- kinetic profiles 和 `pressure/anisotropy` 不能同时指定。
- 若 `current=None` 且 `iota=None`，DESC 默认 `current=0`。
- 若 `pressure=None` 且没有 kinetic profiles，DESC 默认 `pressure=0`。

### 2. `Equilibrium.from_input_file(path)`

DESC 支持两类 input file：

- DESC 原生 input。
- VMEC `&INDATA` input，会先由 `InputReader.vmec_to_desc_input` 转换为 DESC input。

DESC 原生 input 支持的主要几何字段：

- `R1/Z1`：固定边界 surface 系数。
- `R0/Z0`：磁轴初值系数。
- `L_rad/M_pol/N_tor`：谱分辨率。
- `L_grid/M_grid/N_grid`：实空间网格分辨率。
- `bdry_mode`：`lcfs` 或 `poincare`。

VMEC input 支持的主要几何字段：

- `RBC/RBS/ZBC/ZBS`：边界 Fourier 系数。
- `RAXIS_CC/RAXIS_CS/ZAXIS_CC/ZAXIS_CS`：磁轴 Fourier 初值。
- `PHIEDGE`：总 toroidal flux，对应 DESC 的 `Psi`。
- `MPOL/NTOR/NFP/LASYM`：分辨率、场周期、对称性。

profile/物理字段：

- `AM`、`PRES_SCALE`、`PMASS_TYPE`：压力。
- `AI`、`PIOTA_TYPE`：iota。
- `AC`、`CURTOR`、`NCURR`、`PCURR_TYPE`：current。

注意：

- `from_input_file` 最终还是构造 `Equilibrium(**inputs)`。
- 因此 input file 中的 axis 会影响默认体面初值。
- 但如果之后又手动新建 `Equilibrium(surface=..., pressure=..., current=...)`，就会丢掉 input file 里的 axis 初值。
- 我们之前的 DESC 脚本就是这种情况：先写 input，再手动构造新 `Equilibrium`，导致 axis 初值没有真正用上。

### 3. `eq.set_initial_guess(...)`

这是最关键的初值入口。DESC 源码列出的可接受形式如下。

#### 3.1 不传参数

```python
eq.set_initial_guess()
```

行为：

- 使用 `eq.surface`。
- 把边界缩放到 `eq.axis`。
- `L_lmn=0`。
- 如果 `ensure_nested=True` 且初值不嵌套，会尝试用 `GoodCoordinates` 小优化修正。

这是最弱但最常用的初值方式。

#### 3.2 传一个 Surface

```python
eq.set_initial_guess(surface)
```

行为：

- 用给定 surface 缩放构造体面。
- 若 surface 有 `rho` 标签，可作为 interior surface 使用。
- 没有 axis 时，用 surface 自身估计 axis。

#### 3.3 传 Surface + Curve

```python
eq.set_initial_guess(surface, axis_curve)
```

行为：

- 用 surface 和显式 axis 构造体面。
- 仍然是“surface 到 axis 的缩放”，不是多层真实磁面。

#### 3.4 传另一个 Equilibrium

```python
eq.set_initial_guess(eq_old)
```

行为：

- 复制已有 equilibrium 的 `R_lmn/Z_lmn/L_lmn`。
- 同时复制 axis 系数。

这是 continuation 最自然的方式。

#### 3.5 传 DESC/VMEC 输出文件路径

```python
eq.set_initial_guess(path_to_desc_or_vmec_output)
```

行为：

- 加载已有 DESC equilibrium 或 VMEC output。
- 复制其体面和 lambda。

这适合“上一层已经成功的 DESC 解作为下一层初值”。

#### 3.6 传 grid + 点云

```python
eq.set_initial_guess(nodes, R, Z)
eq.set_initial_guess(nodes, R, Z, lambda)
```

其中：

- `nodes` 可以是 DESC `Grid`，也可以是形状 `(k, 3)` 的数组。
- 每个 node 是 `(rho, theta, zeta)`。
- `R/Z/lambda` 是对应节点上的值。

这是对我们最重要的入口。原因是我们可以从 $\psi$ 直接生成多层嵌套面点云：

$$
\psi(R, Z, \Phi)=\psi_0(\rho)
$$

然后把每个点映射成：

$$
(\rho_{\mathrm{DESC}}, \theta_{\mathrm{init}}, \zeta)
  \mapsto (R, Z).
$$

若暂时没有可靠的 straight-field-line 修正，可先取 `lambda=0`。后续如果有 field-line conjugacy 或 Boozer 参数化，可以把角坐标修正写入 `lambda`，这会比单纯边界缩放强很多。

### 4. `Equilibrium.from_near_axis(...)`

DESC 还支持从 near-axis 解初始化：

```python
Equilibrium.from_near_axis(na_eq, r=0.1, M=8, ...)
```

`na_eq` 需要是 pyQSC 或 pyQIC 的 near-axis solution。

这不是我们当前默认流程的直接输出，但理论上如果以后从线圈反推出 near-axis/QSC 初值，可以接这条路径。

### 5. HDF5 / pickle / DESC output

DESC `Equilibrium` 继承 `IOAble`，支持：

```python
eq.save(...)
Equilibrium.load(...)
```

以及从 DESC output / VMEC output 文件恢复 equilibrium。对我们有用的场景是 continuation：

1. 小磁面 DESC 成功。
2. 保存为 DESC output。
3. 更大磁面调用 `set_initial_guess(previous_desc_output)` 或 `set_initial_guess(eq_previous)`。

## 我们当前默认流程能提供哪些初值

当前 `local_surface_evaluator` 默认/完整流程已经产生以下高质量信息。

### 1. 磁轴

文件：

- `axis_data.npz`

包含：

- `phi`
- `R`
- `Z`
- `R_phi`
- `Z_phi`
- `best_R`
- `best_Z`
- `nfp`

可用于 DESC：

- 拟合成 `FourierRZCurve`。
- 或写入 VMEC 风格 `RAXIS_CC/ZAXIS_CS`。
- 或写入 DESC 原生 `R0/Z0`。

在 `cem_qh03` 上，8 阶 Fourier 拟合磁轴残差：

- `R rms = 1.39e-7 m`
- `Z rms = 2.34e-7 m`

实测效果：

- 原始 DESC 平均 normalized force 约 `1.09e5`。
- 加入磁轴初值后降到约 `5.86e2`。
- 说明磁轴初值确实有价值。

### 2. $\psi$ 模型

文件：

- `psi_model.npz`

包含：

- 多项式/Fourier 系数。
- 模数 `poly_degree`、`m_tor`。
- 磁轴采样。
- validation angle residual。

当前形式：

$$
\psi(x,z,\Phi)
=x^2+\sum c_{pqm}x^p z^q T_m(nfp\,\Phi).
$$

其中：

$$
x=\frac{R-R_{\mathrm{axis}}(\Phi)}{a},\quad
z=\frac{Z-Z_{\mathrm{axis}}(\Phi)}{a}.
$$

它能提供的 DESC 初值非常强：

1. 多层等 $\psi$ 面。
2. 每层的点云 `(R, Z, phi)`。
3. 局部几何角 `theta`。
4. 层标签 `rho_DESC`。

推荐映射：

$$
\rho_{\mathrm{DESC}} = \sqrt{\psi/\psi_{\mathrm{edge}}}.
$$

这样可把多层 $\psi$ 面喂给：

```python
eq.set_initial_guess(nodes, R, Z, lambda)
```

其中：

```python
nodes[:, 0] = rho_DESC
nodes[:, 1] = theta_DESC
nodes[:, 2] = zeta
```

如果暂时没有更好的角坐标修正：

```python
lambda = 0
```

这已经比“只给边界，然后 DESC 线性缩到轴”强。

### 3. 候选等 $\psi$ 面

当前流程已有：

- `surface_points_from_level(...)`
- `surface_points_from_level_gpu(...)`
- `level_*/boozer_surface.npz`

可提供：

- 每个 $\psi_0$ 的隐式面点云。
- 拟合后的 `SurfaceXYZTensorFourier`。
- Simsopt Boozer Newton 修正后的 surface dofs。

对 DESC 的用途：

- 最外层：作为固定边界。
- 内层：作为体坐标初值。
- 多层一起：构造更稳定的 `R_lmn/Z_lmn/L_lmn`。

### 4. Simsopt Boozer 面

文件：

- `level_*/boozer_surface.npz`

包含：

- `dofs`
- `iota`
- `G`
- `psi_level`

优点：

- 已经通过 Simsopt Boozer residual 求解。
- surface 几何通常比原始 $\psi$ 等值面更接近 Boozer 坐标。

局限：

- 当前只保存单层面，没有保存多层体坐标。
- DESC 需要的是整个 plasma volume 初值，不只是 LCFS。
- 单层 Boozer 面直接转 `FourierRZToroidalSurface` 后，DESC 内部仍会自己缩放体面。

### 5. iota 初值

来源：

- `surface_screen.levels[*].iota_estimate`
- `best_surface.iota`
- `boozer_initial_iota.initial_iota_used`

可用于 DESC：

- 作为 `iota` profile 的常数初值。
- 但在 vacuum/zero-current 固定边界场景中，是否指定 `iota` 要谨慎，因为 DESC 不允许同时指定 `iota` 和 `current`。

当前建议：

- 如果走 zero-current vacuum-like equilibrium，继续 `current=0`。
- 如果要强行给 iota profile，需要明确物理语义，并避免和 `current` 冲突。

### 6. toroidal flux / PHIEDGE

当前流程可以用真实 Biot-Savart 场通过边界计算：

```python
ToroidalFlux(surface, field).J()
```

这比随便指定 `PHIEDGE` 更合理。

注意：

- 符号可能受坐标手性影响。
- DESC 曾提示 left-handed coordinate，必要时要统一 `theta` 符号、`iota/current` 符号和 surface orientation。

## 当前接 DESC 的问题

### 问题 1：只给了最外层边界

当前脚本主要流程：

1. 从 Simsopt Boozer surface 得到外边界。
2. 转为 `SurfaceRZFourier` / VMEC input。
3. DESC 从边界和默认/估计磁轴生成内部体面。

这会导致：

- 初始面可能 non-nested。
- 内部面形状不符合真实 $\psi$ 嵌套结构。
- 对极端截面形状尤其容易失败。

### 问题 2：曾经丢掉磁轴初值

如果流程是：

```python
eq0 = Equilibrium.from_input_file(input)
boundary = FourierRZToroidalSurface.from_input_file(input)
eq = Equilibrium(surface=boundary, ...)
```

那么 `input` 里的 axis 会被 `eq0` 读到，但第二个 `Equilibrium(...)` 又重新估计 axis，导致 axis 初值丢失。

正确做法之一：

```python
eq = Equilibrium.from_input_file(input_with_axis)
```

或者：

```python
eq = Equilibrium(
    surface=surface,
    axis=axis_curve,
    ...
)
```

但这仍只是“边界 + 轴”，不如多层体面初值。

### 问题 3：没有给 `lambda`

DESC 的 `lambda` 是角坐标修正。当前我们如果只给 `R/Z`，`lambda=0`，DESC 仍需要自己调整角坐标。

后续若能从 field-line conjugacy / Boozer 面参数化得到更接近 straight-field-line 的角坐标修正，应把它转成 `L_lmn` 或 `lambda` 点值。

### 问题 4：坐标手性

DESC 的警告：

- left-handed coordinates detected

这说明边界 Fourier 转换后的参数方向与 DESC 默认约定可能相反。影响：

- `theta` 符号。
- `iota` 符号。
- `current` profile 符号。

这不一定导致失败，但会让 iota/QS 图解释更容易混乱。后续应统一 surface orientation。

## 推荐的接入方案

### 方案 A：最小修正，边界 + 磁轴

做法：

1. 用最大连续分支 Boozer 面作为 DESC boundary。
2. 从 `axis_data.npz` 拟合 `FourierRZCurve`。
3. 直接用 Python API：

```python
eq = Equilibrium(
    Psi=phiedge,
    NFP=nfp,
    L=8,
    M=8,
    N=8,
    surface=boundary,
    axis=axis_curve,
    pressure=PowerSeriesProfile([0, 0, 0]),
    current=PowerSeriesProfile([0]),
    ensure_nested=True,
)
```

优点：

- 改动小。
- 已实验证明能降低 residual。

缺点：

- 对复杂边界仍可能 non-nested。
- 不能充分利用 $\psi$。

适合短期作为默认 DESC 接入的安全改进。

### 方案 B：多层 $\psi$ 面点云初值

做法：

1. 选最大可信 `psi_edge`。
2. 取多层：

```text
rho_DESC = 0.1, 0.2, ..., 1.0
psi = rho_DESC^2 * psi_edge
```

3. 对每层、每个 `theta/zeta`，用当前 $\psi$ 模型解出 `(R,Z)`。
4. 构造：

```python
nodes = np.column_stack([rho_DESC, theta, zeta])
eq.set_initial_guess(nodes, R, Z)
```

优点：

- 充分利用我们最擅长的 $\psi$ 局部磁面信息。
- 能直接给 DESC 一个嵌套体面族，而不是让 DESC 自己缩放。
- 预计能显著降低 non-nested 风险。

缺点：

- 初始 `theta` 只是几何角，不一定是 DESC 最优角。
- 若 $\psi$ 在外层不可靠，外层点云会污染 DESC 初值。
- 需要做好层筛选和 branch 过滤。

这是我认为最值得做的下一步。

### 方案 C：多层 Boozer/field-line 修正后初值

做法：

1. 对多层 $\psi$ 面分别跑 Simsopt Boozer。
2. 用 Boozer 参数化面族拟合 DESC 的 `R_lmn/Z_lmn`。
3. 若能得到角坐标修正，进一步构造 `L_lmn`。

优点：

- 初值更接近 straight-field-line 坐标。
- 对 DESC solve 最友好。

缺点：

- 每层 Boozer solve 有分支跳跃风险。
- 成本比方案 B 高。
- 多层 Boozer 面之间可能参数化不一致，需要统一 `(theta,zeta)` 标签。

适合作为精修方案，不建议马上作为默认。

### 方案 D：Continuation

做法：

1. 从很小磁面开始，DESC solve 成功。
2. 保存 `eq_small`。
3. 逐层扩大边界：

```python
eq_large.set_initial_guess(eq_small)
```

或：

```python
eq_large.set_initial_guess(path_to_small_desc_output)
```

优点：

- 符合 DESC 自身设计。
- 对复杂面更稳。

缺点：

- 总耗时更长。
- 需要管理多层边界和失败回退。

可以和方案 B 结合：第一层用 $\psi$ 点云，后续用 continuation。

## 优先级建议

### 第一优先级：修正当前 DESC 接入

必须改：

- 不要先 `from_input_file` 再丢弃 axis。
- 直接使用 `Equilibrium.from_input_file(input_with_axis)`，或 Python API 显式传 `axis=axis_curve`。
- DESC input 中加入磁轴初值。

这是低成本、确定有效的改进。

### 第二优先级：实现 $\psi$ 多层点云初值

新增函数建议：

```python
build_desc_initial_guess_from_psi(
    psi_model,
    psi_edge,
    rho_levels,
    ntheta,
    nzeta,
    lambda_mode="zero",
)
```

输出：

```python
nodes, R, Z, lambda
```

然后：

```python
eq = Equilibrium(surface=boundary, axis=axis_curve, ...)
eq.set_initial_guess(nodes, R, Z, lambda, ensure_nested=True)
```

这应当作为下一轮主要实验。

### 第三优先级：加入多层质量筛

每一层 $\psi$ 面进入 DESC 前应检查：

- 是否闭合。
- 半径是否小于 `a` 外推边界。
- 一周期 field-line drift。
- 与相邻层是否嵌套。
- Fourier 拟合残差。

不要把所有 $\psi$ 层无条件喂给 DESC。

### 第四优先级：处理 `lambda/L_lmn`

初期可以 `lambda=0`。若仍然 residual 高，再考虑：

- 从 field-line conjugacy 得到角坐标修正。
- 从多层 Boozer surface 参数化反推 `lambda`。
- 或先让 DESC 的 `GoodCoordinates` / equilibrium solve 自己优化 `L_lmn`。

## 对 `cem_qh03` 的具体解释

当前结果：

- 最大连续分支面：`a=0.12, psi=0.3`
- Simsopt Boozer 面本身可解。
- Poincare 修正后显示点云和边界截面基本对应。
- DESC 仅用边界时失败严重。
- 加磁轴初值后 residual 显著改善，但仍不可信。

因此更合理的判断是：

1. Boozer 面作为一张边界面是有意义的。
2. DESC 当前失败不一定说明边界完全错；更可能是 DESC 初始体坐标太差，尤其是内部嵌套面构造不适合这个强 3D/强变形边界。
3. 下一步应该用 $\psi$ 构造多层体面初值，再喂给 DESC，而不是继续只调 DESC solve 容差。

## 推荐下一步实验

建议以 `cem_qh03` 为例做三组对照：

1. `boundary only`
   - 当前原始方式。
2. `boundary + axis`
   - 已验证有明显改善。
3. `boundary + axis + psi multi-surface initial guess`
   - 新实现。

评估指标：

- `eq.is_nested()` 初值是否嵌套。
- DESC 初始 force residual。
- DESC 最终 force residual。
- 是否出现 non-nested warning。
- solve 时间。
- DESC 的 boundary、B contours、iota 图。

预期：

- 第 3 组应显著优于前两组。
- 如果第 3 组仍失败，再说明问题可能不仅是初值，而是边界与 zero-pressure/zero-current DESC equilibrium 的物理兼容性不足，或者 Boozer 面分支/手性仍有问题。
