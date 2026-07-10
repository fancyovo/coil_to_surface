# 稠密线性最小二乘 Clebsch $\alpha$ 初值实验报告

日期：2026-07-10

对象：`cem_qh03`，外层拟合不变量取 $s_{\mathrm{edge}}=0.16$

## 1. 先给结论

### 1.1 已经确认成功的部分

1. **真实环向磁通标定是稳定且几乎线性的。**
   由多个环向截面对 $B_\phi$ 积分得到

   $$
   |\Phi_T(s_{\mathrm{edge}})|=1.385027\times10^{-3}\ {\rm Wb}.
   $$

   与已有 Boozer 曲面直接积分的磁通幅值只差 $0.265\%$。不同环向截面之间的
   最大相对标准差为 $1.61\times10^{-4}$。

2. **十万级点云上的普通线性最小二乘可以稳定求出 $\alpha$。**
   正式实验使用 120,000 个均匀训练点和 60,000 个独立验证点，主求解器是单卡
   FP64 GPU QR。没有使用非线性优化、迭代初值、凸约束或谱正则。

3. **求出的 $\lambda$ 确实把磁力线拉直。**
   八个场周期、三层磁面、每层四条磁力线的平均直线残差从约
   $0.218\ {\rm rad}$ 降到 $0.01154\ {\rm rad}$，改善 18.9 倍。

4. **无约束 LS 没有造成坐标折叠。**
   最终模型在独立验证点上的

   $$
   \min\left(1+\partial_\theta\lambda\right)=0.382>0,
   $$

   未发现不可逆点。因此当前没有证据说明必须改用带不等式约束的凸优化。

5. **用直场线角重新参数化后，R/Z 可以继续用线性谱投影稳定得到。**
   测试的 DESC `LMN=6,8,10,12` 四档体几何全部嵌套，$\sqrt g$ 全域单符号。

### 1.2 还没有解决的部分

1. 最终 $\alpha$ 对线圈磁场的独立验证相对残差仍为 $5.79\%$，远高于由
   $\boldsymbol B\cdot\nabla\psi\neq0$ 导致的理论法向下限
   $3.52\times10^{-5}$。剩余误差主要集中在 $\rho<0.2$ 的近轴区域。

2. 直场线 R/Z 初值虽然嵌套，但尚未足够接近 DESC 的力平衡解。
   `LMN=6` 的初始归一化力均值为 0.754，仍不是小残差平衡。

3. 直接调用 DESC solve 仍然物理发散。优化器报告 `xtol` 成功，但解变成非嵌套，
   目标和力残差大幅上升。

### 1.3 本轮最重要的新发现

已有 Boozer 曲面文件记录

$$
\iota_{\mathrm{file}}=-0.486970.
$$

但从该曲面上的点出发，直接追踪 32 个场周期得到

$$
\iota_{\mathrm{trace}}=-0.565327\pm0.000261.
$$

稠密 alpha LS 得到

$$
\iota_{\alpha}=-0.565228.
$$

alpha 与独立长场线追踪符合到约 $10^{-4}$，旧 Boozer 文件中的 iota 则与真实
绕转率明显不符。这是旧 Boozer 链路中独立于 DESC 的接口或参数化问题，必须单独
修正。

---

## 2. 实验前存档和运行约束

实验前已将原有 DESC 初值研究完整存档为 Git 提交：

```text
d5e8e5a Archive DESC volume initial guess investigation
```

本轮远端实验固定使用一张 RTX 5090；OpenMP、OpenBLAS、MKL 和 NumExpr 均限制为
最多 16 线程。所有任务均以前台进程运行。结束后检查结果为：无本任务残留进程，
僵尸进程数为 0。

## 3. 准确计算流程

### 3.1 从拟合不变量 s 标定物理磁通

原模型给出无量纲近似不变量 $s(R,Z,\phi)$，其等值面几何已经通过嵌套性和
Poincare 检查，但 $s$ 本身不是物理磁通。

在固定环向截面 $\phi=\phi_j$ 上，对 $s=s_k$ 内部区域积分：

$$
\Phi_T(s_k,\phi_j)
=\int_{D(s_k,\phi_j)}B_\phi(R,Z,\phi_j)\,dR\,dZ.
$$

实验使用 8 个均匀环向截面、256 个极向角和 24 点 Gauss-Legendre 径向积分。
然后定义

$$
\psi_T(s)=\frac{\langle\Phi_T(s,\phi_j)\rangle_j}{2\pi},
$$

并用无常数项四次多项式做小规模线性拟合：

$$
\psi_T(s)=\sum_{k=1}^{4}a_k s^k.
$$

标定结果如下：

- 边界 $\psi_T=2.204339\times10^{-4}\ {\rm Wb/rad}$；
- 边界 $|\Phi_T|=1.385027\times10^{-3}\ {\rm Wb}$；
- 多项式对积分数据的相对 RMS 为 $1.41\times10^{-8}$；
- 与 PCHIP 标定曲线的相对 L2 差为 $1.26\times10^{-6}$；
- $d\psi_T/ds$ 始终同号，标定严格单调。

![环向磁通标定](alpha_clebsch_experiment/flux_calibration.png)

有向磁通的正负取决于曲面和极向角方向。标定积分与 Simsopt 曲面积分的**幅值**
相差 0.265%，DESC 接口处使用 Simsopt 曲面的方向符号。

### 3.2 alpha 的拓扑形式

采用顺时针几何极向角

$$
\theta=-\operatorname{atan2}(Z-Z_a(\phi),R-R_a(\phi)),
$$

使 $\nabla\psi_T\times\nabla\theta$ 的环向方向与线圈磁场一致。Clebsch 势写为

$$
\alpha=\theta+\lambda(\rho,\theta,\phi)-\iota(\rho)\phi,
\qquad
\rho=\sqrt{\frac{\psi_T}{\psi_{T,\mathrm{edge}}}}.
$$

磁场目标为

$$
\boldsymbol B
=\nabla\psi_T\times\nabla\alpha.
$$

周期部分使用实 Fourier-Zernike 展开：

$$
\lambda
=\sum_{lmn}c^{(c)}_{lmn}R_l^m(\rho)
\cos(m\theta-nN_{\rm FP}\phi)
+\sum_{lmn}c^{(s)}_{lmn}R_l^m(\rho)
\sin(m\theta-nN_{\rm FP}\phi).
$$

纯磁通函数 $m=n=0$ 被删除，以固定 gauge。iota 使用

$$
\iota(\rho)=\sum_k b_k\rho^{2k}.
$$

由于 $\rho$ 是 $\psi_T$ 的函数，所有径向导数项与 $\nabla\psi_T$ 平行，
在叉乘中严格消失。因此未知系数只线性地进入

$$
\boldsymbol B
=\boldsymbol C_\theta
+\sum_j c_j
\left(
\partial_\theta f_j\,\boldsymbol C_\theta
+\partial_\phi f_j\,\boldsymbol C_\phi
\right)
-\sum_k b_k\rho^{2k}\boldsymbol C_\phi,
$$

其中

$$
\boldsymbol C_\theta=\nabla\psi_T\times\nabla\theta,
\qquad
\boldsymbol C_\phi=\nabla\psi_T\times\nabla\phi.
$$

这就是本轮 GPU QR 的完整线性方程，没有隐藏的非线性步骤。

### 3.3 点云和验证集

取点方式与 s 拟合保持同一原则：在磁轴附近的物理 $(R-R_a,Z-Z_a,\phi)$ 空间中
建立均匀笛卡尔格点，再筛选

$$
\rho_{\min}\le\rho\le1.
$$

正式参数为：

| 数据集 | 点数 | 生成方式 |
|---|---:|---|
| 训练集 | 120,000 | $144\times144\times96$ 平移均匀格点内筛选 |
| 验证集 | 60,000 | 不同平移量的 $145\times145\times97$ 格点内筛选 |

两者没有共享格点。每个物理点提供 $B_R,B_\phi,B_Z$ 三行，因此最高阶系统有
360,000 行。

### 3.4 GPU QR

主实验使用 FP64 `torch.linalg.lstsq(..., driver="gels")`。只做列范数归一化，不加
ridge。最高阶 $L=12,M=12,N=16$ 含 3001 个未知量，GPU QR 时间约 18.9 s。

## 4. alpha 阶数扫描

| 阶数 | 列数 | 训练相对 L2 | 验证相对 L2 | 验证最小 $1+\lambda_\theta$ | 折叠比例 |
|---|---:|---:|---:|---:|---:|
| $L4,M4,N4$ | 136 | 0.14342 | 0.14380 | 0.302 | 0 |
| $L6,M6,N6$ | 364 | 0.10643 | 0.10671 | 0.238 | 0 |
| $L8,M8,N8$ | 764 | 0.08393 | 0.08407 | 0.365 | 0 |
| $L10,M10,N12$ | 1648 | 0.06874 | 0.06888 | 0.278 | 0 |
| $L12,M12,N16$ | 3001 | 0.05769 | 0.05793 | 0.382 | 0 |

![alpha 阶数扫描](alpha_clebsch_experiment/order_scan.png)

训练和验证误差始终接近，说明没有观察到统计过拟合。残差随阶数单调下降，但到
$L12,M12,N16$ 尚未完全谱收敛。

将环向阶数单独提高到 $N=24$ 并没有改善结果；对应验证误差为 0.0690。因此当前
主误差不是简单的环向带宽不足。

### 4.1 residual 来自哪里

最终模型的分量相对误差为：

| 分量 | 相对 L2 |
|---|---:|
| $B_R$ | 0.0600 |
| $B_\phi$ | 0.0590 |
| $B_Z$ | 0.0478 |

径向分箱结果为：

| rho 区间 | 相对 L2 |
|---|---:|
| 0.0-0.2 | 0.2426 |
| 0.2-0.4 | 0.0787 |
| 0.4-0.6 | 0.0321 |
| 0.6-0.8 | 0.0191 |
| 0.8-1.0 | 0.0208 |

由 $\boldsymbol B\cdot\nabla\psi_T\neq0$ 形成的不可消除法向下限只有
$3.52\times10^{-5}$。所以 5.79% 主 residual **不是 psi 法向误差造成的**，而是
近轴坐标/基底和均匀体积采样对小体积内层的分辨率不足。

### 4.2 为什么最终选择常数 iota

允许四次 $\iota(\rho)$ 时，总磁场 residual 几乎不变，但内层 iota 出现没有
独立场线支持的摆动。原因是均匀体积采样中内层点数按面积减小，而 iota 主要由
较小的极向场分量约束。

改成常数 iota 后得到

$$
\iota=-0.5652282747.
$$

总磁场 residual 不变，但八周期场线直线残差从变 iota 版本的 0.0392 rad 进一步
下降到 0.01154 rad。因此常数版本是当前更可信的初值。

## 5. 磁力线是否真的变直

下图使用真实 Biot-Savart 场重新追踪，并未使用 LS 训练行生成轨迹。左图为原始几何
角，右图为

$$
\vartheta=\theta+\lambda.
$$

黑色虚线是斜率 $\iota=-0.565228$ 的直线。

![磁力线直线化](alpha_clebsch_experiment/fieldline_straightening.png)

十二条线的统计为：

- 原始平均直线 residual：约 0.218 rad；
- 修正后平均直线 residual：0.01154 rad；
- 改善倍数：18.9；
- $\rho=0.3$ 修正后约 0.018-0.020 rad；
- $\rho=0.9$ 修正后约 0.0052-0.0059 rad。

坐标可逆性图如下。在 $\phi=0$ 截面上没有零线或负值区域。

![坐标可逆性](alpha_clebsch_experiment/coordinate_invertibility.png)

## 6. iota 的独立长轨迹核验

从已有 Boozer 边界曲面上的 8 个点出发，追踪 32 个场周期。对未展开的顺时针几何
角做长时间线性回归，周期性角畸变在斜率中被平均掉，得到

$$
\iota_{\mathrm{trace}}=-0.5653270532,
\qquad
\sigma_{\mathrm{lines}}=2.61\times10^{-4}.
$$

对比：

| 来源 | iota |
|---|---:|
| alpha 稠密 LS | -0.565228 |
| 32 周期真实场线 | -0.565327 |
| 旧单周期 screen | -0.598348，线间标准差 0.200836 |
| Boozer 文件 | -0.486970 |

32 周期后，终点到原 Boozer 截面曲线的平均最近距离为 0.224 mm，最大为 0.455 mm。
相对于约 21 mm 的平均小半径，曲面仍在同一磁面邻域，但不是数值上完全无漂移。

结论是：alpha 的 iota 与真实长轨迹一致；旧 Boozer 文件的 iota 不能继续作为真值。

## 7. alpha 之后的 R/Z 线性投影

对每个由 $s$ 等值面提取的物理点，先计算

$$
\vartheta=\theta+\lambda,
$$

再把 $(\rho,\vartheta,\phi;R,Z)$ 送入 DESC Fourier-Zernike 基底。R 和 Z 分别用
GPU QR 线性拟合，DESC 初始 $\lambda$ 直接设为零。

正式 RZ 数据包括 21 个内部径向层，每层 $37\times37$ 个角点，另有独立边界和
磁轴数据。四个分辨率结果为：

| DESC 分辨率 | R RMS [m] | Z RMS [m] | 边界 RMS [m] | 是否嵌套 | 初始力均值 | 初始力 p95 |
|---|---:|---:|---:|---|---:|---:|
| LMN=6 | 1.625e-4 | 1.867e-4 | 2.519e-4 | 是 | 0.754 | 2.115 |
| LMN=8 | 1.519e-4 | 1.762e-4 | 2.154e-4 | 是 | 0.809 | 3.001 |
| LMN=10 | 1.369e-4 | 1.552e-4 | 2.031e-4 | 是 | 0.944 | 2.605 |
| LMN=12 | 1.333e-4 | 1.491e-4 | 1.966e-4 | 是 | 1.034 | 2.839 |

所有档位的 $\sqrt g$ 都保持单符号。最低几何 RMS 在 `LMN=12`，但最低力 residual
在 `LMN=6`。高阶更精确地拟合线圈磁面几何，却放大了 DESC 力导数中的高频误差，
因此 DESC 初值不能只按几何 RMS 选阶。

修复 `eq.save()` 坐标翻转副作用后的截面对比如下。实线是 psi 目标，虚线是
`LMN=6` 的 DESC 谱投影，两者基本重合。

![直场线 RZ 谱投影](alpha_clebsch_experiment/rz_straight_fit_sections.png)

## 8. DESC solve 结果

选择初始力最小的 `LMN=6`，固定同步后的边界，运行最多 50 次迭代的 DESC solve。
优化器在 10 次迭代后因 `xtol` 停止并报告 success，但实际结果为：

| 指标 | 初始 | solve 后 |
|---|---:|---:|
| 是否嵌套 | 是 | 否 |
| 归一化力均值 | 0.754 | $7.85\times10^5$ |
| 归一化力 p95 | 2.115 | 552.0 |
| 归一化力最大值 | 12.59 | $2.94\times10^9$ |
| 优化目标 | 约 $2.49\times10^3$ | $1.00\times10^{14}$ |

因此这次 solve 是明确的物理发散，不能因为 optimizer success 而接受。

## 9. 对原设想的最终判断

原设想可以拆成三句话：

1. 标定物理 psi；
2. 用十万级均匀点和一次线性 LS 得到 alpha、iota、lambda；
3. 之后稳定得到 DESC 需要的内容。

本轮结论是：

- 第 1 步已经走通；
- 第 2 步已经走通，而且普通无约束 QR 当前比凸约束更合适；
- 第 3 步中的坐标和 R/Z 谱投影已经走通；
- 第 3 步中的最终 DESC 力平衡还没有走通。

这不是 alpha 路线失败。相反，alpha 已经解决了此前最病态的“短场线点云同时拟合
每条线截距、lambda 和 iota”问题，并暴露了旧 Boozer iota 不正确这一独立问题。

另一方面，alpha 不能凭空消除所有物理和表示误差。对精确的无电流线圈真空场，若
存在精确嵌套磁面，则理论上应存在 $p=0,\boldsymbol J=0$ 的真空平衡。因此当前
DESC 发散不应解释成“该线圈理论上没有平衡”，而应解释为以下近似尚未闭合：

1. Boozer 曲面和 iota 链路存在已确认的不一致；
2. alpha 在近轴区仍有较大表示误差；
3. 边界和 psi 仍是近似磁面，长轨迹存在约 0.2-0.5 mm 漂移；
4. 把线圈场的直场线坐标交给 DESC，并不自动使 DESC 自身由边界、磁通和 profile
   生成的初始磁场等于线圈磁场；
5. 高阶 R/Z 虽然几何更准，但导数误差会恶化初始力。

## 10. 下一步优先级

### 优先级 1：修复和审计旧 Boozer iota

这是本轮发现的确定性矛盾。应直接检查 BoozerSurface 的角变量尺度、NFP 因子、旧版
API 兼容分支和 iota 自由变量写回过程。修复后用 32 周期追踪作为强制回归测试。

难度：中等。问题已被稳定复现，验证标准清楚。

### 优先级 2：保持线性 QR，专门加强近轴 alpha

不应先上凸优化，因为当前没有折叠。更合适的线性改进是：

1. 对 $\rho<0.2$ 分层补点或增加径向权重；
2. 保持总点数十万级，但避免均匀体积采样让内层占比过低；
3. 使用更严格的轴正则 Fourier-Zernike 组合；
4. 暂时固定长轨迹确认的常数 iota，只拟合 lambda；
5. 分别监控三分量和各 rho 分箱 residual。

这些改动仍然是单次线性 GPU QR。

难度：低到中等。

### 优先级 3：在进入 solve 前比较 DESC B 与线圈 B

目前只比较了 R/Z 和 DESC force。下一步应在同一物理点上明确计算：

$$
\boldsymbol B_{\mathrm{coil}},
\qquad
\boldsymbol B_{\alpha},
\qquad
\boldsymbol B_{\mathrm{DESC,initial}}.
$$

逐层区分误差到底在 alpha、R/Z 投影，还是 DESC 根据 profile 重建磁场的接口处。
若 $\boldsymbol B_{\alpha}$ 好而 $\boldsymbol B_{\mathrm{DESC,initial}}$ 差，则应调整
DESC 的 iota/current formulation 或初始 profile，而不是继续堆 R/Z 阶数。

难度：中等，需要仔细核对 DESC 0.16.0 的磁场输出和坐标约定。

### 优先级 4：最后才考虑约束或 continuation

只有当近轴加权导致

$$
1+\partial_\theta\lambda\le0
$$

时，才有必要引入线性不等式约束的凸二次规划。DESC solve 则应采用低阶、限步长、
拒绝目标上升的 continuation，而不能再次接受 `xtol success` 作为物理成功。

## 11. 代码和原始结果

主要新增实现：

- `stellarator_eval/alpha_clebsch.py`
- `scripts/alpha_clebsch_ls_experiment.py`
- `scripts/desc_alpha_rz_projection_experiment.py`
- `scripts/diagnose_boozer_iota_fieldline.py`
- `tests/test_alpha_clebsch.py`

本报告资产目录同时保留：

- `alpha_summary.json`
- `boozer_iota_long_trace.json`
- `rz_projection_summary.json`
- `desc_solve_summary.json`

测试结果：alpha 新测试与相关既有回归测试合计 9 项全部通过。
