# QUASR no_surface 深入诊断

## 目的

本轮在完整流程批量结果基础上，进一步定位 `no_surface` 的原因。重点检查：

1. $\psi$ 拟合本身是否失败；
2. $\psi$ 等值线是否在磁轴附近是合理闭合面；
3. 半径 $a$ 范围内，一周期追踪是否出现明显异常；
4. 已经通过 screen 的样本为什么在 Boozer LS/Newton 失败。

诊断输出：

- `runs/new_axis_surface_failure_diagnostics/`
- 闭合残差总览图：`runs/new_axis_surface_failure_diagnostics/closure_contact_sheet.png`
- $\sqrt{\psi}$ 截面总览图：`runs/new_axis_surface_failure_diagnostics/psi_contact_sheet.png`

## 样本选择

代表样本：

| 类别 | IDs |
| --- | --- |
| QA 成功对照 | `10302` |
| QA angle 差 | `20060, 27084, 47466` |
| QA angle 不差但 drift 巨大 | `47722, 48699, 51755` |
| QA screen 通过但 Boozer 失败 | `42816, 50137, 60784` |
| QH 成功对照 | `1118987` |
| QH angle 差 | `1407754, 1574935` |
| QH drift 边缘失败 | `1330574` |
| QH no_axis | `1301688` |
| QH screen 通过但 Boozer 失败 | `1570875` |
| QH angle 不差但 drift 巨大 | `1748677` |

## 一个重要修正

“半径 $a$ 范围内密集点追踪一周期后回到同一个 $(R,Z)$”这个量不能被简单理解为磁面存在的必要条件。

原因是：非轴磁面上磁力线绕一个场周期后通常会推进一个 poloidal 相位，除非正好是特殊有理转角，否则不应该回到同一个截面点。因此，一周期点返回残差在普通好磁面上也可以是 $10^{-2}$ 到 $10^{-1}$ 量级。

这个热力图仍然有用，但主要用于看：

- 是否存在清楚、连续的低残差轴邻域；
- 是否有大面积逃逸或斑点状异常；
- 失败样本与成功样本相比是否更混沌或更病态。

## 一周期闭合残差对照

| ID | 状态 | axis residual | closure p50 | closure p95 | closure max | 半径内 residual < $10^{-3}$ 比例 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10302 | surface | $3.55\times 10^{-9}$ | 0.0185 | 0.0431 | 0.0553 | 0.00339 |
| 1118987 | surface | $5.20\times 10^{-9}$ | 0.0629 | 0.0947 | 0.1159 | 0 |
| 20060 | no_surface | $6.58\times 10^{-9}$ | 0.0375 | 0.0789 | $2.56\times 10^4$ | 0.000282 |
| 47466 | no_surface | $8.85\times 10^{-9}$ | 0.231 | 2.735 | $2.67\times 10^4$ | 0.000282 |
| 47722 | no_surface | $8.12\times 10^{-9}$ | 0.0917 | 0.219 | 0.287 | 0.000282 |
| 42816 | no_surface | $6.52\times 10^{-9}$ | 0.0254 | 0.0527 | 0.0610 | 0.000847 |
| 1407754 | no_surface | $2.37\times 10^{-8}$ | 0.192 | 0.504 | $4.72\times 10^4$ | 0 |
| 1330574 | no_surface | $5.14\times 10^{-9}$ | 0.0925 | 0.197 | 0.246 | 0 |
| 1301688 | no_axis | $5.94\times 10^{-2}$ | 1.40 | $4.78\times 10^3$ | $2.56\times 10^4$ | 0 |
| 1570875 | no_surface | $4.74\times 10^{-9}$ | 0.0533 | 0.101 | 0.116 | 0.000282 |
| 1748677 | no_surface | $2.49\times 10^{-8}$ | 0.0708 | 0.107 | 0.130 | 0 |

结论：

- 成功样本的 closure p95 也并不小，因此不能要求整个 $a$ 范围都一周期闭合。
- `1301688` 的残差图是明确异常：无轴判断可信。
- `1407754`、`20060`、`47466` 等有斑点状巨大残差/逃逸，说明轴附近或半径 $a$ 范围内确实存在不稳定区域。
- `42816/50137/60784/1570875` 的 closure 图并不比成功样本差，说明它们不是磁轴或邻域混沌导致的失败。

## $\psi$ 拓扑与 Hessian 分类

对磁轴附近的 $\psi$ 二次型做 Hessian 分类：

| ID | 状态 | angle p95 | Hessian 分类 | 解释 |
| --- | --- | ---: | --- | --- |
| 10302 | surface | $1.42\times 10^{-5}$ | elliptic | 成功对照 |
| 1118987 | surface | $6.20\times 10^{-5}$ | elliptic | 成功对照 |
| 20060 | no_surface | $5.80\times 10^{-2}$ | saddle | $\psi$ 明显不适合当磁面函数 |
| 27084 | no_surface | $1.30\times 10^{-3}$ | elliptic | 拟合 angle 偏差，screen 边缘失败 |
| 47466 | no_surface | $3.18\times 10^{-3}$ | saddle | $\psi$ 拓扑错误，且闭合残差异常 |
| 47722 | no_surface | $4.69\times 10^{-5}$ | saddle | angle 小但等值线是 X 形开口，angle 指标误导 |
| 48699 | no_surface | $2.28\times 10^{-5}$ | saddle | 同上 |
| 51755 | no_surface | $5.03\times 10^{-6}$ | saddle | 同上 |
| 42816 | no_surface | $3.35\times 10^{-5}$ | elliptic | $\psi$ 和 screen 都较好，失败在 Boozer 后续 |
| 50137 | no_surface | $4.38\times 10^{-5}$ | elliptic | 同上 |
| 60784 | no_surface | $2.34\times 10^{-5}$ | elliptic | 同上 |
| 1407754 | no_surface | $1.67\times 10^{-2}$ | elliptic | 二次型闭合但 angle 差，拟合质量不足 |
| 1574935 | no_surface | $1.47\times 10^{-2}$ | saddle | 拟合和拓扑都差 |
| 1330574 | no_surface | $4.56\times 10^{-4}$ | elliptic | screen 边缘失败，可能半径/level 太激进 |
| 1570875 | no_surface | $1.00\times 10^{-3}$ | elliptic | screen 通过后 Boozer 失败 |
| 1748677 | no_surface | $4.18\times 10^{-4}$ | saddle | 拓扑近退化/开口，screen 失败合理 |

最关键的发现是：`47722/48699/51755` 这类 QA 样本虽然 angle p95 很小，但 $\psi$ 在轴附近是 saddle。也就是说，$\nabla\psi\cdot B$ 局部小并不保证 $\psi$ 是可用的磁面函数；还必须要求磁轴附近的二次型是正定/椭圆型。

## 分类结论

### 1. 真的不该期待有好 $\psi$ 的样本

代表：

- `20060`
- `47466`
- `47722`
- `48699`
- `51755`
- `1574935`
- `1748677`

原因：

- $\psi$ Hessian 是 saddle，或截面图出现 X 形/open-channel。
- 这类样本即使 angle 残差小，也不能说明存在闭合磁面。
- 问题很可能在 $\psi$ 表示形式、训练目标、选点权重，或评估半径 $a$ 太大导致拟合把非椭圆结构纳入。

建议：

- 在 $\psi$ 拟合后增加“轴附近 Hessian 正定性检查”。
- 若 Hessian 不是 elliptic，直接判定该 $\psi$ 不适合进入 surface screen/Boozer。
- 对 saddle 样本尝试减小 `a`，看 Hessian 是否恢复 elliptic。

### 2. 拟合质量不足但拓扑仍可能正确的样本

代表：

- `27084`
- `1407754`
- `1330574`

特征：

- Hessian 是 elliptic；
- 但 angle p95 或 screen distance 超阈值；
- `1407754` 还有明显闭合残差异常区域。

建议：

- 提高 $\psi$ 模数或训练点密度；
- 减小 `a`；
- 改进训练点权重，让靠近磁轴/候选 level 的区域优先；
- 对 `1407754` 这类有逃逸斑点的样本，需要先看半径内是否已经混沌，不能只靠提高模数。

### 3. $\psi$ 和 screen 都较好，但 Boozer 后续失败

代表：

- `42816`
- `50137`
- `60784`
- `1570875`

特征：

- Hessian elliptic；
- screen 有多个 level 通过；
- closure 图不比成功样本差；
- 失败发生在 Boozer LS/Newton。

这类样本最值得单独继续查，因为它们说明前面的 $\psi$ 面可能已经够好，瓶颈在后续参数化/约束/初值。

建议：

- 对这些样本输出 Boozer residual 分解；
- 比较初始面与 LS/Newton 后的几何变化；
- 尝试降低 surface order、调整 initial iota/G、或使用 fieldline-conjugacy/G 重参数化初值；
- 检查 volume 约束是否把面拉到错误分支。

### 4. 无轴或明显混沌邻域

代表：

- `1301688`

特征：

- axis residual 为 $5.94\times 10^{-2}$；
- 半径 $a$ 范围内 closure p95 达到 $4.78\times 10^3$；
- 图像呈斑点状巨大残差。

结论：

- 这个样本当前可视为真 `no_axis` 或至少“当前搜索范围内无可信磁轴”。
- 应单独做更大范围固定点向量场搜索，而不是进入 $\psi$。

## 对后续流程的建议

建议在主流程中增加三道 cheap 检查：

1. 轴附近 Hessian 正定性检查  
   在 $\psi$ fit 后，对多个 $\Phi$ 截面检查 $\psi$ 的二维 Hessian。若出现 saddle，直接记录为 `psi_saddle_near_axis`，不要继续 screen。

2. screen 失败分类  
   对 `screen_ok_count=0` 的样本，区分：
   - angle 大：`psi_fit_bad`
   - Hessian saddle：`psi_topology_bad`
   - angle 小且 Hessian elliptic 但 drift 大：`fieldline_drift_bad`

3. Boozer 后续失败专项  
   对 screen 通过但 Boozer 失败的样本保留更多副产品：
   - 初始 residual 分解；
   - LS 后 residual；
   - Newton 后 residual；
   - 初始/最终面到 $\psi=\psi_0$ 的距离；
   - iota/G/volume 变化。

当前最有价值的下一步不是盲目扩大批量，而是先处理 `42816/50137/60784/1570875` 这一组 Boozer 后续失败样本，因为它们前面的 $\psi$ 和 screen 都相对健康，最可能暴露后端算法问题。
