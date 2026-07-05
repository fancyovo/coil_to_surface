# QUASR QH no-axis 密集热力图诊断

原始产物：
- [runs/quasr_qh_noaxis_dense_remote/report.md](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/report.md)
- [runs/quasr_qh_noaxis_dense_remote/dense_axis_summary.json](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/dense_axis_summary.json)

本次做法：
- 对两批 QH 失败样本中的 `11` 个 `no_axis` 样本
- 在搜索框内做一次 `128 x 128` 的一周期闭合残差热力图
- 每张图上同时叠加
  - GA 每代最优点轨迹
  - GA 最终点
  - dense-grid 最优点
- 残差定义与原 axis 搜索一致：一周期后起点和终点的闭合距离

## 1. 总结结论

这 `11` 个样本并不是一种失效模式，而是至少分成三类：

1. **GA 的确跑偏了，但即便更好的盆地也离真磁轴很远**
   - `1265101`
   - `2116461`
   - `2186800`

2. **GA 已经抓到了一个极窄的低残差尖谷，128x128 全局网格分辨不出来**
   - `1569052`
   - `1663361`
   - `1673322`
   - `1738668`

3. **存在低残差谷，但仍没有看到真正趋近于 0 的区域**
   - `1409259`
   - `1836024`
   - `1886709`
   - `2148019`

也就是说：
- 这批样本里，**不能简单说“都没有磁轴”**
- 但也**不能简单说“GA 都跑偏了”**
- 更准确地说，QH 的 `no_axis` 至少混合了：
  - 真的坏样本
  - 边界样本
  - 以及极窄低谷导致的分辨率问题

## 2. 明确跑偏的样本

### `1265101`

- GA 最终：`8.01e-03`
- dense-grid 最优：`2.79e-03`
- 两者距离：`0.527 m`

热力图：[ID 1265101](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/id_1265101_axis_residual_heatmap.png)

这里有明显更好的低残差竖向谷，但 GA 最终停在了下方另一个次优区域。  
不过更好的谷底也只有 `1e-3` 量级，仍然离 `1e-7` 很远，所以结论是：

- **GA 跑偏了**
- **但即便纠正，也看不到真磁轴**

### `2116461`

- GA 最终：`5.72`
- dense-grid 最优：`7.79e-02`
- 两者距离：`0.291 m`

热力图：[ID 2116461](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/id_2116461_axis_residual_heatmap.png)

这是最明显的坏例子之一。GA 最终停在一个很差的位置，而 dense 图上另一个区域明显更好。  
但最好的地方也只有 `1e-1` 到 `1e-2` 量级，远不是磁轴。

### `2186800`

- GA 最终：`2.80e-01`
- dense-grid 最优：`3.03e-03`
- 两者距离：`0.230 m`

热力图：[ID 2186800](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/id_2186800_axis_residual_heatmap.png)

这例也很清楚：GA 停在了左下方的错误盆地。  
dense 图在上方还有明显更深的低谷。

但注意：
- 更好的区域虽然比 GA 好很多
- 仍然只是 `1e-3`，离 `1e-7` 差四个数量级

所以它依然不是“其实有磁轴只是没找到”，而是“GA 确实跑偏，但全局看仍不像有磁轴”。

## 3. 极窄低谷 / 边界样本

### `1569052`

- GA 最终：`1.05e-07`
- dense-grid 最优：`4.17e-03`
- 两者距离：`4.08e-03 m`

热力图：[ID 1569052](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/id_1569052_axis_residual_heatmap.png)

GA 最终点和 dense-grid 最优格点几乎重合，但全局 128x128 网格完全分辨不出那个尖底。  
这说明这里更像是：

- **GA 已经抓住了一个非常窄的低残差尖谷**
- **不是 GA 跑到错误区域**

换句话说，这类样本的 `no_axis` 更像“阈值边界 + 分辨率问题”，而不是“明显无轴”。

### `1663361`

- GA 最终：`1.04e-07`
- dense-grid 最优：`8.09e-03`

热力图：[ID 1663361](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/id_1663361_axis_residual_heatmap.png)

和 `1569052` 的模式类似：
- GA 最终点就在左边界附近的尖底上
- dense-grid 只能看到一个很窄的低谷区域，但分辨不出真正的最深点

这类样本不能拿 128x128 热力图去否定 GA 找到的近轴点。

### `1673322`

- GA 最终：`1.00e-07`
- dense-grid 最优：`7.14e-03`

它和 `1569052 / 1663361` 同类，也是近阈值边界样本。  
从数值上看，依然更像“极窄低谷没有被 coarse grid resolve 掉”。

### `1738668`

- GA 最终：`1.52e-07`
- dense-grid 最优：`4.29e-04`

热力图：[ID 1738668](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/id_1738668_axis_residual_heatmap.png)

这一例尤其值得注意，因为之前延长 GA 代数后它已经能从 `1.52e-07` 降到 `7.47e-08`。  
这和热力图完全一致：

- 存在一个很窄的深低谷
- 全局 128x128 只能看到谷的大致位置，分辨不到最深点

所以 `1738668` 基本可以确认：
- **不是“真的没有轴”**
- **而是 axis search 在边界精度附近还没完全收敛**

## 4. 低残差谷存在，但不像真磁轴

### `1409259`

- GA 最终：`6.24e-04`
- dense-grid 最优：`3.55e-03`

热力图：[ID 1409259](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/id_1409259_axis_residual_heatmap.png)

GA 实际上已经找到了一条相对较深的低残差谷，而且比 dense-grid 的最优格点还更好。  
但问题在于：

- 最好的量级也只是 `1e-4` 到 `1e-3`
- 仍然和 `1e-7` 差得很远

所以这更像是：
- **有比较深的低谷**
- **但谷底并没有真正逼近磁轴**

### `1836024`

- GA 最终：`4.75e-03`
- dense-grid 最优：`5.05e-03`

这例很直接：GA 和 dense-grid 基本一个量级，说明没有漏掉什么极深盆地。  
结论就是当前搜索框里看不到真轴。

### `1886709`

- GA 最终：`1.37e-07`
- dense-grid 最优：`3.38e-03`

热力图：[ID 1886709](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/id_1886709_axis_residual_heatmap.png)

这一例介于第 2 类和第 3 类之间：
- GA 最终点和 dense-grid 最优点都落在同一条右侧低谷上
- 但两者沿谷方向分开了一段距离

所以更像是：
- **这条低谷是真实存在的**
- **GA 沿谷推进到了更深的位置**
- 但 128x128 还不够分辨谷底的最深尖部

它比 `1569052` 那类更不干净，但也不像纯粹跑偏。

### `2148019`

- GA 最终：`2.07e-02`
- dense-grid 最优：`1.98e-02`

热力图：[ID 2148019](D:/Typora/Typ/学习/stellarator/programs/local_surface_evaluator/runs/quasr_qh_noaxis_dense_remote/id_2148019_axis_residual_heatmap.png)

这里最有意思的是：
- GA 和 dense-grid 给出的最好残差几乎一样
- 但位置差得很远

这说明它更像是一个**宽而平的低残差地形**，不是一个孤立尖锐的轴极小值。  
因此虽然“更好区域”很多，但都不像真轴。

## 5. 对“到底有没有磁轴”的判断

综合这次 128x128 热力图，可以把这 11 个样本分成两批：

### 基本支持“当前框里没有真轴”的

- `1265101`
- `1409259`
- `1836024`
- `2116461`
- `2148019`
- `2186800`

这些样本要么：
- 更好的盆地仍远离阈值
- 要么整个低谷都停留在 `1e-2 ~ 1e-3` 量级

### 不能据此否定轴存在的边界样本

- `1569052`
- `1663361`
- `1673322`
- `1738668`
- `1886709`

这些样本的共同点是：
- GA 最终残差已经到 `1e-7` 左右
- 热力图显示确实有一条很窄的低残差谷
- 128x128 的全局 coarse grid 本身不够分辨那条谷底

因此这几例更像：
- **存在非常窄的近轴极小值**
- **GA 已经基本抓到，只差最终验证或局部细化**

## 6. 结论

这次密集热力图给出的信息是明确的：

1. QH 的 `no_axis` 样本不是统一机理
2. 确实有几例是 GA 跑偏了
3. 但也有一批样本，GA 并没有跑偏，而是已经抓住了极窄低谷
4. 对这批边界样本，128x128 全局图只能说明“存在尖谷”，不能说明“真的没有磁轴”

如果下一步只挑最值得继续救的 `no_axis` 样本，我建议优先：

1. `1738668`
   - 已知增加 GA 代数后能过阈值
2. `1569052`
3. `1663361`
4. `1673322`
5. `1886709`

这五个样本最不像“真的无轴”，而更像“极窄低谷 + 验证边界”的问题。
