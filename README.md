# Local Surface Evaluator

从 Fourier 线圈参数出发，快速评估局部磁面、旋转变换和体准对称性，并为高价值样本执行完整的 Boozer 磁面与 DESC 物理验收。

当前项目有两条边界清晰、互不替代的正式路径：

| 路径 | 用途 | 数值方法 | 典型成本 |
|---|---|---|---:|
| ABI-9 原生 score | 大批量筛选和优化 | C++/CUDA 定长追踪、FP32 QR、固定规模归约 | 空闲 RTX 5090 上代表性高分样本约 5--8 s |
| 完整物理评估 | 验收单个高分样本 | 样本自适应 $\psi$、GPU FP32 $\alpha+\nu$、Simsopt LS/Newton、DESC | 通常数分钟，DESC 可使用 CPU |

原生链路依次完成批量磁轴追踪、局部不变量 $s$、物理磁通 $\psi(s)$、$\alpha$ 与 $\iota$ 联合拟合和体 QA/QH/QP 统计。训练后的 flow matching 用作 Fourier 参数空间的可逆预条件器，不被当作高质量生成保证。正式评分与完整评估中的批量前端使用 GPU，并对失败返回结构化状态；只有明确允许的 DESC 等步骤可以使用 CPU。

快速 score 是有物理含义的排序代理，但不是平衡存在性的证明。正式结论必须来自完整评估支线。

## 计算流程

共享前端和两条后端可以概括为

$$
\text{coils}
\rightarrow \boldsymbol B
\rightarrow \text{magnetic axis}
\rightarrow s
\rightarrow \psi(s)
\rightarrow (\alpha,\iota)
\begin{cases}
\rightarrow \text{volume QS}\rightarrow \text{score},\\
\rightarrow \nu\rightarrow \text{Simsopt LS/Newton}\rightarrow \text{DESC}.
\end{cases}
$$

核心步骤如下：

1. **磁轴**：在一个场周期的 Poincare 映射上批量搜索椭圆固定点，再作有界 Newton 精修和轴追踪。
2. **局部磁面标签 $s$**：在磁轴附近用完整的二维多项式和环向 Fourier 基底拟合 $\boldsymbol B\cdot\nabla s=0$。
3. **物理磁通 $\psi$**：通过多截面环向磁通积分标定 $\psi(s)=\Phi_t(s)/(2\pi)$。$s$ 是几何标签，不是物理磁通。
4. **$\alpha$ 与 $\iota$**：以 Zernike--Fourier 基底对 $\boldsymbol B\cdot\nabla\alpha=0$ 做一次联合线性最小二乘，直接得到全体积直线场线角修正和旋转变换。
5. **快速 score 分支**：不求 $\nu$，直接用 $\psi$、$\iota$、$\boldsymbol B$ 和 $\nabla\boldsymbol B$ 计算体微分 QA/QH/QP。
6. **完整评估分支**：求环向修正 $\nu$，构造近 Boozer 面，再交给标准 Simsopt LS/Newton 和 DESC 独立验收。

评分主线的大规模数值步骤都在 C++/CUDA 中完成。Python 只负责输入输出和作业编排，不会把可批量并行步骤静默回退到慢一至两个数量级的 CPU 实现。

## 当前状态

当前正式评分接口为 **ABI 9**。它包含以下已验证修正：

- 真空协变电流函数使用 $G=\mu_0 I_{\rm link}/(2\pi)$，且电流整体符号约定已经统一。
- 体 QS 使用柱坐标物理体积权重；达到预算后固定压紧为 100000 个点，不能通过减少有效点数刷低误差。
- 椭圆磁轴存在性使用严格的 $|\operatorname{tr}J|/\sqrt{\det J}<2$，拓扑 margin 只参与连续质量评分。
- QH score 对 $\iota\simeq0$、过小磁面和错误 helicity 优势施加显式门控，磁面尺寸达到有效逆纵横比 0.03 后饱和。

独立的 1024 个 QUASR QH 样本与 1024 个同条件随机 flow 样本上，QUASR 的平均分为 48.019、中位数为 75.520，随机样本分别为 24.087 和 0.372；`score >= 80` 的样本数为 443 对 17。该结果证明 score 有明确区分梯度，但仍需对高分样本做完整物理评估。

flow matching 当前不被视为“一次采样即可生成优质线圈”的生成器。它的主要作用是可逆地重参数化 Fourier 参数空间：修正后的 landscape 实验中，潜空间相对随机原空间方向的下降 5 分盆地宽度中位数放大 8.63 倍，且 FP32 RK4-256 的反向--正向线圈位置闭环 RMS 为 $2.26\times10^{-8}$--$4.57\times10^{-8}\,\mathrm m$。

当前标准优化器默认使用连续磁面 score、严格磁轴 continuation、FP32 RK4-128 流水线、两个正交中心差分方向和 $\beta_1=0.7$ 的 Adam，并启用无效端点整步跳过、中心回退和跨步 median/MAD 脏梯度保护。代表性结果包括：

| 条件 | 起点 | 最佳 ABI-9 score | 独立物理验收 |
|---|---:|---:|---|
| $N_{\rm FP}=4$，3 个基本线圈 | 85.502 | 92.383 | 选中 $s=0.49$；体积 $0.06586\,\mathrm m^3$；面 QH error $5.88\times10^{-6}$；DESC 保持嵌套 |
| $N_{\rm FP}=6$，2 个基本线圈 | 74.436 | 86.641 | 选中 $s=0.49$；体积 $0.07061\,\mathrm m^3$；面 QH error $1.88\times10^{-4}$；DESC 最终力残差均值 $6.42\times10^{-4}$ |

第二个实验在第 491 步后逐渐锁在 `no_surface` 可行性边界；从第 400 步到第 700 步的 300 步续跑中，只有 110 个更新被接受。其快速分数虽高于第 400 步，独立磁面体积和面 QH 并未更好。因此当前优化仍受离散可行性分支限制，不能只按快速分数宣称物理改进。

详细方法、实验和图见 [QH 原生评分与潜空间优化](docs/QH原生评分与潜空间优化方法.md) 与 [小条件潜空间 Adam 报告](reports/qh_small_condition_adam_report.md)。

## 输入格式

每个基本线圈由 100 个参数表示：$x/y/z$ 各 33 个实 Fourier 系数，再加一个电流。JSON 输入采用以下结构：

```json
{
  "raw": {
    "x": [[0.0, 0.0]],
    "y": [[0.0, 0.0]],
    "z": [[0.0, 0.0]],
    "current": [100000.0],
    "current_unit": "A",
    "metadata": {"helicity": 1}
  },
  "nfp": 4
}
```

实际 `x/y/z` 每行必须等长，正式模型使用 33 项。`current_unit` 支持 `A` 和 `MA`。基本线圈之外的线圈由恒星器对称和场周期旋转生成。

对于 `scripts/smoke_native_score.py` 和 `scripts/batch_native_score.py`，`helicity=0` 对应 QA 目标 $(M,N)=(1,0)$，非零值对应 QH 目标 $(1,N_{\rm FP})$。若 case 内没有 metadata，必须显式传入 metadata 文件，避免目标对称性默认为 QA。

## 安装与构建

基础 Python 包需要 Python 3.10 或更高版本：

```bash
python -m pip install -e .
```

按用途安装可选依赖：

```bash
python -m pip install -e '.[plot]'
python -m pip install -e '.[train]'
python -m pip install -e '.[simsopt]'
```

完整 Boozer/QS 评估需要 `simsopt`。正式评分和完整评估中的可批量并行前端禁止静默回退到慢速 CPU 实现；CPU 路径只保留给显式选择的历史对照、后处理和允许使用 CPU 的 DESC。

ABI-9 score 需要 CUDA、CMake、cuBLAS 和 cuSOLVER。RTX 5090 的参考构建命令为：

```bash
cmake -S gpu_backend -B gpu_backend/build_native_score \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build gpu_backend/build_native_score --parallel
```

输出库为 `gpu_backend/build_native_score/libstellarator_gpu.so`。集群上可直接提交固定构建脚本：

在 RTX 5090 上，当前 ABI-9 代表样本总墙钟为 7.236 秒；主要时间在磁轴和定长磁面追踪，具体数值会随线圈条件、CUDA 版本和候选面状态变化。

```bash
mkdir -p logs
sbatch scripts/slurm_build_native_score.sh
```

正式实验应记录代码 commit、score 库 SHA-256、flow checkpoint SHA-256 和 GPU 空闲 pre/postflight；不同 ABI 或不同库哈希的分数不能直接比较。

## 快速评分

### 单个样本

以下入口把原生 C++/CUDA 评分器作为黑箱调用，并输出 `score`、`status`、七个分量、计时和诊断量：

```bash
python scripts/smoke_native_score.py path/to/case.json \
  --metadata path/to/metadata.json \
  --lib gpu_backend/build_native_score/libstellarator_gpu.so \
  --device 0 \
  --output runs/native_score/case.json
```

也可以直接调用 Python ctypes 包装层：

```python
import sys
sys.path.insert(0, "gpu_backend/python")

from stellarator_gpu import score_coils_native

result = score_coils_native(
    "gpu_backend/build_native_score/libstellarator_gpu.so",
    coeffs_x,
    coeffs_y,
    coeffs_z,
    currents_a,
    nfp,
    device_id=0,
    target_helicity=(1, nfp),
)
print(result["score"], result["status"], result["components"])
```

### 批量样本

```bash
python scripts/batch_native_score.py \
  --case-dir /path/to/cases \
  --metadata /path/to/metadata.json \
  --split validation \
  --lib gpu_backend/build_native_score/libstellarator_gpu.so \
  --device 0 \
  --warmup \
  --output runs/native_score/worker0.jsonl
```

多卡时为每张卡启动一个 worker，并设置相同的 `--worker-count` 和不同的 `--worker-index`。测速前必须确认分配到的 GPU 空闲；结束后必须确认没有残留计算进程。

原生输出的主要字段为：

- `score`：0--100，越高表示越可能是实用的高质量线圈。
- `status`：`ok`、`no_axis`、`no_surface`、`drift_rejected`、`flux_rejected`、`alpha_failed` 或 `internal_error`。
- `components`：`axis`、`psi`、`surface`、`coordinate`、`volume_qs`、`iota`、`coil`。
- `diagnostics`：$\iota$、QA/QH/QP 误差、选中磁面层、有效点数和数值质量指标。
- `timing`：分阶段耗时；部分 surface/flux 字段是嵌套计时，不能直接逐项相加。

七个分量的权重依次为 $(10,10,10,10,42,10,8)$。总分还会乘 QH 的 $\iota$ gate 和 helicity-advantage gate，因此不能只对分量做线性加权来复算 QH 总分。

## 完整物理评估

正式入口固定在 [`evaluation/full_physical/`](evaluation/full_physical/README.md)，不要在验收时临时拼接脚本。每个样本都必须重新选择自己的 $a$ 和最大已测可行 $s$，不能复用历史样本的数值。

### 1. 样本自适应 source $\psi$

```bash
export PROJECT=$HOME/local_surface_evaluator_worktrees/<branch>
export GPU_LIB=$HOME/local_surface_evaluator/gpu_backend/build_mixed/libstellarator_gpu.so
export EVAL_ENV=$HOME/local_surface_evaluator/.venv-desc016-py312
export CASE_FILE=$PROJECT/runs/<optimizer>/best.json
export OUTPUT_ROOT=$PROJECT/runs/<evaluation_name>
export A_VALUES=0.04,0.05,0.06,0.08
bash evaluation/full_physical/submit_source_psi_candidates.sh
```

根据验证误差、物理覆盖半径和更外侧失败点选择本样本的 `RUN_DIR`。

### 2. 并行搜索标准磁面

```bash
export RUN_DIR=$OUTPUT_ROOT/source_psi_candidates/a_<selected>
export S_EDGES=0.12,0.20,0.30,0.36,0.49
bash evaluation/full_physical/submit_surface_candidates.sh
```

默认每个候选使用 1 张 GPU 和 4 核 CPU，可在四卡上并行四个候选。`SERIAL_CANDIDATES=1` 只用于明确的资源限制，不是保守默认值。候选先走 GPU FP32 $\alpha+\nu$，最终是否存在磁面只由标准 LS/Newton、独立密网格残差和嵌套分支连续性决定。

### 3. 选择最大已测通过面并运行下游

```bash
python evaluation/full_physical/select_largest_standard_surface.py \
  --candidate-root "$OUTPUT_ROOT/candidates" \
  --output "$OUTPUT_ROOT/selection.json"

export DESC_BACKEND=cpu-p107
bash evaluation/full_physical/submit_downstream.sh
```

`DESC_BACKEND` 必须显式选择 `cpu`、`cpu-p107` 或已验证可用的 `gpu`，禁止静默回退。DESC 允许使用 CPU；磁轴、$\psi$、$\alpha+\nu$ 和可批量并行的完整评估前端不得未经允许回退到旧 CPU 路径。

完整交付至少包含：

- 最大已测通过面的选择依据、score 七个分量和面 QS error。
- Poincare 图以及各条线在各截面的命中统计。
- 白底彩色 $|B|$ 等高线图；颜色表示 $|B|$，不使用热力图或填色等高线。
- 完整线圈和大磁面的三维 PNG/HTML。
- DESC 初始/最终残差、嵌套性、优化器退出原因，以及 DESC 生成的全部图。

报告完成后运行交付校验：

```bash
export REPORT=reports/<report>.md
export DESC_DIR=reports/assets/<case>/desc
bash evaluation/full_physical/validate_delivery.sh
```

完整约束、选择规则和故障处理见 [完整评估固定流程](docs/精简线圈评估流程.md)。

## Flow Matching 与潜空间优化

模型把一根 100 维基本线圈视为一个 token，主体采用无因果注意力、无 RoPE 的 Llama 风格 Transformer：PreNorm、RMSNorm、multi-head attention 和 SwiGLU。$N_{\rm FP}$ 通过条件嵌入注入；训练数据覆盖 QUASR QH 的 $N_{\rm FP}=2\ldots8$ 和 1--5 个基本线圈。

逐维标准化仅用于改善输入分布。训练损失按 Fourier Parseval 权重和原始尺度方差恢复曲线积分 $L^2$ 的物理意义，避免高频尾项因标准化而获得不合理权重。四卡训练入口为：

```bash
export QH_FLOW_REPO=$HOME/local_surface_evaluator
export QH_FLOW_DATA=$HOME/local_surface_evaluator_data/quasr_qh_flow_v1
export QH_FLOW_OUTPUT=$QH_FLOW_REPO/runs/qh_flow_<name>
sbatch scripts/slurm_train_qh_flow.sh
```

标准单起点优化入口如下；除输出目录外，下面列出的优化参数都已有生产默认值：

```bash
export PROJECT=$HOME/local_surface_evaluator_worktrees/<branch>
export RUN_ROOT=$PROJECT/runs/qh_flow_standard_adam/<name>
export ITERATIONS=600
sbatch scripts/slurm_flow_prior_standard_adam.sh
```

默认配置为学习率 0.01、扰动 0.005、$(\beta_1,\beta_2)=(0.7,0.999)$、FP32 RK4-128 流水线、2 个正交中心差分方向、连续磁面 score 和严格磁轴 continuation。`--flow-steps`/`FLOW_STEPS`、`--directions`/`DIRECTIONS`、`--beta1`/`BETA1`、`--score-surface-mode`/`SCORE_SURFACE_MODE` 等接口仍可显式覆盖；Python CLI 的布尔默认项可用对应的 `--no-*` 关闭，Slurm 包装使用环境变量 `0` 关闭。

程序保存每一步的完整线圈、潜变量、score 分量和优化器状态。精确续跑必须保持原作业的 flow 步数、方向数和 Adam 参数，指向原 `RUN_ROOT` 并设置 `RESUME=1` 后重新提交 `scripts/slurm_flow_prior_standard_adam.sh`；不得只从 `best.json` 重启并丢失 Adam 动量。需要“先从 128 个 IID 潜变量筛选起点”的批量实验时，旧的 screened-start 编排仍保留，但它不是标准优化器的默认入口。

## 旧 Python 研究路径

原有 Python CLI、Simsopt LS/Newton 和 DESC 接口仍被保留，用于历史示例和局部算法研究：

```bash
python -m stellarator_eval.cli \
  --case-file examples/01.json \
  --key raw \
  --output-dir runs/01_raw \
  --a 0.05
```

该入口不是 ABI-9 原生 score，也不是当前正式完整评估编排。批量筛选应使用 `scripts/smoke_native_score.py` / `scripts/batch_native_score.py`，物理验收应使用 `evaluation/full_physical/`。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `gpu_backend/` | CUDA 磁场、追踪、QR 和 ABI-9 score；ctypes 包装层 |
| `stellarator_eval/` | Python 物理模块、旧研究 API 和完整评估支撑代码 |
| `flow_matching/` | flow 模型、归一化、ODE 正反向积分 |
| `evaluation/full_physical/` | 正式单样本完整评估固定入口 |
| `scripts/` | 构建、评分、训练、优化、Slurm 和分析脚本 |
| `examples/` | 小型输入示例 |
| `tests/` | 单元测试和接口回归测试 |
| `docs/` | 方法、流程和性能文档 |
| `reports/` | 实验报告及版本化图表/验收产物 |

## 文档索引

- [QH 原生评分与潜空间优化：方法与实验](docs/QH原生评分与潜空间优化方法.md)：当前核心方法、ABI-9 定义和主要实验结论。
- [完整评估固定流程](docs/精简线圈评估流程.md)：从样本自适应 $\psi$ 到 $\alpha+\nu$、LS/Newton、Poincare 和 DESC 的唯一正式流程。
- [小条件潜空间 Adam 实验](reports/qh_small_condition_adam_report.md)：两线圈条件、脏梯度修复、续跑与完整验收。
- [修正后潜空间 landscape](reports/qh_flow_landscape_report.md)：FP32 RK4 闭环、潜空间/原空间宽度与平滑性对照。
- [微分体 QS 指标诊断](reports/qh_differential_qs_metric_investigation.md)：QA/QH/QP 尺度、$G$ 约定和体积权重审计。
- [后续研究方向可行性](reports/qh_future_directions_feasibility.md)：proxy、Reflow 和近似梯度/VJP 的证据与难度。

## 结果解释边界

- 固定成本 score 的 `a=0.05` 和定长筛面只适合排序；完整评估必须按样本重新选择 $a$ 和较大的可行面。
- `drift_rejected` 表示快速场线筛选未通过，不等价于“没有磁轴”，也不能证明标准 LS/Newton 一定找不到磁面。
- score 含候选选择、拓扑和有效性分支，不是全局光滑目标；高分区仍可能出现可行性边界和离散跳变。
- ABI-9 之前受电流符号、$G$ 尺度或体积 QS 权重问题影响的 score、landscape、proxy 标签和优化结果均为历史结果，不能与当前分数直接比较。
- flow checkpoint、score 动态库和代码 commit 共同定义一次实验；缺少其中任一哈希时，结果不能作为可复现实验基线。
