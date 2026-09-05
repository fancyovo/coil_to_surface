# 单作业完整物理评估入口

本目录提供单样本完整评估的固定代码入口。标准流程只提交一个 Slurm
作业，由代码自动完成 source-psi 半径 `a` 选择、磁面层 `s` 选择、标准
Simsopt LS/Newton 验收、Poincare 与 Boozer 图、三维 HTML 以及 DESC
平衡求解。作业运行期间不需要人工筛选或再次提交。

## 标准入口

设置输入和输出后运行：

```bash
export PROJECT=$HOME/local_surface_evaluator_worktrees/<branch>
export GPU_LIB=$HOME/local_surface_evaluator/gpu_backend/build_mixed/libstellarator_gpu.so
export EVAL_ENV=$HOME/local_surface_evaluator/.venv-desc016-py312
export CASE_FILE=$PROJECT/runs/<optimizer>/<job>/best.json
export OUTPUT_ROOT=$HOME/local_surface_evaluator_runs/<evaluation_name>
bash evaluation/full_physical/submit_full_evaluation.sh
```

入口只调用一次 `sbatch`。默认在 P107 的一个四卡、16 CPU、128 GB
分配内运行，最长 4 小时。需要使用 Students 两卡池时设置：

```bash
export FULL_EVAL_POOL=students
bash evaluation/full_physical/submit_full_evaluation.sh
```

For a one-GPU job limited to four hours and 16 GB under `qos_stu_default`, set
`FULL_EVAL_POOL=students-default`. Submit independent samples as independent
jobs so they can run concurrently when resources permit.

提交器返回唯一 Job ID，并写入 `$OUTPUT_ROOT.job_id`。状态保存在
`$OUTPUT_ROOT/run_status.json`；完成后的统一验收入口为
`$OUTPUT_ROOT/evaluation_summary.json`。

## 固定自动流程

1. 同一作业内并行计算 `a=0.04,0.05,0.06,0.08 m` 四个 source-psi
   候选，每张可用 GPU 同时承担一个候选。
2. `select_source_psi_candidate.py` 要求磁轴闭合残差不超过 `1e-6 m`、
   psi 验证 RMS 不超过 `5e-4`、验证/训练 RMS 比不超过 2，并要求存在
   FP64 复核通过的内层和更外侧的失败层。合格候选按最大已验证物理
   平均半径选择，验证误差用于平局裁决。
3. 使用选中的 source-psi，在同一作业内并行计算
   `s=0.12,0.24,0.36,0.49,0.64,0.81,1.0`。候选数超过 GPU 数时自动
   分批，GPU 一空闲就领取下一个候选。
   每个候选都必须生成 `kind=alpha_nu` 的初值；alpha-only 和直接 GPU
   点云产物会在标准求解器入口被拒绝。
4. `select_largest_standard_surface.py` 按标准 Simsopt LS/Newton、独立
   密网格诊断和封闭体积随 `s` 单调增加的连续分支规则，选择最大可继续
   下游的面。稠密 residual 超过严格 `1e-4` 时保留该面并写入质量警告；
   求解器未收敛、绕行方向错误或法向退化仍会拒绝。标准流程要求存在
   更外侧失败点。
5. 对唯一选中面运行 Poincare、直接 Boozer 指标、线圈与磁面 HTML、
   静态图片和 CPU DESC。最终 JSON 汇集配置、代码版本、输入及评分库
   哈希、候选结果、选面结果、数值诊断、资产路径和各阶段耗时。
   下游只接受 `kind=alpha_nu_standard_ls_newton` 的标准面，因此无法用
   alpha-only 或直接点云结果绕过 alpha+nu 与标准 LS/Newton。

外层 `s` 可能因固定预算体采样点不足或硬性 Simsopt/几何检查而正常淘汰；
这两条路径写入结构化 JSON，并以候选状态 `rejected` 参与外侧边界判定。
程序异常、缺少结构化淘汰记录、没有合格 `a` 或 `s`、最大测试 `s`
仍通过而缺少外侧边界，以及下游计算失败，都会让整个作业明确失败，
阶段和错误写入 `run_status.json`。

标准质量等级写在每个 `standard_rho_1/summary.json` 中：
`evaluation_quality=strict_pass` 表示两项独立稠密指标均不超过 `1e-4`；
`evaluation_quality=accepted_with_quality_warning` 表示硬性求解/几何条件
通过但 `dense_relative_l2` 或 `dense_normal_field_p95` 超过该严格等级。
后者可以继续生成 Poincare、Boozer、HTML 和 DESC，但正式报告必须逐项
展示实际数值、严格限值和 warning，不得简称为严格通过。

## 并行与资源

`run_full_evaluation.py` 从 Slurm 提供的 `CUDA_VISIBLE_DEVICES` 建立 GPU
队列。每个 source-psi 或 `s` 候选占用一张 GPU 和 4 个 CPU；P107
默认同时运行四个，Students 同时运行两个。不同候选互相独立，完成
顺序不影响选择。DESC 与最终绘图在候选阶段结束后使用整个 CPU 分配，
这是流程中唯一必要的阶段依赖。

可通过 `A_VALUES` 和 `S_EDGES` 显式登记新的候选网格，但正式复现必须
保存作业生成的配置和清单。修改候选网格属于新评估配置，不能在作业
运行期间人工追加。

## 验收

作业完成后只需：

1. 确认 `run_status.json` 与 Slurm ExitCode 均为完成状态；
2. 读取 `evaluation_summary.json` 和其引用的原始 JSON；
3. 检查数值结果、Poincare、Boozer、线圈/磁面 HTML 和全部 DESC 图；
4. 将报告与资产复制到本地并运行 `validate_delivery.sh`。

```bash
export REPORT=reports/<report>.md
export DESC_DIR=reports/assets/<case>/full/desc
bash evaluation/full_physical/validate_delivery.sh
```

`validate_delivery.sh` 检查固定原始产物，并确认报告逐张引用成功生成的
DESC PNG。它是交付校验，不参与作业内的物理筛选。

## 低层诊断工具

以下旧入口保留用于定位单个阶段的问题，不再作为日常完整评估流程：

- `submit_source_psi_candidates.sh`
- `submit_surface_candidates.sh`
- `submit_downstream.sh`
- `write_submission_policy.py`

调试必须使用新的 debug 输出目录。低层工具产生的局部结果不能冒充
标准单作业完整评估；正式结论以 `evaluation_summary.json` 及其固定
原始产物为准。

## 代码清单

完整机器可读清单位于 `code_manifest.json`。核心入口为：

| 功能 | 固定实现 |
|---|---|
| 单次提交 | `submit_full_evaluation.sh` |
| Slurm 作业 | `scripts/slurm_full_physical_evaluation.sh` |
| 作业内编排 | `run_full_evaluation.py` |
| 自动选择 `a` | `select_source_psi_candidate.py` |
| 自动选择 `s` | `select_largest_standard_surface.py` |
| source-psi 计算 | `scripts/slurm_fit_source_psi.sh` |
| alpha、nu 与 Simsopt 验收 | `scripts/slurm_alpha_nu_guarded_boozer.sh` |
| Poincare、HTML 与 DESC | `scripts/slurm_evaluate_saved_boozer_full_cpu.sh` |

guard 诊断退出码 3 表示保守路径未通过，随后继续标准 LS/Newton。
alpha 固定预算体采样不足和标准 LS/Newton 淘汰也使用退出码 3，并分别
由 `alpha/rejection.json` 与 `standard_rho_1/summary.json` 确认。编排器
只接受带有对应结构化记录的候选退出码 3；其余非零退出均判定为作业
故障。
