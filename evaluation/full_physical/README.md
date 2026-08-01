# 完整单线圈物理评估入口

本目录是正式单样本评估的固定入口。底层物理实现仍只保留在 `scripts/` 和 `stellarator_eval/` 中；
这里提供提交、选面、交付校验和代码清单，禁止评估时临时拼接新脚本。

## 固定阶段

1. `submit_source_psi_candidates.sh`：对样本相关的 `A_VALUES` 并行运行稳定磁轴与 FP32 GPU QR $\psi$ 拟合；根据拟合误差、廉价场线筛选所覆盖的物理半径和外侧失败点选择源 $\psi$，不得复用别的样本的 `a`。
2. `submit_surface_candidates.sh`：对给定的 `S_EDGES` 运行 psi -> alpha -> nu、保守 guard 诊断、标准 LS/Newton 和独立密网格验收。
   默认每个候选申请 4 CPU 和 1 GPU 并行运行；在四卡 P107 上同时评估四个候选。只有资源受限时才显式设置
   `SERIAL_CANDIDATES=1`。可用 `CANDIDATE_CPUS_PER_TASK` 调整单候选 CPU 数，默认值为 4。
3. `select_largest_standard_surface.py`：只按标准 LS/Newton 的最终验证结果选择最大已测通过面；默认要求至少有一个更外侧失败点，否则要求继续外扩。
4. `submit_downstream.sh`：对选中的唯一 `boozer_standard.npz` 运行庞加莱、Boozer 场图、三维 HTML 和 DESC。直接 Boozer 与 DESC 的 $|B|$ 图均固定为白底彩色等高线，颜色表示 $|B|$ 大小，不使用热力图或填色等高线。
5. `validate_delivery.sh`：检查固定原始产物，并确认全部 DESC PNG 已在报告中逐张引用。

三个提交入口会自动运行 `preflight.py` 和 `sbatch --test-only`。也可在提交前单独检查代码包：

```bash
python evaluation/full_physical/preflight.py
```

## 代码清单

完整机器可读清单见 `code_manifest.json`。主要实现为：

| 阶段 | 唯一实现 |
|---|---|
| source $\psi$ 候选 | `scripts/slurm_fit_source_psi.sh` |
| alpha + nu + guard 诊断 + 标准验收作业 | `scripts/slurm_alpha_nu_guarded_boozer.sh` |
| alpha 拟合 | `scripts/alpha_clebsch_ls_experiment.py` |
| nu 拟合与环向修正 | `scripts/diagnose_alpha_toroidal_correction.py` |
| 保守 guard 诊断 | `scripts/guarded_boozer_from_alpha_nu.py` |
| 标准 LS/Newton 验收 | `scripts/solve_boozer_from_alpha_nu.py` |
| 完整下游作业 | `scripts/slurm_evaluate_saved_boozer_full.sh` 或 CPU 版本 |
| 庞加莱、HTML、DESC 编排 | `scripts/evaluate_saved_boozer_surface_full.py` |
| DESC 报告图校验 | `scripts/validate_desc_report_artifacts.py` |

## 固定执行

所有远端路径必须位于 `$HOME` 下。先设置输入和候选层：

```bash
export PROJECT=$HOME/local_surface_evaluator_worktrees/<branch>
export GPU_LIB=$HOME/local_surface_evaluator/gpu_backend/build_mixed/libstellarator_gpu.so
export EVAL_ENV=$HOME/local_surface_evaluator/.venv-desc016-py312
export CASE_FILE=$PROJECT/runs/<optimizer>/<job>/best.json
export OUTPUT_ROOT=$PROJECT/runs/<evaluation_name>
export A_VALUES=0.04,0.05,0.06,0.08
bash evaluation/full_physical/submit_source_psi_candidates.sh
```

源候选完成后，根据验证误差、通过场线筛选的物理半径及更外侧失败点选择该样本的 `RUN_DIR`，再提交磁面候选：

```bash
export RUN_DIR=$OUTPUT_ROOT/source_psi_candidates/a_<selected>
export S_EDGES=0.12,0.20,0.24
bash evaluation/full_physical/submit_surface_candidates.sh
```

监控命令已经写入标准输出；也可使用：

```bash
squeue -u "$USER" -o '%.18i %.12T %.10M %.30j %R'
```

候选全部完成后选择最大已测可行面并提交下游。`DESC_BACKEND` 必须显式选择，不能静默回退：

```bash
python evaluation/full_physical/select_largest_standard_surface.py --candidate-root "$OUTPUT_ROOT/candidates" --output "$OUTPUT_ROOT/selection.json"
export DESC_BACKEND=cpu
bash evaluation/full_physical/submit_downstream.sh
```

报告和产物复制完成后执行最终校验：

```bash
export REPORT=reports/<report>.md
export DESC_DIR=reports/assets/<case>/desc
bash evaluation/full_physical/validate_delivery.sh
```

guard 诊断的退出码 3 只表示保守路径未通过，候选作业会继续运行标准 LS/Newton；除此之外任一步退出码非零时停止正式评估。最终磁面存在性只按标准求解收敛和独立密网格验证判断。调试必须使用新的 debug 输出目录，不能修改或续写正式目录。
