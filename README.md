# Local Surface Evaluator

局部磁面评估器。给定线圈 Fourier 系数、电流和 `nfp`，程序会搜索磁轴、拟合局部不变量 $\psi$、筛选可置信等 $\psi$ 面，并在候选磁面上做 Boozer surface / QS 指标评估。

当前实现主要用于快速筛选和诊断，不替代高精度平衡或长时间磁面验证。默认流程偏向“快速失败”：如果找不到磁轴或局部磁面质量不足，会返回结构化失败原因和最好的残差信息，而不是长时间卡住。

## 安装

```bash
python -m pip install -e .
```

完整 Boozer/QS 评估需要 `simsopt`。GPU 后端需要先编译 `gpu_backend` 下的 CUDA/C++ 库；没有 GPU 后端时，部分脚本只能运行 CPU 或已有的 Python/Simsopt 路径。

## 快速运行

```bash
python -m stellarator_eval.cli \
  --case-file examples/01.json --key raw \
  --output-dir runs/01_raw \
  --a 0.05 \
  --initial-iota -2.0
```

`debug` 示例更极端，建议先用较小半径：

```bash
python -m stellarator_eval.cli \
  --case-file examples/debug.json --key raw \
  --output-dir runs/debug_raw \
  --a 0.02 \
  --initial-iota -4.0 \
  --levels 0.001,0.002,0.004,0.008,0.012
```

默认会把 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 设为 `1`。这是有意的：当前很多小批量磁场计算在多线程下会被同步开销拖慢。

## Python API

```python
from stellarator_eval import EvalConfig, evaluate_case_file

cfg = EvalConfig()
cfg.psi.a = 0.05
cfg.psi.poly_degree = 10
cfg.psi.m_tor = 12

result = evaluate_case_file(
    "examples/01.json",
    key="raw",
    config=cfg,
    output_dir="runs/01_raw",
)
```

也可以直接传线圈：

```python
from stellarator_eval import evaluate_coils

result = evaluate_coils(
    coil_coefficients,
    currents,
    nfp=8,
)
```

扁平输入格式按每根基础线圈组织：

```text
x[0:33], y[0:33], z[0:33], current
```

因此线圈部分总长度为 `n_base_coils * 100`，`nfp` 作为额外参数传入。

## 输出

每次运行会在 `output-dir` 中生成：

- `summary.json`：完整结构化结果。
- `axis_data.npz`：磁轴采样、轴点和闭合残差。
- `psi_model.npz`：$\psi$ 模数、系数和磁轴数据。
- `level_*/boozer_surface.npz`：候选 Boozer surface 的自由度、$\iota$、$G$ 等。

常用字段：

- `axis.has_axis`
- `axis.best_residual`
- `axis.topology_class`
- `axis.topology_ellipse_aspect`
- `psi.fit_info`
- `surface_screen.levels`
- `best_surface.iota`
- `best_surface.volume`
- `best_surface.G`
- `best_surface.qs_error_QA_1_0`
- `best_surface.qs_error_QH_1_1`
- `best_surface.qs_error_QP_0_1`
- `timing`

`axis.best_residual` 是一周期追踪后的最好闭合距离；即使 `has_axis=false`，这个值仍会输出，用于判断是“接近但未达阈值”还是“完全没有可用闭合点”。

## QUASR 批量评估

QUASR 数据集不随仓库发布。运行批量评估时显式传入数据位置，或设置环境变量：

```bash
export QUASR_ROOT=/path/to/quasr/data
export QUASR_METADATA=/path/to/quasr/metadata.csv
```

示例：

```bash
python scripts/eval_quasr.py \
  --quasr-root "$QUASR_ROOT" \
  --metadata "$QUASR_METADATA" \
  --sample-size 128 \
  --sample-seed 20260705 \
  --helicity 1 \
  --output-dir runs/quasr_qh128 \
  --gpu-device 0 \
  --psi-n-r 80 --psi-n-z 80 --psi-n-phi 80 \
  --qs-sdim 16
```

批量输出包括 `batch_summary.json`、`batch_summary.csv`、每个样本的 `summary.json`，以及导出的失败样本 JSON。导出的失败样本默认写入本地 `examples/`，但 `examples/debug_*.json` 已被 `.gitignore` 忽略。

## 文档

- [计算流程](docs/计算流程.md)
- [性能报告](docs/性能报告.md)
