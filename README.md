# 局部磁面评估器

这个目录是从外层研究脚本中整理出来的第一版可维护封装。目标输入是线圈 Fourier 系数、电流和 `nfp`，输出磁轴、局部 $\psi$ 拟合、候选磁面筛选、最大可信磁面的 Boozer/准对称评估结果。

## 快速运行

在 WSL/远端的项目虚拟环境中运行：

```bash
cd /mnt/d/Typora/Typ/学习/stellarator/programs/local_surface_evaluator
../venv/bin/python -m stellarator_eval.cli \
  --case-file examples/01.json --key raw \
  --output-dir runs/01_raw \
  --a 0.05 \
  --initial-iota -2.0
```

`debug` 示例建议先用较小半径：

```bash
../venv/bin/python -m stellarator_eval.cli \
  --case-file examples/debug.json --key raw \
  --output-dir runs/debug_raw \
  --a 0.02 \
  --initial-iota -4.0 \
  --levels 0.001,0.002,0.004,0.008,0.012
```

默认会把 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 设为 `1`。这是有意的：当前小批次磁场调用在多线程下通常会被 OpenMP 同步开销拖慢。

默认磁轴搜索使用 GPU fixed-point 网格候选、Newton refinement 和低成本拓扑筛。最终只接受闭合残差达标且局部 Poincare 映射为 `elliptic` 的候选；若存在多个 elliptic 候选，默认优先选择 `topology_ellipse_aspect` 最接近 1 的候选。

## Python API

```python
from stellarator_eval import EvalConfig, evaluate_case_file

cfg = EvalConfig()
cfg.psi.a = 0.05
cfg.psi.poly_degree = 10
cfg.psi.m_tor = 12

result = evaluate_case_file("examples/01.json", key="raw", config=cfg, output_dir="runs/01_raw")
```

也可以直接传线圈：

```python
from stellarator_eval import evaluate_coils

result = evaluate_coils(
    coil_coefficients,  # dict{x,y,z} 或 shape=(ncoil,3,ncoef) 的数组；也支持扁平向量
    currents,
    nfp=8,
)
```

扁平向量格式按每根基础线圈：

```text
x[0:33], y[0:33], z[0:33], current
```

所以总长度是 `n_base_coils * 100`，`nfp` 作为额外参数传入。

## 输出

每次运行会生成：

- `summary.json`：完整结构化结果。
- `axis_data.npz`：磁轴采样、轴点和闭合残差。
- `psi_model.npz`：$\psi$ 模数、系数和磁轴数据。
- `level_*/boozer_surface.npz`：最终 Boozer surface 的 dofs、$\iota$、$G$。

主要字段：

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

## QUASR 批量评估

远端 QUASR 数据集可用 `scripts/eval_quasr.py` 评估。示例：

```bash
../venv/bin/python scripts/eval_quasr.py \
  --sample-size 128 \
  --sample-seed 20260705 \
  --helicity 1 \
  --output-dir runs/quasr_qh128 \
  --gpu-device 0 \
  --psi-n-r 64 --psi-n-z 64 --psi-n-phi 64 \
  --qs-sdim 8
```

批量输出包括 `batch_summary.json`、`batch_summary.csv`、逐样本 `summary.json`，以及失败样本导出的 JSON。

## 文档

- [计算流程说明](docs/计算流程.md)
- [性能与优化报告](docs/性能报告.md)
- [磁轴拓扑筛与圆度优先](docs/磁轴拓扑筛与圆度优先_20260705.md)
