# 远端 simsopt 环境与 GPU 筛选全链路验证

## 1. 环境

远端服务器：

```text
host: master
path: ~/stellarator_gpu_eval
venv: ~/stellarator_gpu_eval/venv
```

已在独立虚拟环境中安装：

```text
python 3.12.2
numpy 2.5.1
scipy 1.18.0
simsopt 1.10.6
```

验证：

```python
from simsopt.field import BiotSavart
from simsopt.geo import CurveXYZFourier, SurfaceXYZTensorFourier
```

导入成功。

## 2. 01/raw 全链路 smoke

命令核心参数：

```bash
../venv/bin/python -m stellarator_eval.cli \
  --case-file examples/01.json --key raw \
  --output-dir runs/remote_01_gpu_screen_smoke \
  --a 0.05 \
  --levels 0.02,0.12 \
  --max-boozer-candidates 1 \
  --screen-trace-backend gpu \
  --screen-gpu-lib gpu_backend/build_mixed/libstellarator_gpu.so \
  --screen-gpu-precision mixed64 \
  --screen-gpu-verify-precision fp64 \
  --initial-iota -2.9 \
  --qs-sdim 8
```

结果：

```text
axis residual: 8.596e-10
has_axis: True
best surface: psi=0.12
iota: -2.9076809
volume: 0.00615026
G: 7.63113
warnings: []
```

GPU 筛选：

| level | mixed64 p95 | fp64 verify p95 | ok |
|---:|---:|---:|---|
| 0.02 | `7.854e-06` | `7.857e-06` | True |
| 0.12 | `3.464e-05` | `3.465e-05` | True |

关键耗时：

```text
axis_s: 45.28 s
psi_fit_s: 101.30 s
surface_screen_s: 1.063 s
surface_screen_curve_newton_s: 0.0258 s
surface_screen_fieldline_trace_s: 0.0725 s
boozer_candidates_s: 1.904 s
```

## 3. debug/raw 全链路 smoke

命令核心参数：

```bash
../venv/bin/python -m stellarator_eval.cli \
  --case-file examples/debug.json --key raw \
  --output-dir runs/remote_debug_gpu_screen_smoke \
  --a 0.02 \
  --levels 0.004,0.012 \
  --max-boozer-candidates 1 \
  --screen-trace-backend gpu \
  --screen-gpu-lib gpu_backend/build_mixed/libstellarator_gpu.so \
  --screen-gpu-precision mixed64 \
  --screen-gpu-verify-precision fp64 \
  --initial-iota -6.8 \
  --qs-sdim 8
```

结果：

```text
axis residual: 9.371e-09
has_axis: True
best surface: psi=0.012
iota: -2.1967522
volume: 8.62909e-05
G: 8.06092
warnings: []
```

GPU 筛选：

| level | mixed64 p95 | fp64 verify p95 | ok |
|---:|---:|---:|---|
| 0.004 | `1.209e-05` | `1.208e-05` | True |
| 0.012 | `2.200e-05` | `2.199e-05` | True |

关键耗时：

```text
axis_s: 47.22 s
psi_fit_s: 100.67 s
surface_screen_s: 1.106 s
surface_screen_curve_newton_s: 0.0251 s
surface_screen_fieldline_trace_s: 0.0805 s
boozer_candidates_s: 1.968 s
```

## 4. 结论

1. 远端已经可以运行 simsopt 主流程。
2. GPU psi0 筛选已经在完整 pipeline 中跑通。
3. 当前新瓶颈不是 psi0 筛选，而是 `axis_s` 和 `psi_fit_s`。
4. 下一步 GPU 化优先级应转向：

```text
axis search 接入 GPU backend
psi fit 的 B 采样/正规方程装配接入 GPU backend
```

