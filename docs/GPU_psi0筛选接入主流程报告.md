# GPU psi0 筛选接入主流程报告

## 1. 本次目标

把前面已经验证过的 GPU 一周期磁力线追踪接入主 evaluator 的候选 $\psi_0$ 筛选阶段。

这一步只改 cheap screen，不动 Boozer/LS/Newton：

```text
axis search -> psi fit -> psi0 screen -> Boozer candidate
                         ^
                         本次接入 GPU batch trace
```

## 2. 代码改动

### 2.1 配置项

在 `SurfaceScanConfig` 中新增：

```python
trace_backend: str = "cpu"
gpu_lib_path: str = "gpu_backend/build_mixed/libstellarator_gpu.so"
gpu_segments_per_coil: int = 256
gpu_device: int = 0
gpu_trace_precision: str = "mixed64"
gpu_verify_precision: str = "fp64"
gpu_threads_per_line: int = 256
gpu_verify_candidates: int = 3
```

默认仍是 CPU 路径，因此不影响原有运行。

### 2.2 CLI 参数

新增：

```bash
--screen-trace-backend cpu|gpu
--screen-gpu-lib
--screen-gpu-precision mixed64|fp64|fp32
--screen-gpu-verify-precision mixed64|fp64|fp32|none
--screen-gpu-verify-candidates
--screen-gpu-segments
--screen-gpu-device
```

启用 GPU 筛选的典型参数：

```bash
--screen-trace-backend gpu \
--screen-gpu-lib gpu_backend/build_mixed/libstellarator_gpu.so \
--screen-gpu-precision mixed64 \
--screen-gpu-verify-precision fp64 \
--screen-gpu-verify-candidates 3
```

### 2.3 pipeline 接入

`pipeline.py` 中的 $\psi_0$ 筛选阶段现在逻辑是：

```text
如果 trace_backend == "gpu":
    调用 screen_levels_gpu(...)
    如果失败，记录 warning 并回退 CPU
否则:
    使用原 CPU screen_level(...)
```

### 2.4 GPU 批量筛选

新增函数：

```python
screen_levels_gpu(field_input, model, levels, cfg, current_unit)
```

它做的事情是：

1. 对每个 level 用多项式化 Newton 提取 $\Phi=0$ 曲线。
2. 把所有 level 的起点拼成一个 batch。
3. 一次 GPU trace 追踪全部起点一周期。
4. 对每个 level 计算 drift p95 和相对 drift。
5. 对通过筛选的最大若干个 level 再用 `fp64` 复核。

## 3. 实测 smoke

远端服务器没有安装 simsopt，因此不能在那里跑完整 Boozer 主流程；但已经用真实 01/debug 数据直接跑通 `screen_levels_gpu` 组件。

### 3.1 01/raw

测试 level：

```text
0.02, 0.12
```

结果：

| level | mixed64 ok | mixed64 p95 | fp64 verify | fp64 p95 | batch trace |
|---:|---|---:|---|---:|---:|
| 0.02 | True | `8.979e-06` | True | `8.982e-06` | 0.0728 s |
| 0.12 | True | `4.259e-05` | True | `4.260e-05` | 0.0728 s |

总 `screen_levels_gpu` 时间约 `1.06 s`，其中包含 GPU field 创建和 fp64 复核。

### 3.2 debug/raw

测试 level：

```text
0.012, 0.16
```

结果：

| level | mixed64 ok | mixed64 p95 | fp64 verify | fp64 p95 | batch trace |
|---:|---|---:|---|---:|---:|
| 0.012 | True | `2.646e-05` | True | `2.646e-05` | 0.0805 s |
| 0.16 | True | `1.053e-04` | True | `1.053e-04` | 0.0805 s |

总 `screen_levels_gpu` 时间约 `1.13 s`。

## 4. 注意事项

1. 默认没有启用 GPU 筛选，必须显式设置 `trace_backend="gpu"` 或 CLI `--screen-trace-backend gpu`。
2. cheap screen 默认推荐 `mixed64`。
3. 通过筛选的最大若干个 level 用 `fp64` 复核。
4. 如果 GPU 筛选失败，pipeline 会自动回退 CPU，并在 `warnings` 中记录原因。
5. 远端当前没有 simsopt，因此完整主流程仍需要在 WSL/本地 simsopt 环境中跑；远端目前只验证 GPU 筛选组件。

## 5. 下一步

建议下一步做两个小收尾：

1. 在 WSL/simsopt 环境中跑一次完整 CLI：

```bash
../venv/bin/python -m stellarator_eval.cli \
  --case-file examples/01.json --key raw \
  --output-dir runs/01_raw_gpu_screen \
  --screen-trace-backend gpu \
  --screen-gpu-lib gpu_backend/build_mixed/libstellarator_gpu.so
```

2. 如果完整流程确认无误，再考虑把 axis search 也从 CPU simsopt 路径切到 GPU backend。

