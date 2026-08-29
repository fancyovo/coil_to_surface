# Codex 远程算力运行规程

本文件供 Windows 本地 Codex 使用。目标是复用用户已经认证的 WSL SSH 主连接，在 USTC Slurm 集群上开发和提交计算任务。按本规程执行，不在连接或路径异常时临场猜测。

## 固定环境

- WSL 发行版：`Ubuntu`
- SSH 别名：`ustc107`
- 远端用户：`pb24511935`
- 当前登录节点：`tradmin-02`（登录节点可能由平台调整，不把主机名变化单独视为故障）
- Slurm 版本：`25.11.2`
- 可用 account：`competition`、`stu`
- 可用 GPU 分区：`P107-RTX5090`、`P107-A100`、`Students`
- 当前首选：`competition / P107-RTX5090 / qos_p107-rtx5090`
- 当前 GPU 类型：优先申请 `RTX5090`；准确 GRES 写法为 `gpu:RTX5090:N`

权限于 2026-07-17 更新，并使用 `scontrol`、`sacctmgr` 和 `sbatch --test-only` 重新核验。后续若 association/QOS 再次变化，仍需重新只读核验，不能沿用本表。

### 当前可用提交通道（2026-07-17）

| Account / Partition / QOS | 每用户并发资源 | 作业数 | 最长时间 | 节点资源与用途 |
|---|---|---|---|---|
| `competition / P107-RTX5090 / qos_p107-rtx5090` | 16 CPU、4 GPU；QOS 未设内存上限 | 最多运行 4、提交 10 | 4 天 | 15 节点；每节点 128 CPU、500G、8×RTX5090；GPU 训练首选 |
| `competition / P107-A100 / qos_p107-a100` | 16 CPU、4 GPU；QOS 未设内存上限 | 最多运行 4、提交 10 | 4 天 | 2 节点；每节点 128 CPU、1000G、8×A100；仅在排队时间优于 RTX 时使用 |
| `stu / Students / qos_stu_medium_2gpu` | 24 CPU、2 GPU、128G | 未额外限制作业数 | 1 天 | RTX5090 训练的并行补充通道；也承担较长 CPU 作业 |
| `stu / Students / qos_stu_default` | 4 CPU、1 GPU、16G | 最多运行 4、提交 10 | 4 小时 | 只放可在 4 小时内完成、内存不超过 16G 的短作业 |

`competition` association 已明确授予 `qos_p107-rtx5090` 和 `qos_p107-a100`；两个 P107 最大规格的 test-only 均通过。普通 `GPU-RTX5090/GPU-A100/CPU-*` 分区要求的 `qos_gpu-*`/`qos_cpu-*` 未授予，仍不可使用。

本集群的 `sbatch --test-only` 预计启动时间已确认不准确，不得用于判断
等待时长、比较分区或决定是否提交。该命令只验证脚本、account、partition、
QOS 和资源请求能否被调度器接受；实际启动与运行状态只看正式提交后的
`squeue`、`scontrol`、`sacct` 和作业日志。

P107 QOS 没有 `DenyOnLimit`，因此超过 16 CPU/4 GPU/4 天的请求可能被 `sbatch` 接受却永久 pending；最大值以 `sacctmgr` 的 QOS 字段为准，不能以“提交命令返回 0”判断可运行。

## 安全边界

- 不读取、打印、复制或修改 `D:\FPC\2.4.0\bin\i386-win32\new\keys` 中的凭据文件。
- 不要求用户提供私钥 passphrase、六位验证码、TOTP 密钥或恢复码。
- 不自行发起交互式 SSH 认证，不尝试绕过多因素认证。
- 不在未获明确授权时使用 `rsync --delete`、批量删除、覆盖远端数据或终止无关进程。
- 不在远端启动 Codex CLI。本地 Codex 负责规划与操作，远端只运行项目工具和 Slurm 作业。

## 每轮任务的强制预检

执行任何远端读取、编辑、同步或作业操作前，严格按以下顺序检查。一次任务中连接保持正常时不必在每条命令前重复，但在网络切换、电脑休眠或出现 SSH 错误后必须从头检查。

### 1. 检查 WSL

```powershell
wsl.exe -d Ubuntu -- true
```

失败时停止。向用户报告“WSL Ubuntu 未启动或不可用”，建议用户运行：

```powershell
wsl.exe -l -v
wsl.exe -d Ubuntu
```

### 2. 检查主连接

```powershell
wsl.exe -d Ubuntu -- ssh -O check ustc107
```

只有输出包含 `Master running` 才继续。若出现 `Control socket ... No such file`、`Connection refused` 或退出码非零，停止并明确告诉用户：

```text
SSH 主连接已断开或不存在。请在单独 PowerShell 窗口运行下面命令，
输入 passphrase 和六位验证码后保持窗口开启，然后让我重新检查：

wsl.exe -d Ubuntu -- ssh -M -N -o ControlPersist=no ustc107
```

Codex 不得自行运行上述交互式启动命令。

### 3. 检查远端命令通道和身份

```powershell
wsl.exe -d Ubuntu -- ssh -o BatchMode=yes -o ConnectTimeout=10 ustc107 -- id -un
wsl.exe -d Ubuntu -- ssh -o BatchMode=yes -o ConnectTimeout=10 ustc107 -- hostname -s
```

第一条必须输出 `pb24511935`。第二条当前应输出 `tradmin-02`，但平台更换登录节点时允许主机名变化。失败时按“故障预案”报告，不继续执行项目操作。

### 4. 检查远端项目路径

不得猜测远端项目目录。若用户尚未提供绝对路径，先询问一次。获得路径后，以安全引用的绝对路径执行：

```powershell
wsl.exe -d Ubuntu -- ssh ustc107 -- test -d /远端项目绝对路径
wsl.exe -d Ubuntu -- ssh ustc107 -- test -r /远端项目绝对路径
wsl.exe -d Ubuntu -- ssh ustc107 -- test -w /远端项目绝对路径
wsl.exe -d Ubuntu -- ssh ustc107 -- realpath /远端项目绝对路径
```

- 目录不存在：报告准确路径，并用只读命令检查最近的已存在父目录；不得擅自创建或换到同名目录。
- 不可读或不可写：报告 `ls -ld` 的 owner、group、mode，询问用户应改用哪个路径；不得擅自 `chmod -R` 或 `chown`。
- `realpath` 与用户给定路径不同：说明它是符号链接或规范化路径，后续统一使用 `realpath` 输出；这本身不是错误。

### 5. 检查 Slurm 能力

```powershell
wsl.exe -d Ubuntu -- ssh ustc107 -- sinfo --version
wsl.exe -d Ubuntu -- ssh ustc107 -- scontrol show partition Students -o
wsl.exe -d Ubuntu -- ssh ustc107 -- sacctmgr -nP show assoc where user=pb24511935
```

确认三个授权分区均为 `UP`，`competition` 包含两个 P107 QOS，`stu/Students` 包含两个学生 QOS。若名称或上限变化，停止提交，报告实际输出并更新本文件，不得猜测。

`sinfo` 会显示多个硬件分区，但“可见”不等于“可提交”。普通 `CPU-6530`、`CPU-8358P`、`GPU-RTX5090`、`GPU-A100` 所需的专用 QOS 当前仍未授予。

## 登录节点使用禁令

登录节点只允许轻量级操作：浏览和编辑少量文本、`git status/diff`、查看小型日志、检查环境、查询 Slurm、生成提交脚本以及执行 `sbatch/squeue/sacct/scancel`。

以下任务禁止直接在登录节点运行，必须交给 Slurm 计算节点：

- 编译、完整测试套件、benchmark；
- Python/Julia/Matlab 数值计算或批量数据处理；
- 模型训练、推理、GPU 探测或 CUDA 程序；
- 仿真、优化、参数扫描和任何明显占用 CPU、内存、I/O 的任务；
- 长时间后台进程、`nohup` 重型任务或用 `tmux` 绕过 Slurm。

无法判断任务是否轻量时，默认提交 Slurm。不要因为登录节点显示 48 CPU 和 125 GiB 内存就使用这些资源。

## Slurm 资源规则

- 按实际需求申请，不默认占满上限；独立单卡训练保持“一配置一作业”，不把 4 张 GPU 塞进一个作业。
- 每次提交前查询各 QOS 的 running/pending 数和分区空闲 GPU。默认顺序是：P107-RTX5090 → Students medium 并行补充 → P107-A100（仅当预计启动更早）→ Students default（仅短小作业）。
- 单卡训练在 P107 优先申请 `4 CPU + 1 GPU + 24G`；这样可在 16 CPU/4 GPU 上同时运行 4 个作业。若数据加载实测需要 8 CPU，则该 QOS 只能同时运行 2 个，不得超配。
- Students medium 保留现有 `8 CPU + 1 RTX5090 + 24G` 模板，可同时运行 2 个训练；Students default 只运行 `<=4 h、<=16G` 的短任务。
- 需要 RTX 5090/A100 时分别显式写 `--gres=gpu:RTX5090:1`/`--gres=gpu:A100:1`。
- P107 单次提交总数不得超过 10；已满时把后续作业保留在本地提交清单，不要制造无效 pending。
- 已运行作业占满 QOS 上限时，后续合规作业 pending 是正常行为；不要重复提交。
- 不提交超过 QOS 上限的作业；P107 超限请求可能被接受但永远不能运行。

### 动态选择准则

1. 先排除不满足作业时长、内存和 GPU 型号的通道。
2. 正式提交后用 `squeue` reason 和实际状态判断是否启动；忽略 `sbatch --test-only` 输出的预计开始时间。
3. RTX5090 与 A100 结果需保持数值口径一致，但性能测速不得跨型号混合；正式训练可以跨型号，必须在结果元数据中记录硬件。
4. 已运行作业不为迁移而取消；只迁移尚未启动且由本项目提交的 pending 作业，并先确认目标通道有空余 submit slot。
5. 同一 run 的 checkpoint/resume 路径保持不变；迁移前确认旧作业从未启动，避免两个作业同时写同一目录。

P107-RTX5090 最大并发额度模板已通过 `sbatch --test-only` 验证；实际独立训练仍应拆成 4 个单卡作业：

```bash
#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
```

Students 最大规格模板也已通过 `sbatch --test-only`：

```bash
#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --job-name=PROJECT_TASK
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:RTX5090:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /远端项目绝对路径

# 按项目要求加载环境，例如 source venv/bin/activate 或 module load ...
# 在这里运行实际计算命令。
```

提交前必须：

1. 将资源缩减到任务实际需要的数量。
2. 确保远端项目中的 `logs/` 存在。
3. 使用 `sbatch --test-only 脚本路径` 校验脚本、权限和资源请求，并忽略其预计启动时间。
4. 校验成功后才运行 `sbatch 脚本路径`。
5. 立即记录返回的 Job ID、脚本路径、日志路径和提交时间。

## 远端命令执行方式

轻量单命令：

```powershell
wsl.exe -d Ubuntu -- ssh ustc107 -- git -C /远端项目绝对路径 status --short
```

复杂的轻量操作或 Slurm 提交流程应合并成一次 shell 调用，使用 stdin 避免 PowerShell/SSH 多层转义：

```powershell
@'
set -euo pipefail
cd /远端项目绝对路径
git status --short
squeue -u "$USER"
'@ | wsl.exe -d Ubuntu -- ssh ustc107 -- bash -s
```

在修改文件前先确认权威副本位于本地还是远端。不得默认双向同步。需要同步时先确认两端目录和排除规则，默认不用 `--delete`。

## 作业跟踪与结果

提交后使用以下只读命令跟踪，不在登录节点重复运行任务：

```bash
squeue -j JOB_ID -o '%.18i %.16P %.24j %.12q %.10T %.10M %.4D %R'
scontrol show job JOB_ID
sacct -j JOB_ID --format=JobID,JobName,Partition,QOS,State,ExitCode,Elapsed,AllocCPUS,ReqMem,AllocTRES%80
```

- `PENDING`：读取 `Reason`。资源/QOS 原因通常只需等待。
- `RUNNING`：查看日志增量，不频繁轮询；默认间隔至少 30 秒。
- `COMPLETED`：检查退出码、输出文件和必要的数值/测试结果。
- `FAILED`、`TIMEOUT`、`OUT_OF_MEMORY`、`CANCELLED`：读取 `sacct` 和日志后解释根因，再决定是否修改资源或代码；不得无分析地重复提交。
- 只允许取消本轮由 Codex 明确提交且用户不再需要的作业；取消前报告 Job ID 和原因。

## 固定故障预案

| 现象 | 判断 | Codex 必须采取的动作 |
|---|---|---|
| WSL 命令失败 | 本地 WSL 不可用 | 停止；让用户检查 `wsl.exe -l -v` 并启动 Ubuntu |
| `Control socket` 不存在/拒绝 | SSH 主连接已断 | 停止；让用户运行文档中的 `ssh -M -N -o ControlPersist=no` 命令并重新认证 |
| `Permission denied` 或出现验证码提示 | 未复用主连接或连接已失效 | 不输入、不猜测、不重试；让用户重建主连接 |
| `Connection timed out` / `No route to host` | 跳板机、网络或目标不可达 | 停止；报告失败层级，保留本地状态，让用户恢复网络并重建连接 |
| 身份不是 `pb24511935` | SSH 配置或别名错误 | 停止所有远端操作；报告实际用户和主机，不修改数据 |
| 项目路径不存在 | 路径错误、未挂载或项目移动 | 报告准确路径与最近存在的父目录；询问正确路径，不擅自创建 |
| 项目路径权限不足 | 用户/组/ACL 不匹配 | 报告 `ls -ld`；不递归改权限，等待用户确认 |
| `sinfo/sbatch` 不存在 | 登录到了错误环境或 Slurm 未加载 | 停止计算任务；报告 PATH、主机和命令缺失 |
| Partition/QOS/account 无效 | 集群策略发生变化 | 只读核验 `scontrol`/`sacctmgr`，报告变化并更新本文件后再提交 |
| 作业长期排队 | 资源繁忙或 QOS 上限 | 报告 `squeue` Reason；若是正常资源等待则等待，不重复提交 |
| SSH 在 `sbatch` 返回 Job ID 前断开 | 提交结果不确定 | 按 job-name、用户和提交时间查询 `squeue/sacct`；确认不存在后才可重提 |
| 作业失败 | 代码、环境或资源问题 | 收集 Job ID、State、ExitCode、日志尾部和资源信息，解释后再修复 |

每次故障报告必须包含：失败阶段、执行的命令类别、退出码、关键错误、已确认成功的最后一步、用户下一条应执行的准确命令。不得只说“连接失败”或盲目重试。
