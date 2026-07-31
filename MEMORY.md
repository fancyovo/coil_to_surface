# Local Surface Evaluator Project Memory

> **Living source of truth. Last updated: 2026-07-31 (Asia/Shanghai).**
>
> Every new conversation and every post-compaction continuation must read this
> file first. Important changes must be written here in the same turn. Do not
> store credentials in this file.

## 1. Maintenance Protocol

Update this file immediately when any of the following changes:

- active branch, commit, or ownership boundary;
- remote job submission, cancellation, completion, failure, or result path;
- validated model, CUDA library, dataset, checkpoint, or SHA-256;
- score definition, physics convention, interface, or production workflow;
- a numerical conclusion used to choose the next experiment;
- an error that invalidates old results or must not recur;
- a durable user requirement.

Keep current state near the top. Move completed jobs into history rather than
silently deleting them. Distinguish these three quantities explicitly:

1. a score recorded in an old artifact;
2. a score obtained by re-decoding the same latent;
3. a score obtained with the current validated score binary.

The newest dated correction wins if older text conflicts with it. Large reports
remain the detailed evidence; this file records the conclusions and pointers.

Keep this file selective. Record only facts needed to resume after compaction,
durable workflow requirements, validated numerical conclusions, active jobs,
and mistakes that must not recur. Detailed scheduler history, every trial
parameter, and routine intermediate measurements belong in reports/artifacts,
not permanently in memory. It is acceptable to record temporary job details
while work is active, but collapse them to the final outcome and report pointer
once the task is accepted.

## 2. Current Snapshot

- Active local and remote branch: `qh-flow-zo-adam`.
- Current experiment implementation baseline:
  `cc69110d0a5663a50fa56ac97a671973bd6f064d`
  (`Pin Adam jobs to corrected score library`). The branch also contains this
  living-memory documentation; query `git rev-parse HEAD` at session start for
  the actual tip rather than writing a self-referential commit hash here.
- Complete physical-evaluation report and assets were delivered in commit
  `4071dcc9c1132f4bf1f05e85580aa140b19477b3`.
- Local `main`: `8c20859f9c66ca690d5c22cce862c055b634c1d0`.
- Current objective: the CEM-initialized standard-Adam run has been accepted,
  but the unscreened random-start probability experiment is incomplete because
  its multistart runner failed after one seed. Do not restart or rerun it until the
  user explicitly authorizes another run.
- Fixed optimizer learning rate for this experiment: $\eta=0.003$.
- The planned 9-hour single-seed run and the remaining $\eta=0.01,0.03$ sweep
  were cancelled at the user's request.
- Complete physical acceptance of job `29708`'s best sample (score
  `71.73423878408627`) is complete. For this sample, `a=0.08` produced a good
  source-$\psi$ fit; the guarded outward search accepted through `s=0.30` and
  rejected adjacent `s=0.36`. The selected surface has
  $|V|=0.04491\,\mathrm{m}^3$, $\iota=2.4626$, and off-grid relative residual
  $4.63\times10^{-5}$. Poincare passed; DESC remained nested and reduced the
  normalized force error to mean/P95 $2.88\times10^{-3}/6.26\times10^{-3}$,
  although its optimizer hit the 50-iteration limit. Full evidence and all
  required figures are in `reports/qh_flow_standard_adam_acceptance_report.md`
  section 8 and `reports/assets/qh_flow_standard_adam_71p734_full_eval/`.
  Selected-surface SHA-256 is
  `8b0171a25de84532601bc02f10181a0381b3620bdeb9a6b624cfde2a82936c7c`.
- No optimizer or physical-evaluation Slurm job remains active.

### Slurm jobs, accepted 2026-07-31

- No optimizer job remains active.
- `29708`: COMPLETED, exit code 0, 1 h 55 min 23 s. Corrected-score
  CEM-latent Adam completed 273 iterations: initial `69.12277679724532`, final
  `71.72986622806994`, best `71.73423878408627` at iteration 271. All
  perturbation endpoints were valid. Best-case SHA-256:
  `92c8553821837e6c2723586f87ae7a04ef056cf0cdb39fd513c15f9b064a128c`.
- `29709`: FAILED, exit code 2, after 42 min 03 s. Only unscreened seed
  `2026073100` completed all 120 iterations, from `0.3681156045594607` to best
  `7.124938833298255`. Starting seed `2026073101` then failed at module-level
  `import torch` with oneMKL unable to load `libtorch_cpu.so`; the remaining
  seven predetermined seeds never ran. This is not an 0/8 or 1/8 success-rate
  result, and no random-basin probability may be inferred from it.
- Both jobs used corrected score library SHA-256
  `0b7342db471788385931385c25ded8095c72cfb7fcea1e21376a0475dafaa427`.
  Peak RSS was about 3.7 GiB, far below 128 GiB; both GPU postflight files show
  all four GPUs at 2 MiB and 0% utilization, with no optimizer/score workers
  left behind.
- `29726`: COMPLETED with exit code 0 in 1 minute 23 seconds. This CPU-only
  maintenance job re-rendered the most recent direct and DESC Boozer $|B|$
  figures as white-background colored contour lines without rerunning the
  surface or DESC solves. Direct data remained 0.611158548--0.731218674 T.

## 3. Durable User Requirements

- Prefer stable, bounded algorithms: dense linear least squares and fixed-cost
  GPU kernels are preferred over nonlinear solves with long-tailed iteration.
- The production path from coils through magnetic axis and fitted $\psi$ is the
  already validated stable implementation. Do not redesign or optimize it
  unless a required physical quantity is missing.
- The score is oriented so larger is better. A score near 100 should mean an
  exceptionally good and practically meaningful coil set; middle scores should
  be broadly ordered by quality.
- A useful score must reward reasonably large magnetic surfaces, saturate once
  size is sufficient, penalize low $|\iota|$ for QH, emphasize QS quality, and
  resist circular-coil, tiny-surface, low-valid-point, and similar shortcuts.
- For QH optimization, ordinary random starts must not be pre-screened when the
  experiment is intended to measure the probability of entering an optimizable
  basin. The current `8 x 120` experiment follows this rule.
- Numerical training and evaluation run on the new Slurm server, not the old
  server. Use submitted jobs, not heavy computation on the login node.
- Work only under `~/` remotely. Check that allocated GPUs are idle before a
  benchmark. Do not leave worker, background, or zombie processes after jobs.
- Current multi-GPU experiments may use four RTX 5090 GPUs; do not accidentally
  schedule overlapping four-GPU jobs for the same experiment.
- Flow decoding and $\alpha+\nu$ initialization should use FP32 where validated.
  FP64 is not the default merely because it is a physics calculation.
- Mathematical formulas in reports use `$...$` or `$$...$$`, not inline code.
- Reports must be readable, clearly separate verified results from hypotheses,
  and cite the relevant plots and raw summaries.
- A complete evaluation of a selected coil is not optional. It includes the
  largest reasonably feasible surface, white-background colored $|B|$ contour
  lines, full-device coils plus
  surface HTML, Poincare validation, DESC, all required DESC figures, and DESC
  quantities versus $\rho$.
- The source-$\psi$ fit radius `a` and candidate surface levels `s` are
  **sample-specific search results, not fixed workflow constants**. The
  `a=0.08` and current `s` values recorded for the 71.7342 candidate must not
  be reused blindly for a geometrically different coil set. Every new sample
  must re-evaluate the useful $\psi$ fitting domain and perform its own outward
  guarded-surface search.
- Before every remote connection attempt, read
  `REMOTE_CODEX_INSTRUCTIONS.md` and run its WSL/master-connection preflight in
  order. Do not guess a Windows SSH alias, initiate interactive authentication,
  or bypass a failed control-socket check.

## 4. Remote Compute Source of Truth

- Remote commands use the authenticated WSL master connection documented in
  `REMOTE_CODEX_INSTRUCTIONS.md` (`wsl.exe -d Ubuntu -- ssh ... ustc107`). The
  document, not ad hoc local SSH config, is authoritative.
  Never store or repeat passwords.
- Active remote worktree:
  `~/local_surface_evaluator_worktrees/qh-flow-zo-adam`.
- Shared base repository and large artifacts: `~/local_surface_evaluator`.
- QH flow data on the new server:
  `~/local_surface_evaluator_data/quasr_qh_flow_v1`.
- Python environment used by current flow/score jobs: `~/coil/.venv`.
- Preferred Slurm route:
  `competition / P107-RTX5090 / qos_p107-rtx5090`.
- Standard four-GPU request: 1 node, 1 task, 16 CPUs, four RTX 5090 GPUs,
  128 GiB memory.
- The old QUASR source was `/data/zhouyebi/QUASR_08072024/` on the old server.
  It was read-only and should no longer be used after the required subset was
  copied to the new server.
- Login-node work is limited to lightweight inspection, Git, Slurm submission,
  and small text processing. Compilation, model work, numerical evaluation,
  benchmarks, and plotting belong in Slurm jobs.

### Code and artifact roots must remain separate

- Code and the branch-specific native score build come from the active
  worktree.
- The trained flow checkpoint currently comes from the shared base repository.
- Do **not** derive both from one generic `asset_root`. This caused the
  2026-07-31 stale-score regression.
- Slurm scripts must use `SLURM_SUBMIT_DIR` for the project worktree. Under
  `sbatch`, `BASH_SOURCE` points into `/var/spool/slurmd` and cannot identify the
  repository.

## 5. Stable Pipeline and Physics Scope

The stable production evaluator is conceptually:

$$
\text{coils}
\rightarrow \text{magnetic axis}
\rightarrow s\text{ or }\psi\text{ fit}
\rightarrow \text{large feasible magnetic region}
\rightarrow \text{GPU volume-QS score}.
$$

The magnetic-axis and $\psi$ stages are established infrastructure. The current
research focus begins after $\psi$ or in optimization over coil parameters.

### $\alpha+\nu$ route

- Dense, approximately uniform volume samples and linear least squares fit the
  Clebsch/straight-field-line coordinate $\alpha$ from known $\boldsymbol B$ and
  calibrated $\psi$.
- The poloidal correction $\lambda$ and rotational transform $\iota$ follow
  from the fitted straight-field-line relation. A toroidal correction $\nu$
  is then fitted to approach Boozer coordinates more closely.
- This route produces a stable, approximately Boozer volume coordinate and a
  strong initial surface. It does not by itself prove an exact Boozer solution
  or an ideal-MHD DESC equilibrium.
- Production initialization uses dense FP32 GPU field evaluation and QR where
  validated. Small projection/reparameterization pieces and DESC may remain on
  CPU; do not claim a stage is GPU merely because the job requested a GPU.
- A candidate is accepted only after guarded physical residual and coordinate
  invertibility checks. $\alpha+\nu$ is an initializer, not permission to skip
  Boozer/Poincare validation.

### DESC status

- The earlier DESC branch established that approximate Boozer coordinates can
  be obtained before DESC, but DESC force solve can still diverge, produce a
  non-nested volume, or reduce force while flipping the coordinate Jacobian.
- A small optimizer objective is not sufficient. Check boundary mismatch,
  force quantiles, `is_nested`, and the sign of the relevant Jacobian.
- The DESC investigation is paused, not resolved. `CODEX_HANDOFF.md` describes
  an older 2026-07-10 stage and is historical, not current state.
- Detailed recap: `reports/project_progress_recap_20260726.md`.

## 6. Native GPU Score

- Production score evaluation is implemented in C++/CUDA and exposed to Python
  as a black box. Python should orchestrate; it must not reimplement the hot
  numerical path.
- The score searches for a reasonably large usable magnetic region and evaluates
  volume differential QS inside the magnetic volume, rather than in a fixed
  tiny cylinder.
- Important hardening already performed includes surface-size saturation,
  low-$\iota$ QH penalties, QH-vs-QA helicity competition, long-horizon surface
  checks, fixed valid sample budgets, physical volume weighting, and safeguards
  against low-valid-point score inflation.
- The full score is a screening/optimization objective. Final candidates still
  require the complete physical evaluation workflow.

### Critical current-reversal convention

For simultaneous reversal of all coil currents, the field-line geometry and
normalized QS quality should remain invariant. The old CUDA score used a
positive-only $G$ while the signed field quantity changed sign. Commit
`517a041` corrected this by assigning the sign of edge toroidal flux to $G$.

Current binary status:

- **Validated corrected binary for this branch:**
  `0b7342db471788385931385c25ded8095c72cfb7fcea1e21376a0475dafaa427`,
  located under the active worktree's
  `gpu_backend/build_native_score/libstellarator_gpu.so`.
- **Stale pre-fix binary; do not use:**
  `d2cfcab1923e0fd80a2ed5d31dbc8573a72a77e9bfb7cdd4d7e2847f4e18bdc9`,
  currently present in the shared base repository build directory.
- The Adam Slurm entrypoints now verify the corrected SHA before starting.
  A future intentional rebuild must first be validated, then update both the
  scripts and this file with the new SHA.

The score discrepancy for the same CEM latent is now understood:

| Evaluation | Score | Interpretation |
|---|---:|---|
| CEM artifact record, stale binary, 32 flow steps | 50.5862479 | Historical metadata only |
| Cancelled 2026-07-31 job, stale binary, 256-step RK4 | 53.6928721 | Invalid for current comparison |
| Old Adam, corrected binary, 256-step RK4 | 69.1136192 | Valid corrected baseline |
| Current job `29708`, corrected binary, 256-step RK4 | 69.1227768 | Valid corrected baseline; small runtime variation |

Therefore the approximately 69-point Adam start and the 50.586-point CEM
record are the **same latent**, not two different starting points. The large
difference is primarily the score bug fix; flow integration resolution accounts
for only the smaller 32-step versus 256-step difference.

Detailed evidence:

- `reports/qh_flow_landscape_report.md`
- `reports/qh_flow_prior_zo_adam_medium_report.md`
- `reports/assets/qh_flow_prior_cem_29129/best.json`
- `reports/assets/qh_flow_zo_adam_29465/summary.json`

## 7. QH Flow Model

- Target condition currently studied: QH, $N_{\mathrm{FP}}=4$, three base
  coils.
- One token is one coil: 99 Fourier geometry/current-related coefficients plus
  one current value, for 100 values per token.
- Architecture: non-causal Transformer with no RoPE, eight layers, width 512,
  eight attention heads, hidden width 1408, RMSNorm, PreNorm, and SwiGLU-style
  feed-forward blocks. $N_{\mathrm{FP}}$ is injected as a condition.
- The retained normalizer makes the initial noise scale tractable, while the
  training loss was changed to restore physical curve-distance importance so
  high-frequency standardized coordinates do not dominate merely because their
  raw variance is small.
- Current checkpoint:
  `~/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt`.
- Checkpoint state: EMA at step 30,000.
- Checkpoint SHA-256:
  `39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f`.
- Optimization decoding uses FP32 RK4 with 256 steps. The CEM artifact used 32
  flow steps; its recorded score must not be reused as a current baseline.
- Detailed training and first-generation history:
  `reports/qh_flow_matching_first_generation_report.md`.

## 8. Optimization History

### Flow-prior CEM

- Long CEM run artifact: `reports/assets/qh_flow_prior_cem_29129/best.json`.
- Its metadata records score 50.5862479 from the stale score binary. Under the
  corrected binary and 256-step RK4 the same latent is approximately 69.12.
- CEM established that optimizing the trained flow latent can outperform direct
  optimization in raw Fourier space, but a high screening score still requires
  complete physical validation.

### Old zeroth-order “Adam” run `29465`

- Start: same CEM latent, corrected re-score 69.1136192.
- Best after 80 iterations: 70.5777647.
- This was **not standard Adam**. It used $\beta_2=0.99$, score-difference
  clipping, prior penalties, update clipping, three proposal scales,
  accept/reject logic, backtracking, and a growing learning rate.
- It is useful historical evidence but not the baseline requested by the user.

### Current standard Adam definition

Implementation:

- `scripts/optimize_flow_prior_standard_adam.py`
- `scripts/slurm_flow_prior_standard_adam.sh`
- `scripts/slurm_flow_prior_standard_adam_multistart.sh`

For four fresh orthogonal directions $u_j$ per step:

$$
\hat g_t=\frac{1}{4}\sum_{j=1}^4
\frac{S(z_t+c u_j)-S(z_t-c u_j)}{2c}u_j,
\qquad c=0.01.
$$

Then score ascent uses standard Adam:

$$
m_t=0.9m_{t-1}+0.1\hat g_t,
$$

$$
v_t=0.999v_{t-1}+0.001\hat g_t^2,
$$

$$
z_{t+1}=z_t+\eta\frac{\hat m_t}{\sqrt{\hat v_t}+10^{-8}}.
$$

Current experiment uses $\eta=0.003$. There is no AdamW, weight decay, learning
rate schedule, prior penalty, score-difference clipping, update clipping,
parameter clipping, proposal search, backtracking, or accept/reject step.

Each iteration evaluates eight antithetic gradient endpoints and one updated
center. Four native score workers map this to two endpoint waves plus one center
wave on four GPUs. The flow decoder is FP32 RK4 with 256 steps.

### Fixed-seed short comparison

Both runs used random seed `2026073004` and the same direction sequence:

| $\eta$ | Steps | Initial | Best | Best step | Final | Wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 60 | 3.9198080 | 16.2093253 | 48 | 15.7478725 | 1397.9 s |
| 0.003 | 60 | 3.9198080 | 16.5386668 | 60 | 16.5386668 | 1325.6 s |

This particular random start was optimizable but weak. The user therefore
replaced the wider $\eta$ sweep with fewer, longer random-start trajectories.
The $\eta=0.01$ and 0.03 jobs were cancelled before starting.

## 9. Complete Physical Evaluation Contract

The sole orchestration entrypoint is documented in:

- `docs/精简线圈评估流程.md`
- `evaluation/full_physical/README.md`

Use `evaluation/full_physical/`; do not improvise a new evaluation script during
candidate acceptance. The workflow must:

1. run the current native score and preserve the complete metric/timing bundle;
2. expand outward and select a reasonably large guarded surface, not a fixed
   $a=0.05$ micro-tube and not necessarily the mathematically maximal surface;
   both the source fit radius $a$ and the tested $s$ ladder must be adapted to
   the current sample rather than copied from a previous evaluation;
3. run the dense FP32 GPU $\alpha+\nu$ initializer;
4. run guarded Boozer/Poincare validation and reject branch jumps or coordinate
   folding;
5. generate white-background colored $|B|$ contour lines, with line color
   encoding magnitude, and full-device coils plus surface HTML; do not use a
   heatmap or filled contours;
6. run DESC with the documented boundary and flux conventions;
7. include every required DESC figure, including boundary, Boozer modes,
   Boozer $|B|$, QH QS diagnostics, $\iota$, and the available quantities versus
   $\rho$;
8. run `validate_delivery.sh` before reporting completion.

Do not use a raw small-tube Boozer solve as a substitute for the $\alpha+\nu$
path. If fitted $\psi$ residual is small but Poincare or the surface is wildly
inconsistent, assume an implementation/evaluation-path error before claiming
new physics.

## 10. Known Errors That Must Not Recur

1. **Stale score binary:** never compare or optimize with `d2cfc...`; pin and
   record the validated branch-specific score library SHA.
2. **Recorded versus current score:** `50.5862` in the CEM artifact is stale
   metadata. The corrected baseline for the same latent is about 69.12.
3. **Shared root mistake:** checkpoint may come from the base repository, but
   branch code and score library come from the worktree.
4. **Slurm source path mistake:** `BASH_SOURCE` resolves to Slurm spool storage.
   Use `SLURM_SUBMIT_DIR` or explicit `PROJECT`.
5. **Fake standard Adam:** do not quietly add learning-rate growth, clipping,
   proposal selection, backtracking, trust regions, prior penalties, or AdamW
   when the requested baseline is standard Adam.
6. **Wrong starting-point experiment:** random-basin probability experiments
   must run every predetermined random start; no initial-score screening.
7. **GPU accounting mistake:** inspect only Slurm-allocated GPUs, verify they are
   idle before timing, and clean child score workers on every exit path.
8. **CPU fallback hidden as GPU work:** record backend and precision per stage,
   especially for $\alpha$ and $\nu$ field evaluation.
9. **Tiny fixed surface:** complete evaluation must search outward for a large
   feasible surface.
10. **Wrong Boozer route:** use $\alpha+\nu$ initialization before guarded
    surface refinement; a low $\psi$ residual cannot coexist with an arbitrary
    Poincare mismatch without an implementation problem.
11. **Incomplete deliverables:** do not omit DESC figures, $\rho$ profiles,
    Poincare, $|B|$, or the full-device HTML.
12. **Misleading DESC success:** optimizer success or low mean force does not
    override non-nestedness, Jacobian sign changes, boundary mismatch, or force
    singularities.
13. **Ad hoc remote connection:** read `REMOTE_CODEX_INSTRUCTIONS.md` first and
    use its WSL master-connection preflight. A failed control socket is a hard
    stop until the user rebuilds the authenticated master connection.
14. **Sequential multistart runtime:** do not launch many full
    PyTorch/CUDA/multiprocessing lifecycles in one `set -e` shell loop. Job
    `29709` lost seven seeds when the second interpreter failed during
    `import torch`. Use seed-isolated Slurm array tasks and an independent
    aggregation step; also explicitly close and join `NativeScorePool` queues.
15. **`sbatch --wrap` shell:** Slurm may execute a wrapped command with
    `/bin/sh`. Do not put Bash-only options such as `set -o pipefail` directly
    in the wrapper; submit a real Bash script or explicitly invoke Bash.

## 11. Important Files

- Current living memory: `MEMORY.md`.
- Historical DESC handoff: `CODEX_HANDOFF.md`.
- Stable/full progress recap: `reports/project_progress_recap_20260726.md`.
- GPU score design/results: `reports/gpu_native_volume_qs_score_report.md` and
  `reports/volume_qs_score_design.md`.
- Complete evaluation procedure: `docs/精简线圈评估流程.md` and
  `evaluation/full_physical/`.
- Flow training and first-generation evaluation:
  `reports/qh_flow_matching_first_generation_report.md`.
- Current-reversal and landscape audit: `reports/qh_flow_landscape_report.md`.
- CEM validation: `reports/qh_flow_prior_cem_validation.md`.
- Old hybrid Adam report: `reports/qh_flow_prior_zo_adam_medium_report.md`.
- Standard first-order feasibility discussion:
  `reports/qh_flow_prior_first_order_feasibility.md`.
- Standard-Adam job acceptance and multistart failure analysis:
  `reports/qh_flow_standard_adam_acceptance_report.md`.

## 12. Next Actions

1. Do not resubmit or supplement the failed random multistart experiment until
   the user explicitly requests it.
2. Before any future multistart run, isolate seeds as Slurm array tasks, record
   per-seed failures, add a separate aggregation step, and fix queue cleanup.
3. The `71.7342388` candidate's complete physical evaluation is finished and
   validated; do not rerun it unless a new diagnostic or changed algorithm
   requires it. Use section 8 of the acceptance report as the source of truth.
4. A future random-basin probability result must still include every
   predetermined unscreened seed, including failed trajectories; the one
   completed seed from `29709` is only partial evidence.

## 13. Dated Change Log

### 2026-07-31

- Accepted job `29708`: standard Adam from the corrected-score CEM latent rose
  from 69.1228 to best 71.7342 in 273 iterations without optimizer heuristics.
- Completed the same sample's fixed physical evaluation. This sample's
  adaptive search selected `a=0.08`, accepted through `s=0.30`, rejected
  adjacent `s=0.36`, passed Poincare, and produced a nested DESC result with
  final normalized force mean/P95 below $10^{-2}$. Added all required figures
  and raw artifacts to section 8 of the acceptance report; delivery validation
  passed. These `a/s` values are sample-specific, not workflow defaults.
- Marked job `29709` failed rather than complete. Only seed `2026073100`
  finished; the second Python runtime failed during `import torch` with an
  oneMKL `libtorch_cpu.so` load error, so no random-start success rate exists.
- Preserved all available artifacts and wrote
  `reports/qh_flow_standard_adam_acceptance_report.md`. Per user instruction,
  no rerun or additional physical evaluation was submitted.
- Standardized direct Boozer and DESC $|B|$ outputs as white-background colored
  contour lines; heatmaps and filled contours are no longer accepted for these
  report figures.
- Re-rendered the latest full-evaluation report assets with Slurm job `29726`.
  Direct PNG SHA-256 is `463adeae6983d12f9f9af8092b4a5f934a434e6a812957dbbf473bb0d1611495`;
  DESC PNG SHA-256 is `eb4ecce0eb3c119274688c29a322514c43a056236f975e59b27debcce56bda5d`.
  `validate_delivery.sh` passed and found all eight successful DESC PNGs cited.
- Made `REMOTE_CODEX_INSTRUCTIONS.md` a required pre-read before every remote
  connection attempt. Corrected its stale `ustc107-jump` alias to the current
  `ustc107`; the WSL master connection, `pb24511935` identity, and `tradmin-02`
  login node were then verified.
- Added clean random-start standard Adam with $\beta_1=0.9$, $\beta_2=0.999$,
  fixed $\eta$, and no optimizer heuristics.
- Completed controlled 60-step runs for $\eta=0.001$ and 0.003; cancelled the
  remaining learning rates after the user changed the experiment.
- Added CEM-latent initialization with zero Adam moments and an unscreened
  sequential multistart job.
- Detected that jobs `29702/29703` used stale base-repository score binary
  `d2cfc...`; cancelled both after about six minutes.
- Corrected code/artifact root handling, pinned score binary `0b7342...`, and
  submitted replacement jobs `29708/29709`.
- Established that the 50.586 and approximately 69 scores refer to the same CEM
  latent under different score binaries/resolutions, not different starts.
- Created `MEMORY.md` and root `AGENTS.md` as the persistent cross-session memory
  mechanism.
