# Local Surface Evaluator Project Memory

> **Living source of truth. Last updated: 2026-08-10 (Asia/Shanghai).**
>
> This file was compacted on 2026-08-08. The exact pre-compaction memory is
> preserved as `MEMORY_archive_20260808.md` (2,884 lines, 198,359 bytes,
> SHA-256 `c16776912751d7ecdb1c0b139d83bfc6aa9fd9c0d2f196cb9ec27dce3b695b6d`).
> The archive is immutable historical evidence; this file is authoritative for
> current work. Consult the archive only when this file or a report points to it.
> The Git snapshot immediately before compaction is commit `7898bc7`.

## 1. Maintenance Rules

- Read this file at the start of every conversation and immediately after any
  context compaction or handoff.
- Update it in the same turn when an important branch, job, validated hash,
  interface, numerical conclusion, user requirement, bug, or invalid result
  changes. Date every update and make supersession explicit.
- Keep this file under a soft limit of 500 lines. Active jobs may temporarily
  carry enough detail to resume safely; after acceptance, reduce each to one
  outcome and a report/artifact pointer.
- Do not duplicate report tables, routine scheduler history, failed setup
  attempts, or step-by-step debugging here. Put them in reports or the dated
  archive. A historical detail belongs here only when forgetting it could make
  a future run incorrect.
- The newest dated correction in this file wins over older reports and the
  archive. Preserve old artifacts, but label results produced by obsolete code
  or score definitions as invalid or historical.
- Never store passwords, authentication tokens, private keys, one-time codes,
  or credential-bearing URLs in repository files.

## 2. Current State

### Repository and publication

- Active local branch: `codex/score-eval-compression`, created on 2026-08-10
  directly from production `main` commit `81a06e3` in the repository root
  worktree. Do not create a secondary worktree for this experiment. The
  compact memory/archive history was carried forward from the validated
  compaction lineage; the prior reduced-latent branch is archived at
  `a95a8d9`.
- Clean integration worktree: `.worktrees/main-integration`; local `main` was
  `81a06e3` when this file was compacted. Do not assume the dirty root worktree
  can be used for integration without inspecting it first.
- The private research repository contains personal/infrastructure paths and
  must not be published directly. The sanitized public project is maintained
  separately at `../opensource_staging` and published as
  `https://github.com/fancyovo/StellCoilOpt` under the MIT license. Public commit
  `4062b163cea12db636e2ccdc37b1da5d650f2ea5` passed 63 local tests and a remote
  RTX 5090 CUDA 13 build/example smoke on 2026-08-08.

### Active score-evaluation compression experiment

- 2026-08-10: branch `codex/score-eval-compression` profiles only ABI-10 calls
  with a supplied axis hint and strict branch continuation. It does not time or
  alter standalone global-axis search. Exact loop-invariant removals are active;
  the fixed-matrix QR and psi-grid experiments did not change production
  defaults.
- The established 69-case strict-continuation holdout remains the production
  timing reference at P50/P95 `0.990/1.254 s`. A new 8-case high-score profile
  measured `1.323 s`, but it does not supersede that reference: fixed-size
  FP32 QR was anomalously about `0.44 s` versus historical raw records near
  `0.23 s`. Source/defaults/CUDA target match; node/build/NVTX effects remain
  unisolated and require a same-input, same-idle-GPU A/B before any absolute
  performance conclusion.
- Two exact loop-invariant removals are accepted: cache toroidal trigonometry
  per psi-evaluation point and cache axis interpolation per phi grid point.
  Within the new run, strict-hint P50/P95/max improved from
  `1.323/1.463/1.496 s` to `1.117/1.262/1.264 s`. Maximum score and component
  differences were `1.42e-14` and `2.84e-14`; no status changed. The `1.117 s`
  result is an internal paired benchmark, not a replacement production median.
- The steady representative precision split is CPU FP64 `107 ms`, GPU FP64
  `190 ms`, GPU mixed FP32-field/FP64-state `220 ms`, and GPU FP32-dominant
  `542 ms`. Inside the `444 ms` full-GPU psi fit, design-matrix assembly is
  only about `13 ms`; the FP32 QR branch is `414 ms` and is the actual
  bottleneck. Its algorithmic throughput in this anomalously slow run is about
  `4.68 TFLOP/s`. NCU counters are unavailable on the cluster
  (`ERR_NVGPUCTRPERM`), so no counter-derived trace TFLOPS is claimed.
- 2026-08-10 psi-grid acceptance used all 69 historical strict-continuation
  `legacy-ok` holdout cases, five grids, two repeats, and fixed 4000-point
  independent physical validation. Grid 48 reduced physical fit rows from
  389440 to 82176, psi-fit P50 from `0.4402` to `0.0983 s`, and score-call
  P50/P95 from `1.013/1.266` to `0.665/0.913 s`. All 138 calls were `ok`;
  independent angle-P95 median/P95 ratios were `0.9991/1.0137`, score Spearman
  was `0.999927`, and top-decile overlap was 100%. Grid 48 is the next
  optimization-neighborhood candidate, not yet the production default.
- 2026-08-10 fixed-matrix result: the exact augmented FP32 problem is
  `391014 x 1574`, with 389440 physical rows and 1574 ridge rows. The frozen
  QUASR case-1739363 snapshot is 2463400872 bytes with SHA-256
  `e8878c17a3d6b7c64f5459391c8d98cbc9eccb9b52410acc5c7d2ca90f3dd6b2`;
  it remains under `~/local_surface_evaluator_data/qr_bench_20260810/`.
- Standalone single-RTX5090 cuSOLVER Householder least squares is stable at
  P50 `181.933 ms` (`10.635` Householder-equivalent TFLOP/s). LDA-256 padding
  is exactly equivalent at `179.639 ms`, only `1.26%` faster. Generic API,
  nondeterministic mode, BF16x9 mode, stable TSQR/block-GS, and mixed-precision
  iterative refinement do not improve latency. MAGMA 2.10 Householder is
  accurate but about 14.5x slower (`2.632 s`) because of its hybrid panel path.
- The fixed-matrix field named `physical_residual_relative` is actually the
  unnormalized training-equation residual
  `||A_data x-b_data||/||b_data||`; it is not the independent normalized
  `|B.grad(psi)|/(|B||grad(psi)|)` angle. Same-matrix solver screening uses the
  former plus coefficient/normal residuals; production acceptance uses the
  latter, downstream score, status, and ranking.
- Fusing the RHS as the last column of `[A|b]` removes the standalone `Sormqr`
  pass: single-RTX5090 P50/P95 became `162.428/162.492 ms` (1.120x), coefficient
  relative difference `8.91e-6`, and training-equation residual ratio
  `0.999999`. This is a useful exact-form candidate but is not integrated into
  production after only one frozen-matrix test.
- Shifted Gram and short PCGLS reach actual estimated `35--45 TFLOP/s`, proving
  the GEMM throughput is available, but their training-equation residuals are
  6--24x the Householder baseline and their coefficient errors are order one. FP32
  unshifted Gram/CholeskyQR2 loses positive definiteness at pivot 579; TF32
  fails earlier. LSQR does not converge before losing its speed advantage.
- No tested method is both at least 1.5x faster and reference-accurate, so no
  alternative was connected to the score path and no misleading end-to-end
  timing was run. Full methods, raw JSON, figures, and acceptance definitions
  are in sections 12--13 of `reports/qh_score_evaluation_compression_20260810.md`
  and `reports/assets/qh_psi_qr_benchmark_20260810/`. The fixed-matrix
  implementation is complete through `31d17f1`; the accepted report/result
  snapshot is commit `30f79a4`.

### Completed 10000-step optimization continuation

- P107 job `33799` completed the exact 5000-to-10000 continuation with
  `status=ok`, `stop_reason=completed_iterations`, 10000 contiguous history
  rows, 9822 cumulative Adam steps, zero-byte stderr, and idle pre/postflight
  RTX 5090 states. The state restored current/best latents, both Adam moments,
  Adam step, both RNGs, and flow prefetch state; it was not a restart from best.
- The additional 5000 steps produced no new best. Their maximum current score
  was `93.3271797` at iteration 5064, below the unchanged `93.3672653` best at
  iteration 4341. The best remained unchanged for 5659 consecutive iterations;
  final current score was `92.3147119`. Under the fixed LR `0.01`, beta
  `(0.7,0.999)`, perturbation `0.005`, two-direction central difference, and
  FP32 RK4-128 configuration, mechanically adding more steps is no longer the
  default next action. This is evidence of optimizer stagnation, not proof of
  global optimality.
- Added wall time was `30648.96 s` (8 h 30 min 49 s), or `6.130 s/step`
  including periodic serialization/plotting. Iteration compute averaged
  `5.299 s` with P95 `5.530 s` and max `6.914 s`; score, flow, and iteration-
  external artifact work used 68.0%, 18.4%, and 13.6% of added wall time.
  There were 4903 applied updates, 97 safe skips, 12 temporal rejections, 28
  backtracked centers, and no non-`ok` final center.
- The frozen best SHA-256 remains
  `d4517e03d66913d958bfac88b42b7d56228a9717c4b445f5ac28f242b049cc29`,
  exactly the already fully evaluated iteration-4341 sample. No duplicate full
  physical evaluation is required. Evidence is section 22 of
  `reports/qh_score_throughput_and_continuous_surface_plan.md` and
  `reports/assets/qh_score_fast_beta1_0p7_continue10000_20260808/`.

### Current accepted best

- Highest fully evaluated native-score sample: QH, $N_{\mathrm{FP}}=4$, three
  base coils, score `93.3672653337` at iteration 4341. Input SHA-256:
  `d4517e03d66913d958bfac88b42b7d56228a9717c4b445f5ac28f242b049cc29`.
- Its sample-adaptive complete evaluation selected `a=0.08`; standard
  LS/Newton accepted `s=0.24/0.36/0.49` with increasing enclosed volume and
  selected `s=0.49`, volume `0.0658787 m^3`. `s=0.64` was the tested outer
  failure because GPU-ray supplied `174967 < 180000` required points. These
  `a/s` values belong only to this sample.
- Selected-surface diagnostics: iota `1.53758`, dense relative L2
  `8.7371e-6`, normal-field P95 `1.1583e-5`, and face QA/QH/QP errors
  `4.9096e-3 / 2.9965e-6 / 4.9715e-3`. Poincare and direct/DESC colored
  contours show a nested QH configuration.
- CPU-P107 DESC preserved nesting and reduced normalized force mean/P95/max
  from `1.2562 / 1.7427 / 2.7540` to
  `7.6700e-4 / 1.7054e-3 / 4.8221e-3`. It reached the 50-step cap, so physical
  acceptance passed but strict solver convergence is false.
- Evidence: `reports/assets/qh_score_fast_beta1_0p7_best933673_full_eval_20260808/`
  and sections 20--22 of
  `reports/qh_score_throughput_and_continuous_surface_plan.md`.

## 3. Production Mainline

The maintained workflow is:

$$
\text{coils}
\rightarrow \boldsymbol B
\rightarrow \text{magnetic axis}
\rightarrow s
\rightarrow \psi(s)
\rightarrow (\alpha,\iota)
\begin{cases}
\rightarrow \text{volume QS}\rightarrow \text{native score},\\
\rightarrow \nu\rightarrow \text{Simsopt LS/Newton}\rightarrow \text{DESC}.
\end{cases}
$$

### Stable front end

- **Magnetic axis:** batch-search elliptic fixed points of a one-period
  Poincare map, then use bounded Newton refinement and axis tracing. During
  local optimization, strict continuation from the previously validated axis
  is allowed and expected; standalone/corpus scoring must remain
  history-independent and use global search.
- **Geometric label $s$:** fit $\boldsymbol B\cdot\nabla s=0$ near the axis
  with a complete two-dimensional polynomial basis combined with toroidal
  Fourier modes. $s$ is a geometric surface label, not physical magnetic flux.
- **Flux calibration:** compute toroidal flux on multiple sections and set
  $\psi(s)=\Phi_t(s)/(2\pi)$. The coils-to-axis-to-$s/\psi$ route is validated
  infrastructure; do not redesign it unless a required physical quantity is
  genuinely unavailable.
- **Straight-field-line coordinate:** on dense, approximately uniform volume
  samples, solve one linear least-squares system for the Zernike-Fourier
  expansion of $\alpha$ and iota using
  $\boldsymbol B\cdot\nabla\alpha=0$. This yields a volume straight-field-line
  coordinate, not merely one fitted surface.

### Fast native score branch

- The production evaluator is C++/CUDA ABI 10. Python is an orchestration and
  ctypes layer only; it must not reimplement or silently move the hot numerical
  chain to CPU.
- From calibrated $\psi$, fitted $\alpha/\iota$, $\boldsymbol B$, and
  $\nabla\boldsymbol B$, compute differential volume QA/QH/QP on a fixed
  100000-point physical-volume sample. This branch does not need $\nu$.
- Score range is 0--100 and larger is better. Components are `axis`, `psi`,
  `surface`, `coordinate`, `volume_qs`, `iota`, and `coil`, with nominal
  weights `(10,10,10,10,42,10,8)`. QH iota and helicity-advantage gates are
  multiplicative, so the total cannot always be reconstructed by a simple
  weighted component sum.
- The score rewards a reasonably large magnetic region but saturates after
  useful size, strongly weights QH quality, penalizes low $|\iota|$, and blocks
  circular-coil, tiny-surface, wrong-helicity, and low-valid-point shortcuts.
- Current branch-specific score library SHA-256:
  `387495353bd4c8a3c2984fcfdb6625937da47da0efa2e578610d666c5a8a2f52`.
  Production launchers must verify this hash. An intentional rebuild requires
  fresh numerical validation and an update here before use.
- Current score conventions include
  $G=\mu_0 I_{\mathrm{link}}/(2\pi)$ with the sign of the linked current,
  cylindrical physical-volume weights, fixed point count, strict elliptic-axis
  existence $|\operatorname{tr}J|/\sqrt{\det J}<2$, continuous surface
  confidence, low-iota and helicity gates, and surface-size saturation.
- The score is a bounded screening/optimization objective, not proof of a
  Boozer surface or MHD equilibrium. Every selected candidate still requires
  the complete physical evaluation contract below.

### Full physical evaluation branch

- Fit toroidal correction $\nu$ after $\alpha/\iota$ to construct a near-Boozer
  volume coordinate and strong surface initial guesses. Dense field evaluation,
  sampling, and QR use validated FP32 GPU paths; $\alpha+\nu$ is an initializer,
  not an exact-surface certificate.
- Search outward using sample-specific source-fit radii $a$ and surface levels
  $s$. Select the largest reasonably feasible tested surface, with at least one
  nearby outer failure when practical. Never reuse another sample's $a/s$.
- Standard Simsopt LS/Newton plus independent dense residual, normal-field,
  regularity, nesting, and branch-continuity checks decide whether a magnetic
  surface exists. The conservative `guarded` path is diagnostic only and may
  reject a surface that standard LS/Newton accepts.
- DESC refines the accepted boundary. It may run on explicit CPU-P107; report
  force quantiles, boundary mismatch, nesting, Jacobian sign, and optimizer
  exit reason. A low mean force or `optimizer_success=true` cannot override
  non-nestedness, folding, or singular residuals.
- The earlier DESC exploration remains paused: near-Boozer initialization is
  now robust, but DESC convergence is not guaranteed. See
  `reports/project_progress_recap_20260726.md` for the detailed history.

## 4. Current Flow and Optimizer

### Flow model

- Training data: all extracted QUASR QH groups, 170755 samples total and
  153747 training samples, covering $N_{\mathrm{FP}}=2\ldots8$ and one through
  five base coils. Training sampled the empirical joint `(nfp,n_coils)`
  distribution, not independent uniform conditions.
- One token is one base coil: 99 Fourier geometry coefficients plus one current.
  Architecture: eight-layer non-causal Transformer, no RoPE, width 512, eight
  heads, FFN width 1408, PreNorm, RMSNorm, and SwiGLU; $N_{\mathrm{FP}}$ enters
  through a condition embedding.
- Per-coordinate normalization stabilizes the input distribution. The loss
  restores raw Fourier/Parseval curve-distance importance so standardized
  high-frequency tails do not dominate merely because their variance is small.
- Checkpoint:
  `~/local_surface_evaluator/runs/qh_flow_physical_lr_longselect_20260729/lr_3em4/checkpoint_latest.pt`,
  EMA step 30000, SHA-256
  `39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f`.
- Flow matching is retained as an invertible reparameterization that produces
  wider/smoother useful search directions. It is not accepted as a direct
  high-quality generator. Formal inversion/landscape checks may use RK4-256;
  current optimization uses self-consistent FP32 RK4-128. Never resume a saved
  optimizer state with a different flow discretization.

### Default optimizer

- Entry points: `scripts/optimize_flow_prior_standard_adam.py` and
  `scripts/slurm_flow_prior_standard_adam.sh`.
- Current default is score-only zeroth-order Adam in flow latent space: two
  fresh orthogonal directions, four centered score endpoints, perturbation
  `0.005`, LR `0.01`, beta `(0.7,0.999)`, FP32 RK4-128, continuous score, and
  strict axis continuation. No flow VJP, native-score gradient, G1--G4 path,
  or black-box-gradient experiment may leak into this production route.
- Cross-iteration pipelining decodes the accepted center together with the next
  endpoints. On the validated two-GPU setup, 600-step jobs averaged
  `5.27--5.40 s/step`; native score consumed about 75--77% of wall time and
  flow about 21--22%, without an extreme latency tail.
- Any non-`ok` directional endpoint skips the complete gradient/moment/parameter
  step. Valid but exceptional direction deltas are handled with scale-adaptive
  median/MAD filtering. An invalid proposed center triggers bounded feasibility
  backtracking and full rollback. Preserve the running best independently of
  the current state.
- Exact resume must restore current and best latents, both moments, Adam step,
  direction and flow RNGs, and prefetched endpoint state. Starting from a
  `best.json` with zero moments is a new staged optimization, not continuation.

## 5. Complete Evaluation Contract

Use only `evaluation/full_physical/`, following
`docs/精简线圈评估流程.md` and `evaluation/full_physical/README.md`. Do not
assemble a new acceptance script during evaluation.

Every complete evaluation must:

1. Re-run the current native score and preserve total score, all seven
   components, complete diagnostics, timing, code commit, score-library hash,
   checkpoint hash, and input identity.
2. Search sample-specific source-$\psi$ radii and surface levels outward; do not
   default to an `a=0.05` micro-tube or claim strict maximality.
3. Use GPU-ray sampling and FP32 GPU $\alpha+\nu$ initialization. Do not
   silently fall back to legacy Cartesian/CPU preprocessing; if no bounded GPU
   route exists, stop and ask the user whether to accept the slower method.
4. Run standard Simsopt LS/Newton and independent dense validation. Treat
   `guarded` rejection, initial $\psi$ distance, and displacement only as
   branch diagnostics, not final nonexistence proof.
5. Run batched GPU Poincare tracing and report per-section hits and nesting.
6. Report the selected surface's face QA/QH/QP error explicitly.
7. Generate white-background colored $|B|$ contour lines, with color encoding
   magnitude. Heatmaps and filled contours are not accepted replacements.
8. Generate complete-device coil plus large-surface PNG/HTML. Do not connect
   each field period's surface seam to itself; periodic copies must join in the
   physically correct order.
9. Run DESC with explicit backend and include every generated DESC figure,
   including boundary, Boozer modes, Boozer $|B|$, QH diagnostics, iota, and
   all available quantities versus $\rho$.
10. Run `evaluation/full_physical/validate_delivery.sh` and cite every required
    artifact in the report before declaring completion.

If $\psi$ residual is small but Poincare or the reconstructed surface is wildly
inconsistent, first diagnose code, coordinate, branch, or evaluation-path
errors. That combination is not credible evidence of new physics by itself.

## 6. Remote Compute Rules

- Before every remote connection, read `REMOTE_CODEX_INSTRUCTIONS.md` and run
  its WSL/master-connection preflight in order. Current alias is `ustc107`,
  authenticated through the existing WSL `Ubuntu` master connection. Never
  initiate interactive authentication or guess an alias.
- All remote project code, data, logs, and artifacts must remain under `~/`.
  Heavy computation, builds, tests, plots, and benchmarks run as Slurm jobs,
  never on the login node.
- Current preferred route is
  `competition / P107-RTX5090 / qos_p107-rtx5090`; up to four RTX 5090 GPUs and
  16 CPUs are available under its documented limits. Use only resources needed
  by the task and do not overlap jobs writing the same run directory.
- Before timing, verify every allocated GPU is idle. After every job, verify no
  score worker, child process, GPU allocation, or zombie process remains.
  `sbatch --test-only` validates resource syntax but its start-time estimate is
  unreliable.
- Native coils-to-score and parallel full-evaluation preprocessing must remain
  C++/CUDA or validated GPU code. Do not independently choose a CPU fallback
  that is one or two orders of magnitude slower. DESC is the explicit exception
  and may use `DESC_BACKEND=cpu-p107`.
- Keep branch code/builds in the active worktree, large shared artifacts in
  `~/local_surface_evaluator`, QH data in
  `~/local_surface_evaluator_data/quasr_qh_flow_v1`, and use `~/coil/.venv`
  where the current flow/score launchers require it. Do not collapse these into
  a generic asset root; that previously selected a stale score library.
- Under Slurm, use `SLURM_SUBMIT_DIR` or an explicit `PROJECT`; `BASH_SOURCE`
  resolves into `/var/spool/slurmd`. Use real Bash scripts for Bash-only shell
  behavior rather than relying on `sbatch --wrap` and `/bin/sh`.
- The old server and `/data/zhouyebi/QUASR_08072024/` are historical read-only
  sources and should not be used for current numerical work.
- Routine latent-score corpus collection was permanently stopped on 2026-08-02
  after the proxy experiments and a score-definition correction. Do not launch
  or report background collectors unless the user explicitly reopens that task.
  Unrelated Student-partition jobs are outside this project and must be ignored.

## 7. Errors and Invalid Results That Must Not Recur

- **Score/version mixing:** ABI-9 results before current-sign, $G$-scale,
  physical-volume-weight, fixed-point-budget, topology, or continuous-surface
  corrections are historical and cannot be compared numerically with current
  score. Always pin code, library, flow checkpoint, and decode resolution.
- **Current reversal:** simultaneous reversal of all coil currents must leave
  field-line geometry and normalized QS invariant. The signed linked-current
  convention for $G$ is mandatory.
- **Elliptic-axis threshold:** existence uses strict normalized trace `<2`.
  The topology margin affects preference/quality only; using it as an existence
  threshold created false `no_axis` results and dirty optimizer gradients.
- **Drift status:** `drift_rejected` means the bounded quick surface screen
  failed; it does not mean no magnetic axis and does not prove LS/Newton cannot
  find a surface. Do not add slow all-candidate or very-long-trace fallbacks to
  production without measured need and unchanged bounded latency.
- **History leakage:** optimizer continuation may use the previous axis as a
  strict hint, but standalone score/corpus evaluation must remain independent.
- **Surface constants:** `a` and `s` are sample-specific. Historical values such
  as `a=0.08`, `s=0.49`, or `s=0.30` are results, never global defaults.
- **Incomplete acceptance:** high score, low $\psi$ residual, a guarded solve,
  or reduced DESC force alone is insufficient. Run the full contract.
- **Hidden CPU fallback:** do not label a stage GPU merely because the Slurm
  job requested a GPU. Record backend and precision for $\psi$, $\alpha$, and
  $\nu$; legacy Cartesian alpha preprocessing is prohibited by default.
- **Wrong optimizer identity:** the current method is score finite differences
  plus Adam. Old hybrid “Adam”, CEM, G2/G3/G4, BFGS, proxy, and trust-region
  experiments must not be silently mixed into it.
- **Resume from best:** resuming only the best latent discards moments, current
  state, RNG, and pipeline cache. It is a restart and must be labeled as such.
- **Score artifacts:** distinguish an old artifact's recorded score, a new
  decode of the same latent, and a score from the current validated library.
- **Remote hygiene:** do not use stale shared builds, guessed paths, busy GPUs,
  login-node compute, sequential candidate evaluation as a “conservative”
  default, or unjoined multiprocessing workers.

## 8. Historical Milestones

These are intentionally summaries. Detailed job IDs, parameter sweeps, failed
attempts, and numerical tables remain in `MEMORY_archive_20260808.md` and the
linked reports.

- **DESC initial-guess exploration:** dense linear LS for $\psi$, then
  $\alpha/\iota$ and $\nu$, produced useful near-Boozer volume coordinates and
  robust surface initial guesses. It did not make DESC convergence automatic.
  See `reports/project_progress_recap_20260726.md`.
- **Native GPU score:** the coils-to-volume-QS path was moved to bounded
  C++/CUDA ABI 9 and corrected for current sign, $G$ scale, volume weights,
  valid-point count, low-iota cheating, and helicity competition. See
  `reports/gpu_native_volume_qs_score_report.md` and
  `reports/qh_differential_qs_metric_investigation.md`.
- **Flow model:** direct generation remained poor, but high-accuracy inversion
  and landscape tests showed that the learned latent coordinates broaden useful
  search basins. See `reports/qh_flow_matching_first_generation_report.md` and
  `reports/qh_flow_landscape_report.md`.
- **CEM and finite-difference Adam:** latent-space CEM produced the first clear
  optimization breakthrough; standard finite-difference Adam then surpassed it
  and became the maintained optimizer. See
  `reports/qh_flow_standard_adam_acceptance_report.md` and
  `reports/qh_score_throughput_and_continuous_surface_plan.md`.
- **Proxy experiments:** inverse-latent classification was possible, but neither
  classifier nor score-regression proxy reliably ranked physical score; routine
  corpus collection was stopped. See
  `reports/qh_latent_proxy_active_optimization_report.md` and
  `reports/qh_latent_score_regression_proxy_report.md`.
- **Analytic/reference-gradient exploration:** frozen-front G2/G3 directions
  were internally correct but systematically biased for the full score because
  geometry, surface, and branch responses were omitted. G4/G5 were judged too
  difficult relative to benefit; production returned to score-only finite
  differences. See `reports/qh_blackbox_gradient_exploration_report.md`.
- **Throughput/continuity work:** continuous surface confidence, strict axis
  continuation, flow pipelining, central two-direction differences, and robust
  update guards reduced the standard optimizer to roughly `5.3 s/step` on two
  RTX 5090 GPUs while preserving high-score ranking and physical acceptance.
- **Long-run result:** the same $N_{\mathrm{FP}}=4$, three-coil trajectory rose
  from score `92.3826` at 600 steps to `93.0409` at 2000 and `93.3673` at 4341;
  the best passed complete physical evaluation. Exact continuation to 10000
  produced no further best update, providing strong evidence that the current
  fixed optimizer configuration had exhausted this trajectory.
- **Reduced-latent flow:** exact-zero source tails at $k=16\ldots80$ failed
  geometry reconstruction (best median relative curve RMS `47.24%`); manifold
  flow was then rejected. Do not use those checkpoints. See
  `reports/qh_reduced_latent_flow_plan.md` for the complete experiment.

## 9. Important Files

- Current source of truth: `MEMORY.md`.
- Exact full history before 2026-08-08 compaction:
  `MEMORY_archive_20260808.md`.
- Archived early DESC handoff: `CODEX_HANDOFF.md`.
- Current user-facing overview and commands: `README.md`.
- Remote access and Slurm rules: `REMOTE_CODEX_INSTRUCTIONS.md`.
- Current methodology: `docs/QH原生评分与潜空间优化方法.md`.
- Fixed complete-evaluation procedure: `docs/精简线圈评估流程.md` and
  `evaluation/full_physical/README.md`.
- Current optimization and complete-evaluation record:
  `reports/qh_score_throughput_and_continuous_surface_plan.md`.
- Score design: `reports/gpu_native_volume_qs_score_report.md` and
  `reports/volume_qs_score_design.md`.
- Flow training: `reports/qh_flow_matching_first_generation_report.md`.
- Corrected landscape: `reports/qh_flow_landscape_report.md`.
- Differential-QS audit: `reports/qh_differential_qs_metric_investigation.md`.
- Abandoned gradient direction: `reports/qh_blackbox_gradient_exploration_report.md`.

## 10. Next Actions

1. Validate grid 48 on saved Adam trajectory neighborhoods and gate-boundary
   perturbations, focusing on local ranking, score smoothness, and accept/reject
   decisions before changing the production default.
2. If another small gain matters after grid-48 acceptance, connect augmented-RHS
   Householder QR behind an explicit mode and run the same end-to-end physical
   and ranking checks. Never substitute the rejected FP32 normal-equation path.
3. Do not restart manifold-flow, score collection, proxy, black-box-gradient,
   or paused DESC-method work without a new explicit user request.
