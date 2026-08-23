# Local Surface Evaluator Project Memory

> **Living source of truth. Last updated: 2026-08-24 (Asia/Shanghai).**
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

- 2026-08-10 production `main` fast-forwarded through score-compression commit
  `f9b5ebc`; current documented content commit is `e6fc9a2`. After integration,
  the temporary `.worktrees/main-integration` worktree was retired and `main`
  was checked out directly in the repository root. The completed feature branch
  remains at `f9b5ebc` only as a historical pointer.
- The private research repository contains personal/infrastructure paths and
  must not be published directly. The sanitized public project is maintained
  separately at `../opensource_staging` and published as
  `https://github.com/fancyovo/StellCoilOpt` under the MIT license. Public
  `main` commit `9de21bc4849252915b53825e64e9788bf8d65664` was pushed on
  2026-08-10 with the current score defaults and synchronized README/method
  docs. It passed 66 local tests and remote RTX 5090 CUDA 13/sm120 job `35941`:
  defaults read back as `48/48/48` and `iota_degree=3`; the packaged example
  returned `ok`, score `86.319425035571`, stage 8, and no residual GPU process.

### Production score-evaluation defaults

- 2026-08-10: the former `codex/score-eval-compression` work is merged into
  production `main`. Exact loop-invariant removals, psi grid 48, cubic iota, and
  strict-hint mode 2 are production defaults. Fixed-matrix QR alternatives
  remain experimental; standalone global-axis search remains history-independent.
- The established 69-case strict-continuation grid-80 timing reference was
  P50/P95 `0.990/1.254 s`. A later 8-case run was anomalously slower, so its
  absolute times are not a new baseline. Within that run, exact caching of
  toroidal trigonometry and axis interpolation improved P50 from `1.323` to
  `1.117 s` with score differences below `3e-14`. Its FP32 QR, not matrix
  assembly, was the bottleneck; NCU counters were unavailable. See report
  sections 5--11 rather than reusing these node-sensitive raw timings.
- 2026-08-10 psi-grid acceptance used all 69 historical strict-continuation
  `legacy-ok` holdout cases, five grids, two repeats, and fixed 4000-point
  independent physical validation. Grid 48 reduced physical fit rows from
  389440 to 82176, psi-fit P50 from `0.4402` to `0.0983 s`, and score-call
  P50/P95 from `1.013/1.266` to `0.665/0.913 s`. All 138 calls were `ok`;
  independent angle-P95 median/P95 ratios were `0.9991/1.0137`, score Spearman
  was `0.999927`, and top-decile overlap was 100%. On 2026-08-10 the user
  accepted grid 48 as production default in core ABI and both direct CLIs.
  CUDA13 smoke job `35813` built commit `34cdf2a`, read back `48/48/48`, and
  scored one no-override case `ok` with no zombies. Do not extrapolate below 48.
- 2026-08-10 strict-hint mode 2 deletes the five-line FP64 replay but retains
  four mixed-precision topology traces and all branch/residual rejection gates.
  Across 69 cases x two repeats, exact-hint P50/P95 fell from `660/905` to
  `551/730 ms` (1.198x); all calls stayed `ok`, rank and downstream physics were
  unchanged. A `1e-3` hint offset gave 1.107x because Newton then dominates.
  Optimizers default to mode 2; mode 1 retains formal FP64 verification.
  Evidence is section 15 of `reports/qh_score_evaluation_compression_20260810.md`.
- 2026-08-10 corrected the production joint fit from constant iota to
  `iota(u)=c0+c1*u+c2*u^2+c3*u^3`, where `u=rho^2=psi/psi_edge`. Across 69
  cases, all 138+138 calls were `ok`, every cubic fit reduced the joint
  alpha/iota residual (P50 ratio `0.9452`), added no measurable time, and kept
  the top decile exactly. A 200-step rerun from historical `start_10` rose from
  `85.5157` to `91.5263` at `4.117 s/step`; all updates were valid and step 200
  was best. Implementation commit `c8b185f`; accepted evidence commit `233f57e`.
  Evidence is section 16 and assets
  `qh_iota_degree_calibration_35856` / `qh_iota_cubic_adam200_35864`.
- 2026-08-10 matched 1000-step P107 jobs `35902/35903` completed cleanly and
  were identical through step 300. Two fresh random directions reached best
  `92.6818` at step 994 with 1000 accepted updates. Reusing the previous update
  plus one random direction reached only `92.5087` and then suffered 77 repeated
  temporal-gradient rejections at steps 924--1000. Keep fresh random directions
  as default; full evidence is section 17 of the current compression report.
- 2026-08-10 fixed-matrix screening used the exact augmented FP32 problem
  `391014 x 1574`. The best accurate single-RTX5090 result fused RHS into
  Householder QR at P50/P95 `162.428/162.492 ms` (1.120x over the baseline),
  but no tested QR, TSQR, Gram, PCGLS, LSQR, BF16, or refinement method was both
  at least 1.5x faster and reference-accurate. Nothing replaced production QR.
  `physical_residual_relative` in those artifacts is a training-equation
  residual, not the independent normalized physical angle. Frozen inputs,
  exact solver evidence, and rejected alternatives are in sections 12--13 of
  `reports/qh_score_evaluation_compression_20260810.md` and
  `reports/assets/qh_psi_qr_benchmark_20260810/`.

### Completed 10000-step optimization continuation

- Exact continuation restored all optimizer/RNG/prefetch state and completed
  10000 contiguous iterations cleanly, but steps 5001--10000 found no new best.
  Score `93.3672653` at iteration 4341 remained unchanged for 5659 iterations;
  this is strong stagnation evidence under that fixed Adam configuration, not
  proof of global optimality. The already fully evaluated best SHA-256 remains
  `d4517e03d66913d958bfac88b42b7d56228a9717c4b445f5ac28f242b049cc29`.
  Timing, resume proof, and diagnostics are in section 22 of
  `reports/qh_score_throughput_and_continuous_surface_plan.md`.

### Current physically accepted best (historical score definition)

- Highest fully evaluated sample remains QH, $N_{\mathrm{FP}}=4$, three base
  coils. Its historical constant-iota score was `93.3672653337` at iteration
  4341; the physical evaluation remains valid, but that score is not directly
  comparable with the current cubic-iota objective. Input SHA-256:
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
  expansion of $\alpha$ and $\iota$ using
  $\boldsymbol B\cdot\nabla\alpha=0$. This yields a volume straight-field-line
  coordinate, not merely one fitted surface. Current default is cubic in
  $u=\rho^2=\psi/\psi_{\rm edge}$; artifacts with `iota_degree=0` are historical.

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
  `565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729`.
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
  cubic `iota(psi/psi_edge)`, with strict axis continuation using mixed topology
  and no FP64 replay (mode 2).
  `--axis-hint-verification fp64` restores mode 1. No gradient experiment may
  leak into this production route.
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

Only current decisions are retained here; exact history is in
`MEMORY_archive_20260808.md` and the linked reports.

- Dense linear $\psi\rightarrow\alpha/\iota\rightarrow\nu$ fitting produced
  useful near-Boozer volume coordinates but did not make DESC automatic. See
  `reports/project_progress_recap_20260726.md`.
- The bounded native GPU score was corrected for current sign, $G$, volume
  weights, valid counts, low-iota cheating, and helicity competition. See
  `reports/qh_differential_qs_metric_investigation.md`.
- Flow is useful as an invertible search reparameterization, not as a direct
  high-quality generator. Latent CEM and then finite-difference Adam established
  this route; proxy and G2--G5 analytic-gradient attempts were rejected. See
  `reports/qh_flow_landscape_report.md`,
  `reports/qh_latent_score_regression_proxy_report.md`, and
  `reports/qh_blackbox_gradient_exploration_report.md`.
- The validated $N_{\rm FP}=4$, three-coil long run reached score 93.367 at step
  4341 and passed complete evaluation; exact continuation to 10000 found no new
  best. See `reports/qh_score_throughput_and_continuous_surface_plan.md`.
- Exact-zero reduced latent tails $k=16\ldots80$ failed reconstruction and must
  not be used. See `reports/qh_reduced_latent_flow_plan.md`.

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
- Current optimization/evaluation record: `reports/qh_score_throughput_and_continuous_surface_plan.md`.
- Score design: `reports/gpu_native_volume_qs_score_report.md` and
  `reports/volume_qs_score_design.md`.
- Flow training: `reports/qh_flow_matching_first_generation_report.md`.
- Corrected landscape: `reports/qh_flow_landscape_report.md`.
- Differential-QS audit: `reports/qh_differential_qs_metric_investigation.md`.
- Abandoned gradient direction: `reports/qh_blackbox_gradient_exploration_report.md`.

## 10. Active Next Action

- 2026-08-24 active branch is `codex/summary1-project-report-qh`, checked out
  directly in the repository root. The user rejected the first report and
  supplied `reports/summary1/技术报告第一版审阅.md`; the report was rewritten from
  the implementation and current evidence as `reports/summary1/技术报告.md`.
  Public README/release synchronization remains explicitly deferred until the
  user accepts the technical report.
- The rewritten report has five paper-style sections and three evidence
  appendices. It defines the complete physical chain, current ABI-10 score,
  all 82 native result fields and 32 timing fields, three evaluator modes and
  their credibility boundaries, the actual 64-direction trajectory optimizer,
  quantitative calibration, performance, and complete physical acceptance.
- Same-library evaluator results: independent/strict P50 are `2.848/0.845 s`;
  neighborhood batch-128 P50 is `3.069 s` total and `0.02397 s` per candidate.
  Strict versus independent common-`ok` score Spearman is `0.99878`; proxy
  versus strict is `0.98514`. Evidence is under
  `reports/summary1/assets/evaluator_modes_current_20260824/`.
- Eight matched 100-step cases gave Flow/data median gain `1.872/0.000`; Flow
  won 6/8 but cost 1.57x wall time at fixed steps. A common 158.1-second budget
  retained median gain `1.831/0.000`. Evidence is under
  `reports/summary1/assets/flow_pairs_current_20260824/`.
- Current-score landscape job `42189` completed cleanly on four RTX 5090s in
  13m32s: 3 centers, 12 matched directions, and 1095 independent scores. Flow
  paths are comparable to their transported local tangents and substantially
  wider/smoother than matched random data-space directions; this supports
  learned direction transport, not universal nonlinear smoothing. Evidence is
  under `reports/summary1/assets/landscape_current_20260824/`.
- Current report dynamic-library SHA-256 is
  `50877cdb7afa79433b2c337ac02953ac288b772a5c0cfc4658ec688a1d1791f5`;
  Flow checkpoint SHA-256 remains
  `39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f`.
  All benchmark GPUs ended at `0%`, `2 MiB`; postflight zombie count was zero.
- Final local verification on 2026-08-24: 199 pytest tests passed, analysis
  regeneration was byte-identical, all report links resolved, and native field
  coverage was complete. The next action is user review of the report itself.
