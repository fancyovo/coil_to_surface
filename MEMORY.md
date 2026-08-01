# Local Surface Evaluator Project Memory

> **Living source of truth. Last updated: 2026-08-01 (Asia/Shanghai).**
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

- Active local and remote branch: `qh-flow-latent-proxy`, created from
  `qh-flow-zo-adam` at `e0b21ca1ebf72a32c73b8448c731942fedf1c889` on
  2026-07-31. The remote worktree was switched after synchronizing implementation
  commit `eb6dbb0e56b01da4df2ad0bf9fef4af665a1bb4c`.
- Current experiment implementation baseline:
  `cc69110d0a5663a50fa56ac97a671973bd6f064d`
  (`Pin Adam jobs to corrected score library`). The branch also contains this
  living-memory documentation; query `git rev-parse HEAD` at session start for
  the actual tip rather than writing a self-referential commit hash here.
- Complete physical-evaluation report and assets were delivered in commit
  `4071dcc9c1132f4bf1f05e85580aa140b19477b3`.
- On 2026-08-01, complete physical evaluation of the interrupted $\eta=0.01$
  `start_10` best sample (current native score `58.151369810251744`) passed the
  corrected standard-surface workflow. Sample-specific source fitting selected
  `a=0.06`; complete standard LS/Newton plus independent dense validation
  accepted `s=0.24,0.36` and rejected `s=0.49,0.64`, so `s=0.36` was selected.
  Its $|V|$ is `0.0262317417 m^3`, $\iota=1.89480243$, dense relative residual
  `3.05807e-5`, normal-field P95 `4.98981e-5`, and surface QH error
  `1.67507696e-4`. Poincare passed. DESC stayed nested and reached normalized
  force mean/P95/max `3.043e-3/6.631e-3/1.311e-2`, while its optimizer hit the
  50-iteration limit. The standard-surface SHA-256 is
  `ac7fa3430e0ce3ed8ef3a44a4a655adb20b20067b30485e6941336d9d727f5f7`;
  DESC equilibrium SHA-256 is
  `49bf4ebe5d17ca5ebde5c76a435433efd957c03858f7e6c39d264a7c7f43f6de`.
  Full evidence, all native score components, HTML, and all eight DESC figures
  are in `reports/qh_random_start_score_adam_report.md` section 7 and
  `reports/assets/qh_score_adam_eta001_58p151_full_eval_20260801/`. The exact
  evaluated input is frozen as `evaluated_case.json` there, with SHA-256
  `e7a33bd80b660761d77b88f7308ac26720bceecc7d05fe71145b9a018d2ede18`,
  because the live `start_10/best.json` continues to be replaced by later
  optimizer improvements.
- Local `main`: `8c20859f9c66ca690d5c22cce862c055b634c1d0`.
- Current objective status: the latent-support proxy, active-optimization, and
  IID-start native-score Adam experiments are complete. The 4,096-sample IID
  pool had maximum score `41.0500821`, one sample at 40 or above, and zero at
  50 or above. The 12-start, 40-step standard-Adam panel spanned one rejected
  start and `status=ok` scores near 2, 5, 8, 10, 12, 15, 20, 25, 30, 38.7, and
  41.05. All 12 jobs completed. Initial score versus best final score was
  strongly correlated (Pearson/Spearman `0.940/0.951`), but initial score
  versus optimization gain was only weak-to-moderately correlated
  (`0.249/0.517`). The 8.00 and 19.75 starts gained 19.36 and 13.90, while the
  41.05 start gained only 4.19; local basin structure, not initial score alone,
  controls short-run optimizability. No trajectory reached 50. Full evidence
  is in `reports/qh_random_start_score_adam_report.md` and
  `reports/assets/qh_score_adam_start_sweep_29996/`. Baseline proxy evidence is in
  `reports/qh_flow_latent_proxy_experiment_report.md`; active-tail evidence is
  in `reports/qh_latent_proxy_active_optimization_report.md`. Natural Gaussian
  samples retain essentially zero proxy/score correlation. Exact-gradient
  free Adam over 8,192 starts did produce moderate enrichment in its top 512:
  median score 7.078 versus 4.837 for the reused IID control, and `status=ok`
  70.7% versus 56.3%. Per-start-RMS-projected Adam did not enrich score
  (median 3.694, `status=ok` 55.5%). The free tail moved to latent RMS 0.810
  versus IID 1.004, while within-tail raw-logit/score Pearson and Spearman were
  -0.0078/-0.0045. Thus the gain is a diverse low-radius feasibility shift,
  not evidence that the classifier ranks high physical quality. A matched-RMS
  random-direction control is required before attributing the gain to learned
  angular proxy structure. The current full local suite has 100 passing tests.
- Fixed optimizer learning rate for the earlier native-score standard-Adam
  experiment: $\eta=0.003$; the completed differentiable proxy experiment used
  $\eta=0.01$.
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
- Slurm controller access recovered. Array `29996` completed starts `0-9`, and
  supplemental array `30025` completed starts `10-11`, all with 40 steps and
  exit code 0. Remote implementation commit is
  `28e421f8db378461a6f487dc2206cdb8e46dedcb`.
- The empirical-prior background score collector is implemented at commit
  `28e421f8db378461a6f487dc2206cdb8e46dedcb`. Student smoke job `30020`
  completed cleanly with four retained samples. On 2026-07-31 the user paused
  normal collection before the next foreground experiments: active jobs `30021`
  and `30079` and queued jobs `30022` and `30023` were cancelled. Completed
  atomic shards remain in `~/local_surface_evaluator_data/qh_iid_score_corpus_v1`
  and must not be removed. A metadata-only recount on 2026-07-31 found 3,012
  completed samples in 49 shards from four streams; refresh this count at every
  later delivery.
- The current 12-start native-score Adam panel's best case is `start_10`, with
  current validated score `47.200617843580396`; its remote input is
  `~/local_surface_evaluator/runs/qh_score_adam_start_sweep_29996/start_10/best.json`.
  On 2026-07-31 the user required this case to receive the fixed complete
  physical evaluation before the follow-up optimizer jobs; that evaluation is
  now complete. The follow-up keeps the same 12 IID starts, uses $\eta=0.01$,
  runs starts 0--9 for 40 steps and starts 10--11 for 200 steps, and uses the
  current corrected score implementation.
- Complete evaluation of `start_10` began on 2026-07-31 from fixed commit
  `ebb03cf8e833ac4129a9be927bd97bf1bb584dd3`. Source-$\psi$ candidate jobs
  `30091`, `30093`, `30095`, and `30097` test `a=0.04,0.05,0.06,0.08`
  respectively under
  `~/local_surface_evaluator_worktrees/qh-flow-zo-adam/runs/qh_score_adam_start10_47p200_full_eval_20260731`.
  Select from their measured fit errors and physically covered field-line-screen
  radius; do not assume the prior sample's `a=0.08` is valid.
- All four source-$\psi$ jobs completed `0:0` in 73--75 seconds. Their numerical
  payload took about 9.8 seconds each. The sample-specific comparison selected
  `a=0.08`: validation RMS `7.383e-4`, direction-error P95 `9.591e-5`, largest
  cheap-screen pass `s=0.49` at mean radius `0.05716 m`, and explicit failure
  at `s=0.64`.
- Guarded candidate job `30099` (`s=0.24`) and replacement job `30107`
  (`s=0.36`) completed and passed. Jobs `30109` (`s=0.49`) and `30111`
  (`s=0.64`) exited 3 because the fixed off-grid residual/normal-field guards
  rejected them; this is the intended physical rejection, not an infrastructure
  failure. Pending 16-CPU duplicates `30101/30103/30105` were cancelled before
  startup and replaced by 4-CPU jobs so the last three candidates could run in
  parallel. The selected `s=0.36` surface has $|V|\approx0.04741\,\mathrm{m}^3$,
  $\iota\approx1.6971$, and off-grid relative residual `3.056e-5`; `s=0.49`
  is its required neighboring outer failure. CPU-DESC downstream job `30120`
  completed `0:0` in 5 min 40 s. DESC remained nested and reduced normalized
  force mean/P95/max to `2.796e-3/6.748e-3/1.546e-2`; its optimizer reached the
  50-iteration limit (`success=false`). All fixed artifacts and all eight DESC
  figures passed delivery validation and are reported in
  `reports/qh_random_start_score_adam_report.md` section 6, with assets under
  `reports/assets/qh_score_adam_start10_47p200_full_eval_20260731/`. Selected
  surface SHA-256 is
  `db0895246a74d93622763292292ee03d26e7ff0348e15f9bac02b54755af3965`;
  DESC equilibrium SHA-256 is
  `399ddbb4afaeeaa5a497145c4ee74ea0587ef2a18ba5d2a9bc72d2ed64ecf7c7`.
- After complete-evaluation acceptance, P107 array `30124_[0-9]` was submitted
  for the first ten IID starts with standard Adam, $\eta=0.01$, 40 steps,
  `0-9%1`, FP32 RK4-256, and corrected native score. Its common output root is
  `~/local_surface_evaluator/runs/qh_score_adam_eta001_start_sweep_20260731`.
  Elements 0--2 completed `0:0` in 11:34, 14:48, and 15:25; their
  initial-to-best scores were `0.0908226 -> 0.3662429`,
  `2.0132327 -> 4.7271975`, and `5.0007726 -> 9.4243062`. All three recorded
  clean four-GPU postflight state. Element 3 was running and 4--9 were pending
  at the handoff.
- After those three stable completions freed the required QOS slots, P107 array
  `30165_[10-11]` was submitted for starts 10--11 with $\eta=0.01$, 200 steps,
  `10-11%1`, a 10,000-second optimizer wall guard, and a 3-hour Slurm limit.
  Independent daily P107 collector job `30166` was submitted at the same time.
  Scheduler inspection explicitly confirmed `Dependency=(null)` and
  `Nice=10000` for `30166`: it can start whenever resources are free even if
  any Adam task fails, while yielding priority to the foreground jobs. Both
  submissions use fixed remote implementation commit
  `77115ef38e2423c97a5be67333d69e811268db66`.

### Slurm jobs, accepted 2026-07-31

- `29992`: COMPLETED `0:0` in 68 seconds. Recorded/startup scores were
  `0.0908069/0.0908226`; one Adam step ended at `0.0912093`. All four GPUs were
  2 MiB and 0% before and after, and commit hash `b71aac1` matched.
- `29996`: COMPLETED starts `0-9`, all `0:0`; `30025`: COMPLETED starts
  `10-11`, both `0:0`. Each used 40 standard-Adam steps with the same direction
  seed, FP32 RK4-256, and corrected native score. Mean trajectory wall time was
  861.8 seconds. The attempted original 12-element array was rejected before
  job creation by `QOSMaxSubmitJobPerUserLimit`; it did not create duplicates.
- `30020`: COMPLETED `0:0`. Two Student RTX 5090 ranks each retained two
  formal-quality RK4-256 samples. Both streams used the 153,747-sample,
  33-group empirical QUASR QH training prior, distinct seeds `480320/480321`,
  and validated checkpoint/library hashes; all four rows passed schema/SHA/ID
  checks. Both GPUs were 2 MiB and 0% before and after.
- `30021` and `30079`: CANCELLED on 2026-07-31 at the user's request to pause
  collection before foreground evaluation/optimization. Their completed shards
  are retained. The resulting batch-step exit 143/SIGTERM is the expected
  cancellation state, not a numerical failure.
- `30022` and `30023`: CANCELLED while queued behind `30021`; neither started.
- `29958`: COMPLETED `0:0` in 17 seconds. The four-sample RK4-8 smoke generated
  all artifacts; all four allocated GPUs were idle before and after, and no job
  process remained. Python emitted harmless duplicate semaphore-unlink warnings
  during worker shutdown.
- `29960`: numerically complete. Slurm accounting was temporarily unavailable
  at acceptance, but the manifest is `stage=complete`, all 4,096 rows exist,
  and all four postflight GPUs are at 2 MiB and 0%. FP32 RK4-256 decoding took
  38.34 seconds and four-worker corrected native scoring took 5165.32 seconds.
  Overall mean/median/P90/P95/P99/max are
  `4.341/3.494/9.420/11.765/21.031/41.050`; 2,149/4,096 samples are
  `status=ok`. Exceedance counts at 10/20/30/40/50 are 326/45/6/1/0.
  Source artifact SHA-256 values are `49cccc0d7b6dcb8aa8a7f9e620f897817610278a2edac215aa77edcd02a9abb8`
  for `scored_cases.jsonl` and
  `88bdeefab57f1d2f0320fb4cc339ae3a374eb25243b6ff2f70ccad614d16ea12`
  for `random_latents.npz`.
- `29900`: all 1,024 native scores completed in 1336.21 seconds after 23.22
  seconds of optimization/preparation. Slurm state is `FAILED 1:0` only because
  the final analysis interpreter inherited an invalid cwd after score output;
  all score artifacts and hashes are complete and all four GPUs ended at 2 MiB,
  0%. The script now explicitly re-enters the project before postprocessing.
- `29914`: corrected CPU-only postprocessing completed `0:0` in 24 seconds,
  including bootstrap statistics, plots, and latent-diversity diagnostics.
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
- The current initial-score/Adam study uses only IID standard-Gaussian starts.
  Do not mix proxy-ranked, proxy-optimized, CEM, QUASR-inverted, or otherwise
  constructed starts into its score distribution or optimization panel.
- Maintain an interruptible background IID score-data collector whenever GPU
  resources are otherwise idle. Each retained sample must include the exact
  flow latent, complete decoded/raw coil parameters, current total score, full
  score diagnostics/components needed to recompute future score weightings,
  and checkpoint/library/config provenance. The dataset must remain usable if
  score weights or the flow model later change; a scalar score alone is not an
  acceptable record.
- The two Student-partition RTX 5090 GPUs should continuously collect with
  distinct random seeds in one-day jobs, with multiple compliant jobs queued
  when useful. The four P107 RTX 5090 GPUs are lower-priority background
  collectors: cancel only these collector jobs when foreground project work
  needs P107, and relaunch them when P107 is otherwise idle. All six concurrent
  streams must use disjoint seed/sample namespaces. Collection is append-only
  and shard-based; interruption and restart need not resume a partially scored
  shard, but completed shards must never be overwritten or duplicated.
- Background collector conditions must be sampled from the **empirical joint
  `(nfp, n_coils)` distribution of the QUASR QH training split**, matching flow
  training. Do not sample `nfp` and `n_coils` independently and do not use a
  uniform distribution over groups. All Student and P107 streams write
  completed, uniquely named shards into the single dataset root
  `~/local_surface_evaluator_data/qh_iid_score_corpus_v1`; per-stream manifests
  remain separate under that root for provenance.
- Until the user explicitly terminates background collection, every task
  delivery must end with a short statement of the current total number of
  completed score samples in the unified corpus. Query shard metadata rather
  than estimating from running-job progress.
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
- Every complete evaluation must also tabulate the total native score and all
  of its score components, and report the selected largest accepted surface's
  surface QS error explicitly. Do not make the user infer either quantity from
  plots or a scalar total score.
- Candidate-surface existence is decided by a complete standard Simsopt
  least-squares solve followed by full Newton convergence and independent dense
  residual/normal-field/regularity checks. The stepwise `guarded` solver is a
  conservative wrong-branch diagnostic only: its rejection must never be
  reported as proof that the tested $s$ has no magnetic surface. Initial
  $\psi$-surface distance and displacement remain branch diagnostics, not
  per-step gates that replace final LS/Newton convergence.
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

- The flow model was trained on the complete extracted QUASR QH dataset, not a
  fixed condition: 170,755 total QH samples, including 153,747 training
  samples. Its supported training groups span $N_{\mathrm{FP}}=2\ldots8$ and
  one through five base coils. Each training step selected a joint
  `(nfp, n_coils)` group with probability proportional to that group's training
  count. Fixed $N_{\mathrm{FP}}=4$, three-base-coil settings belong only to
  later CEM/Adam/proxy experiments.
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
16. **Guard rejection misclassified as no surface:** the guarded solver's
    one-step Newton, monotone line search, initial-$\psi$ distance, and geometry
    gates are intentionally conservative. Use standard LS/Newton plus final
    independent validation to decide whether an $s$ is physically acceptable.

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
- Latent-support proxy active-optimization result:
  `reports/qh_latent_proxy_active_optimization_report.md`.

## 12. Next Actions

1. Do not treat the all-QUASR-vs-Gaussian proxy logit as a physical-quality
   ranking. If the user continues this direction, first score random directions
   rescaled to match the free-Adam RMS distribution near 0.81. This separates a
   generic low-radius effect from learned angular proxy structure. A production
   proxy should ultimately use current-score/feasibility labels.
2. Do not resubmit or supplement the failed random multistart experiment until
   the user explicitly requests it.
3. Before any future multistart run, isolate seeds as Slurm array tasks, record
   per-seed failures, add a separate aggregation step, and fix queue cleanup.
4. The `71.7342388` candidate's complete physical evaluation is finished and
   validated; do not rerun it unless a new diagnostic or changed algorithm
   requires it. Use section 8 of the acceptance report as the source of truth.
5. A future random-basin probability result must still include every
   predetermined unscreened seed, including failed trajectories; the one
   completed seed from `29709` is only partial evidence.

## 13. Dated Change Log

### 2026-08-01

- Corrected complete-evaluation surface acceptance. The old guarded solver is
  retained only as a wrong-branch diagnostic. Production now runs full
  standard Simsopt LS/Newton and requires independent dense-grid residual,
  normal-field, winding, and regularity checks. For the 58.1514 sample, the
  old guard rejected inner candidates that standard LS/Newton validated, while
  `s=0.49/0.64` produced collocation residuals near machine precision but failed
  off-grid checks; solver `success` alone is therefore also insufficient.
- Job `30395` completed the 58.1514 sample's fixed downstream evaluation `0:0`
  in 5 min 40 s. Delivery validation passed and found all eight successful DESC
  PNGs cited in the report.
- Student background collector `30399` is running independently. The corrected
  $\eta=0.01$ panel now has complete starts 0--10. Jobs `30406_6/7` completed;
  task 5's pre-score cwd `ENOENT` failure was archived and replacement
  `30455_5` completed `0:0` in 13 min 58 s, improving `11.97646 -> 25.58460`.
  The cwd fix is to launch optimizer workers from `/`. Resume smoke `30403_11`
  proved state/momentum/RNG/history continuity before the long resume.
- On 2026-08-01, inspection of the active $\eta=0.01$ `start_10` trajectory
  found that hard score gates, not only Adam beta choices, dominate some local
  failures. Step 93 reached score `59.3632156562` with all probes `ok` and
  update RMS `6.70e-4`; at step 94, two of eight probes became `no_axis`, two
  directional deltas were about `-58`, gradient RMS rose to `1028.54`, update
  RMS rose to `6.43e-3`, and the center left the best point. The same pattern
  occurred at step 51 with `drift_rejected` probes. Current
  $(\beta_1,\beta_2)=(0.9,0.999)$ gives second-moment memory much longer than a
  40--200 step run, but beta tuning must be controlled against these gate
  crossings. The planned local higher-order option is a fixed low-dimensional
  subspace BFGS/quadratic trust region started from the saved best point after
  an explicit smoothness check; a full 300-dimensional Hessian is out of scope.
  Detailed evidence and the proposed beta comparison are in
  `reports/qh_random_start_score_adam_report.md` section 8.
- Long-resume element `30411_10` completed `0:0` at 200/200 steps in 51 min
  58 s for the resumed segment. Its re-decoded initial score was `38.65902`,
  step-40 best was `55.41755`, and final/best score was `59.9799763154` at
  step 200; total trajectory numerical wall time was `4902.43 s`. Exactly
  3/200 iterations contained invalid probes, but they produced gradient RMS
  `743--1029` and update RMS about `0.006`, dominating local regressions despite
  99.6875% mean endpoint validity. Array element `30411_11` was cancelled
  before startup (`CANCELLED`, zero elapsed) by user request and must not be
  restarted unless requested. Across starts 0--10, the mean 40-step best gain
  was `12.2813` for $\eta=0.01$ versus `7.2086` for $\eta=0.003$; detailed
  trajectories are in `reports/qh_random_start_score_adam_report.md` sections
  8--9 and `reports/assets/qh_score_adam_eta001_start_sweep_20260731/`.
- Local fixed-subspace damped BFGS with $h/h/2$ smoothness checks, trust-radius
  capping, and batched line search was implemented at commit `771c4d3`; the
  complete local suite has 107 passing tests. The immediate task is to pilot it
  from the score-59.98 `start_10` best. Only if it improves should the same
  fixed protocol be applied to prior Adam best cases above score 40, including
  the best $\eta=0.003$ endpoint.
- Subspace-BFGS smoke job `30477` completed `0:0` in 67 s with clean four-GPU
  postflight. It reproduced the start score as `59.97997631540494`, but the
  fixed-subspace gradients at $h=0.0025$ and $0.00125$ had cosine `-0.2190`
  despite all probes being `ok`, so the strict smoothness gate correctly
  stopped before a BFGS step. A fine probe itself reached `60.13922835`, proving
  a nearby improvement exists but not yet proving a smooth/superlinear local
  regime. The next diagnostic must use smaller $h$ and trust radius; do not
  classify this smoke as BFGS success.
- Fine-scale smoke `30479` completed `0:0` in 75 s with clean postflight. At
  $h=0.00125/0.000625$, projected-gradient cosine improved to `0.8443`, all
  probes remained `ok`, and one accepted damped-BFGS step improved
  `59.97997632 -> 59.99072840`. It accepted line alpha `0.125` with latent-RMS
  step `6.25e-5`; the resulting inverse-Hessian condition number was high
  (`2.82e5`), so this is a positive one-step result, not yet proof of sustained
  superlinear convergence. The former instruction to continue treating these
  as validated fine scales is superseded by the diagnostics below.
- Medium BFGS job `30485` stopped after only two accepted steps and three line
  rejections: `59.97997632 -> 59.99709281`, with accepted latent-RMS steps
  `6.25e-5` and `3.125e-5`. It then shrank the trust radius to `2e-5`. This
  `+0.0171` result is too small and too conservative to establish useful BFGS
  convergence. More importantly, calibration-only job `30490` found gradient
  cosine `-0.6566` for $h=0.000625/0.0003125$, while the larger
  $0.0025/0.00125$ pair had already given `-0.2190`. Thus there is no verified
  asymptotically smooth finite-difference range around this score-59.98 point;
  the intermediate `0.8443` cosine was only a two-scale coincidence.
- Empirical scale calibration on 2026-08-01: previous high-score Adam improving
  steps have latent-RMS median/P75 about `7.13e-4/1.05e-3`; the score-69 to 72
  standard-Adam run's final 100 steps have median `4.11e-4`; and the old hybrid
  run averaged `8.74e-4`. The earlier landscape used $h\approx0.01$ and showed
  the most consistent broad directional signal near $0.009$--$0.018$, while
  the current point begins hitting hard gates by `0.005`. Therefore later
  local comparisons use an evidence-based pattern/trust radius of `0.00125`,
  floor `0.0002`, and cap `0.003`; the old `2e-5` floor and `0.01` cap are not
  calibrated for this point. BFGS and PRP+ nonlinear CG remain controls, but a
  fixed-subspace coordinate pattern search is the method whose assumptions fit
  the observed nonsmooth objective.
- Method-control implementation commit `e2effc2` adds PRP+ nonlinear CG and
  derivative-free coordinate pattern search alongside BFGS; all three use the
  same fixed subspace, decoder, score, and trust cap. The complete local suite
  has 109 passing tests. On the score-59.98 point, calibrated 12-step pilots
  reached `60.2558895` with BFGS, `60.1833348` with pattern search, and
  `60.1037672` with NCG. These are real accepted score gains, but not evidence
  of superlinear convergence: BFGS gradient norms rose and NCG's PRP+ beta
  reached `11.60` under noisy gradients.
- The calibrated BFGS/pattern protocol was applied once to the five remaining
  prior Adam endpoints above score 40 and is complete; do not expand to more
  starts unless requested. For eta=0.01 starts 7/8/9, BFGS gains were
  `+0.2999/0/+0.1603` and pattern gains `+0.3290/0/+0.1668`. For eta=0.003
  starts 10/11, BFGS gains were `+5.8091/+1.7510` and pattern gains
  `+3.2673/+1.1767`. The large eta=0.003 gains primarily show those 40-step
  trajectories were unfinished, not that BFGS is generally superlinear.
- Same-start zero-momentum standard-Adam controls are complete. From score
  `59.9799763`, 12-step eta=0.003 reached historical best `60.3312508` at step
  4 in `102.3 s` but ended `59.6463478`; eta=0.01 reached the overall best
  `60.5642672` at step 2 in `57.6 s` but ended `59.0384841`. Both crossed
  `no_axis` probe gates after their best. Thus zero-momentum Adam with a
  preserved running best beat BFGS on this mature point, while final current
  states are invalid choices. Full evidence and figures are in
  `reports/qh_random_start_score_adam_report.md` section 10 and
  `reports/assets/qh_local_subspace_followup_20260801/`.
- Foreground local-optimizer work ended after jobs `30501`--`30513`. Per the
  durable collection policy, Student collector `30399` remains running and
  low-priority four-GPU P107 collector `30527` was relaunched after foreground
  completion. Neither is a dependency of the other. A metadata-only recount at
  delivery found exactly 10,628 completed samples in 168 shards from 14
  streams (`ok=4597`, `no_axis=2973`, `no_surface=707`,
  `drift_rejected=2281`, `flux_rejected=70`).
- On 2026-08-01, dirty-gradient Adam follow-up array `30532` ran the exact
  beta1=0.9 baseline and beta1=0.7/0.5 controls to completion from the fixed
  score-59.97998 start. Its first robust-filter implementation was invalid:
  retaining and rescaling only the two valid directions at step 3 drove the
  updated center to `no_axis` even though it reduced gradient RMS from about
  1056 to 22.8. Element 3 and pending elements 4--6 were cancelled. This result
  is superseded and must not be treated as evidence against robust filtering.
  The replacement policy skips the entire Adam/moment/parameter step whenever
  any directional pair has a non-`ok` endpoint; all-`ok` directional outliers
  are still winsorized using a scale-adaptive median/MAD ratio. P107 collector
  `30527` remains cancelled during foreground work; Student collector `30399`
  remains independent and running.

### 2026-07-31

- Completed active proxy-tail optimization. From 8,192 paired starts per
  variant, free Adam top-512 improved median native score from IID 4.837 to
  7.078 and `status=ok` from 56.3% to 70.7%, while RMS-projected top-512 did not
  improve. The free tail is strongly low-radius (median 0.810), remains diverse,
  and has zero within-tail proxy/score correlation. The result is moderate
  feasibility enrichment with an unresolved radial confound, not a validated
  physical-quality proxy. Full report and assets are under
  `reports/qh_latent_proxy_active_optimization_report.md` and
  `reports/assets/qh_latent_proxy_optimized_29900/`.
- Hardened latent proxy jobs after smoke diagnostics: direct `sbatch --chdir`
  does not change exported `SLURM_SUBMIT_DIR`, so direct submissions set
  `PROJECT` explicitly; preflights now report missing paths/hashes; score
  analysis accepts batches smaller than ten; final postprocessing is a
  restartable CPU job and explicitly re-enters the project directory.
- Clarified the latent-proxy score scatter in both plot and report: green means
  `status=ok`; red is the union of `no_axis`, `no_surface`, `drift_rejected`,
  and `flux_rejected`; the black decile mean includes both colors, while the
  separately reported status-ok correlation uses green points only. Future
  score scatter plots must label validity colors directly rather than relying
  on unexplained color coding.
- Completed training-only job `29820` on the full inverse-QH latent dataset.
  The best validation AUC was 0.93039 at step 1600; training continued to step
  5100 and stopped only after the final validation plateau. Added an
  authoritative FP32 evaluator with validation-only monotone Platt calibration,
  changed future training validation/test inference to FP32, and changed the
  score-correlation preparation path to consume the same FP32 calibration.
  The original BF16 test summary is retained as provisional ranking evidence,
  not as the final calibrated result.
- Single-GPU authoritative FP32 evaluation job `29822` completed `0:0` in 42
  seconds. On 17,016 held-out balanced test examples it obtained ROC-AUC
  0.93414, AP 0.94555, and accuracy 0.85349 with validation-selected threshold;
  the confusion counts are TN/FP/FN/TP = 8088/420/2073/6435. For the target
  `nfp4_nc3` group, test AUC is 0.95097. The latent-RMS-only baseline test AUC
  is 0.71343. Validation-only Platt calibration reduced held-out test log loss
  from 1.61842 to 0.34705 without changing ranking. Artifacts are at
  `~/local_surface_evaluator/runs/qh_latent_proxy_eval_29822/` and locally under
  `reports/assets/qh_latent_proxy_eval_29822/`.
- Four-GPU native-score correlation job `29824` completed `0:0` in 22 minutes
  08 seconds with corrected score-library SHA `0b7342...`. It predicted 131,072
  prior latents, selected 768 prediction-rank-stratified plus 256 independent
  IID cases, decoded and scored 1,024 cases. All-sample Pearson/Spearman was
  -0.0418/-0.0161; IID was -0.0205/-0.0269; stratified was -0.0566/-0.0120;
  status-ok only was -0.0271/-0.0107. Even the five cases with proxy probability
  at least 0.9 had mean/max score 2.50/8.05 and only 40% status-ok. This
  invalidates using the current classifier as a physical-quality prefilter.
  Artifacts are at
  `~/local_surface_evaluator/runs/qh_latent_proxy_score_29824/` and locally under
  `reports/assets/qh_latent_proxy_score_29824/`. All GPUs were at 2 MiB and 0%
  utilization postflight, with no workers left behind.
- Created branch `qh-flow-latent-proxy` and implemented separate, restartable
  stages for 4-GPU FP32 RK4 inversion, validation-driven proxy training,
  held-out confusion/enrichment evaluation, and optional current-native-score
  correlation. The score follow-up separates PyTorch decode and native score
  worker processes so PyTorch does not retain GPU0. All 83 local tests pass;
  remote smoke/full jobs are not yet submitted.
- Added the score-free latent-proxy feasibility design in
  `reports/qh_flow_latent_proxy_feasibility.md`; no experiment has started. It
  records that inverse-QUASR versus Gaussian classification converges to chance
  for an ideal flow, while the current model's known decoded-quality gap proves
  a latent mismatch is present. The experiment will measure how much of that
  mismatch a cheap held-out classifier can capture; it is not yet a calibrated
  continuous physical-score predictor.
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
