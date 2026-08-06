# Local Surface Evaluator Project Memory

> **Living source of truth. Last updated: 2026-08-06 (Asia/Shanghai).**
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

- On 2026-08-06, branch `score-fast-continuation` reached local/remote commit
  `8e58bf5` and submitted the forward-only optimizer throughput matrix. The
  implementation adds recoverable cross-iteration flow prefetch, central or
  randomly signed one-sided score finite differences, and a shared four-vector
  direction bank; the manifest explicitly records
  `gradient_source=score_finite_differences_only`. It does not call flow VJP,
  autograd/backward, G1--G4, native score gradients, or any earlier black-box
  gradient experiment path. Local validation is `169 passed`. Students smoke
  jobs `32991` (RK4-64 central-4 pipeline) and `32994` (RK4-64 one-sided-4
  pipeline) completed cleanly, demonstrated batch-9/batch-5 prefetch cache
  hits, and left their GPUs at 0% / 2 MiB.
- The formal fixed-start 200-step matrix is active under remote
  `runs/score_fast_optimizer_matrix_20260806/`. It holds the job-32804 start,
  seed, perturbation, Adam settings, continuous score, strict axis
  continuation, and robust guards fixed; all runs enable flow pipelining. Jobs
  are: `32995/32996/32997` = RK4-256/128/64 central-4, `32998/33000/33001` =
  RK4-256/128/64 one-sided-4, and `33002/33003/32999` = RK4-256/128/64
  central-2. Each run owns one RTX 5090 and uses only score GPU 0. At the last
  2026-08-06 check, jobs `32995`--`32998`, `33000`, and `33001` were running
  cleanly at steps 3--9; the other three were normally QOS-pending. Observed
  central-4 and one-sided-4 times were about 13--15 and 7--9 s/step,
  respectively, giving a 60--80 minute estimated six-GPU completion time.
  Acceptance must report per-stage and total wall time, all nine score curves,
  and comparison against historical continuous-score 200-step job `32804`.
- On 2026-08-06, complete physical evaluation of the frozen continuous-score
  Adam best (`best.json` SHA-256 `1b1d7892498a2e67f646c3bba62ab6e81e696e378315bdd29749c55a7c5ccef7`)
  completed through the fixed `evaluation/full_physical/` route. Sample-specific
  `a=0.08` was selected; standard LS/Newton accepted nested `s=0.24/0.36/0.49`
  with increasing volumes and selected `s=0.49`, while `s=0.64` failed the
  fixed candidate-point budget and supplies the nearest outer failure. The
  selected surface has iota `1.58013`, volume magnitude `0.06518 m^3`, dense
  normal-field P95 `3.77e-5`, and face QH error `4.56e-6`. Poincare is nested,
  direct and DESC colored-contour plots are valid, and the 3D HTML uses the
  full coil/surface set. CPU DESC remained nested and reduced independent
  normalized force mean/P95/max from `1.080/2.020/6.007` to
  `1.789e-3/3.871e-3/1.552e-2`; it stopped at the default 50-iteration limit
  with optimizer `success=false`, so this is a strong residual reduction, not
  strict solver convergence. All outputs are frozen under
  `reports/assets/qh_score_fast_continuation_20260805/adam_start10_200_32804/full_evaluation/`
  and documented in sections 13--14 of
  `reports/qh_score_throughput_and_continuous_surface_plan.md`. Jobs
  `32934/32936/32938/32940`, `32942/32944/32946/32948`, and `32951` are all
  complete; no project Slurm job remains active from this evaluation. The
  report and frozen artifacts are synchronized locally/remotely in commit
  `d1de539`.
- The same 2026-08-06 audit resolves the 200-step timing ambiguity. Historical
  `5363.07 s` is job 31058's four-GPU wall time (`1:29:23` internal, about
  `1:29:37` Slurm elapsed), not a single-GPU time or summed GPU-seconds. Old/new
  flow decode was `970.26/947.70 s`, while native-score wall was
  `4359.38/859.45 s`; native score accelerated `5.07x`, flow only `1.02x`, and
  total wall therefore accelerated `2.88x` to `1862.59 s`. Every SPSA step
  already decodes all eight `(8,3,100)` FP32 endpoints in one RK4-256 batch on
  one GPU; measured batch throughput is `3.25 samples/s` and about `7.42x` the
  per-sample throughput of batch one. The decoded endpoints are then scored on
  four persistent GPU workers at about `95.4%` parallel efficiency. Do not
  describe flow as eight serial Python decodes; its remaining cost is the 256
  serial RK4 time steps and the dependent second center decode.
- Correction on 2026-08-06: the two forward-flow batches are dependent within
  one iteration but can be merged by cross-iteration pipelining. After endpoint
  scores produce proposed center `z[k+1]`, decode that center plus its next
  eight antithetic endpoints once as batch nine. Score the center first, then
  score the cached endpoints with the newly validated center-axis hint; discard
  and recompute them only on rejected/backtracked centers or skipped updates.
  This is feasible but not implemented or benchmarked. Current optimization is
  FP32 RK4-256. Existing same-step closure evidence supports RK4-64 for a new,
  self-consistent forward-score optimization and RK4-128 for conservative
  inversion; RK4-32 is unacceptable. Never silently continue a saved RK4-256
  latent with RK4-64 because that changes the discrete flow mapping.
  A separate unvalidated throughput candidate reuses each accepted center
  score and evaluates four randomly signed one-sided orthogonal probes instead
  of four antithetic pairs. This halves endpoint score calls while retaining
  four sampled directions, but adds first-order curvature error; compare it
  against two-direction central differences at equal endpoint-call budget
  before changing the production optimizer.
- On 2026-08-05, branch `score-fast-continuation` accepted the production
  candidate documented through report/artifact commit `06f6452`: opt-in
  p1/t128/k400 continuous surface confidence, six fixed flux bisections, and
  strict local axis continuation; legacy global score remains the default and
  the complete LS/Newton/DESC path remains available. Same-start four-GPU job
  `32804` completed 200 old-configuration SPSA/Adam iterations in 1862.59 s,
  improving continuous score 85.3999 -> best 91.8749 at step 198, with mean
  9.058 s/step versus historical job 31058's 26.655 s/step and 5363.07 s total
  (2.94x iteration speedup). Four endpoint branch losses safely skipped four
  updates, all 200 centers remained ok, and two invalid full proposals recovered
  at the fixed 0.5 backtrack. Cross-score job `32815` then showed that the old
  best scored legacy/continuous 93.1656/91.5188 while the new best scored
  93.3829/91.8742; legacy QH error per helicity improved 0.0023003 -> 0.0019044
  without reducing legacy inverse aspect. Both jobs had empty stderr and idle
  postflight. Final report sections 9--12 are in
  `reports/qh_score_throughput_and_continuous_surface_plan.md`; frozen assets
  are under `reports/assets/qh_score_fast_continuation_20260805/`.
- On 2026-08-05, P107 four-GPU holdout job `32797` completed from remote commit
  `e1b01de` on the untouched second 128-case validation block. All 69 legacy-ok
  cases remained continuous-ok. On that subset, overall/high-score>=90
  Spearman was 0.9730/0.9390, top-20%/top-10% overlap was 92.3%/100%, and
  log-QH Spearman was 0.99978. Strict continuation median/P95 was
  0.990/1.254 s, a 5.68x paired median speedup; global continuous median/P95
  across all statuses was 2.857/4.629 s versus legacy median 5.179 s. The new
  bounded method accepted 44 old hard failures, but their maximum/P95 scores
  were 89.46/88.49, so none entered the old score>=90 extreme tail. This
  accepts p1/t128/k400 plus flux-weighted isotonic confidence for the completed
  production-candidate optimizer test above. Frozen results are under
  `reports/assets/qh_score_fast_continuation_20260805/validation_holdout_offset128/`.
- On 2026-08-05, branch `score-fast-continuation` reached local/remote commit
  `ed6a861`. P107 single-GPU build/smoke job `32794` completed successfully,
  compiling and testing the flux-weighted isotonic confidence follow-up.
  Legacy remained 96.184742, continuous p1 remained ok at 94.852610 and
  3.216 s, exact legacy hint reproduced legacy exactly, and an impossible hint
  returned `branch_lost` in 0.031 s. The next final ranking audit must use the
  second fixed 128-case block of the validation split (`sample_offset=128`),
  because the first block was used to diagnose and fix the near-axis issue.
- On 2026-08-05, branch `score-fast-continuation` reached local/remote commit
  `8cf934b`. P107 single-GPU build/smoke job `32763` completed successfully.
  Legacy score remained 96.184742; an exact strict hint reproduced it exactly,
  reduced the axis stage from about 2.09 s to 0.206 s, and an impossible hint
  returned `branch_lost` in 0.031 s. The selected continuous mode completed in
  3.216 s versus 7.701 s legacy on this global-axis smoke; its surface stage
  was 0.484 s versus 3.520 s legacy. Postflight was idle. Independent four-GPU
  validation job `32766` then completed 128 fixed validation-split QH cases.
  Among 79 legacy-ok cases, selected p1/t128/k400 retained 77 as ok, achieved
  overall/high-score>=90 Spearman 0.9571/0.9034, top-20%/top-10% overlap
  93.3%/85.7%, and log-QH Spearman 0.9965. Strict-hint median time was 0.994 s,
  a 6.10x paired speedup; global continuous median was 2.903 s versus 5.251 s
  legacy across all statuses. Nineteen legacy 16-period drift rejections became
  continuous ok; the highest scored 89.63 and most legacy rejections were just
  beyond the old hard 5% drift cutoff. However two legacy-ok cases became
  no-surface because near-axis relative drift at tiny normalized flux was
  amplified by tiny radius and the prototype's cumulative maximum propagated
  that numerical outlier to every outer level. This is an algorithmic defect,
  not an accepted ranking change. The follow-up replaces cumulative maximum
  with flux-interval-weighted isotonic regression of confidence, preserving a
  nonincreasing continuous confidence while allowing stable outer evidence to
  correct near-axis numerical noise. That follow-up remains remotely
  unvalidated. Frozen validation output is under
  `reports/assets/qh_score_fast_continuation_20260805/validation_selected_128/`.
- On 2026-08-05, corrected four-GPU calibration job `32751` completed all 128
  fixed QH cases in 309.88 s wall time with empty stderr. For the 65 legacy-ok
  cases, continuous p1/t128/k400 preserved all 65 as ok, reduced median score
  time to 1.144 s (5.36x versus the paired legacy median), and preserved the
  physical log-QH ordering with Spearman 0.9983. Overall score Spearman was
  0.9144, but old-score>=90 Spearman was only 0.6930. Component analysis
  localized this high-tail mismatch to the surface subscore (high-tail
  Spearman 0.3388): it still mixes confidence/risk measured at the originally
  predicted edge even after flux bisection has selected a smaller actual edge.
  Axis, psi, iota, and coil components are unchanged; coordinate high-tail
  Spearman is 0.9945 and volume-QS high-tail Spearman is 0.8325. Therefore this
  calibration validates the runtime mechanism and QS calculation but does not
  yet accept the final continuous score definition. Frozen raw output is under
  remote `runs/score_fast_continuation/calibration_fluxfix_32749/` (the path
  contains the planned rather than actual job ID); local analysis is under
  `reports/assets/qh_score_fast_continuation_20260805/calibration_fluxfix_128/`.
  Offline ablation on this calibration subset selected a definition that does
  not reuse stale proposal-edge risk: continuous surface quality is 65% of the
  final physical inverse-aspect size quality plus 35% of a smooth quality of
  the final continuously calibrated normalized-flux level. This reassigns the
  legacy 25% drift plus 10% discrete-count weights to one continuous extent
  quantity. On the development subset it raises overall/high-score>=90
  Spearman to 0.9282/0.8120. These are selection-set figures only; the formula
  remains unaccepted until remote smoke and an independent validation split.
- On 2026-08-05, P107 single-GPU smoke job `32748` completed successfully from
  commit `8f9d996`, rebuilding the ABI-10 library and confirming all legacy,
  continuous p1/p2/p4, valid-hint, and invalid-hint paths. Corrected four-GPU
  calibration job `32751` then completed as recorded above; due to an explicit
  output override it writes `runs/score_fast_continuation/calibration_fluxfix_32749/`
  despite job ID 32751.
- On 2026-08-05, corrected P107 four-GPU calibration job `32741` completed all
  128 fixed QH cases in 318.02 s wall time with empty stderr. Legacy statuses
  were 65 ok, 46 drift-rejected, 13 no-surface, 3 flux-rejected, and 1 no-axis.
  The first continuous calibration is rejected as a production definition:
  among the 65 old-ok cases, all tested variants prematurely returned
  no-surface for 25--26 cases because the prototype still hard-rejected the
  predicted edge before allowing flux search to shrink it. For cases reaching
  QS, old/new log-QH Spearman was already 0.998--0.999 and all axis hints
  matched exactly, so the dominant issue is this early gate, not alpha/QS.
  The local follow-up removes that gate and searches descending old levels only
  to obtain a robust passing lower bracket before six continuous bisections;
  this fix is not yet remotely validated. Frozen first-pass analysis is under
  `reports/assets/qh_score_fast_continuation_20260805/calibration_128/`.
- On 2026-08-05, P107 four-GPU calibration job `32738` was cancelled after
  1m46s because all four Python workers accidentally used the default
  `device_id=0`, causing GPU0 QR allocation OOM. Its partial output is invalid
  and must not enter analysis. The launcher is being fixed to pass each
  worker's explicit device index; this is a benchmark orchestration bug, not a
  score-algorithm failure.
- On 2026-08-05, P107 single-GPU follow-up smoke job `32732` completed from
  source commit `d70635a`. Smooth radius confidence moved the predicted edge
  inward from about 0.65 to 0.568, and out-of-domain strict hints now return
  `branch_lost` in 0.031 s, but the continuous edge still failed the cheap flux
  calibration; this intermediate score is invalid. Commit `fdaca40` therefore
  adds a fixed-cost continuous flux boundary search: test the predicted upper
  edge and innermost level, then do six bisections of the passing/failing
  interval. P107 smoke job `32736` completed successfully under
  `runs/score_fast_continuation/smoke_32736*`. On the same score-96.185 case,
  continuous p1/p2/p4 all reached `status=ok` with scores
  `94.751/94.744/94.732`; p1 took 3.140 s versus 7.728 s legacy, while the
  fixed eight-attempt continuous flux search cost only 0.039 s. This validates
  mechanics on one case only, not ranking or production defaults.
- On 2026-08-05, branch `score-fast-continuation` is at local/remote source
  commit `c71a5fb`. P107 single-GPU build/smoke job `32728` was submitted from
  `scripts/slurm_score_fast_build_smoke.sh`; it writes under
  `runs/score_fast_continuation/smoke_32728*` and tests legacy, continuous
  1/2/4-period, valid strict axis-hint, and invalid strict-hint paths. It passed
  `sbatch --test-only`, compiled successfully, and left the allocated GPU at
  2 MiB/0%. On the score-96.185 smoke case, the strict valid hint reproduced
  the legacy score exactly and reduced axis search+trace from about 2.085 s to
  0.401 s. Fine timing split the 3.445 s legacy surface stage into about
  0.472 s mixed one-period work, 1.070 s FP64 recheck, and 1.901 s long-trace
  work. The first continuous prototype correctly removed the latter two costs,
  but incorrectly retained radius-clipped outer levels and returned
  `no_surface`; this result is diagnostic-only. The local follow-up now applies
  a smooth radius-limit confidence and immediately rejects out-of-domain strict
  hints; it is not yet committed or remotely validated.
- On 2026-08-05, branch `score-fast-continuation` gained an unvalidated ABI-10
  prototype. It preserves legacy defaults while adding 16 fine-grained native
  timing fields, an opt-in continuous short-horizon surface-confidence mode,
  and an opt-in magnetic-axis hint/strict-continuation interface. The prototype
  has passed local Python syntax and diff checks only; it must not be treated as
  numerically accepted or deployed until remote compilation, legacy regression,
  ranking, high-tail, smoothness, and timing A/B tests pass. No Slurm job has
  been submitted for it yet.
- On 2026-08-05, branch `score-fast-continuation` was created from archived
  commit `6e3c0a4` to pursue the native-score throughput and continuity work
  below. The branch is now active; no remote jobs have been submitted yet.
  The previous planning-only/no-branch state is superseded.
- On 2026-08-05, the user ended black-box-gradient development as a research
  direction and redirected the next branch toward native-score throughput and
  score continuity. No new branch has been created and no experiment has been
  launched yet, because this turn was explicitly planning-only. The proposed
  work has two independent tracks: (1) replace discrete surface-level selection
  and production 16-period candidate traces with deterministic stratified
  short-horizon volume samples, a calibrated monotone surface confidence
  function `p(s)`, and a continuous effective edge; retain 16-period traces
  only as offline/full-evaluation audit labels until 1/2/4-period false-positive
  rates are validated; (2) add an optimizer-only magnetic-axis continuation
  API that starts all local endpoints from the same validated center-axis
  branch token and returns `branch_lost` instead of silently switching axes.
  Standalone/corpus score must remain history-independent and retain global axis
  search. Before changing either algorithm, split axis and surface timing into
  non-overlapping GPU/CPU sub-stages; current 1000-sample medians are about
  3.28 s total, 1.62 s axis search, 1.01 s surface screening, 0.43 s for all
  psi stages, and 0.058 s for alpha/iota plus QS. The plain-language design,
  equations, anti-cheat rules, validation gates, and staged action plan are in
  `reports/qh_score_throughput_and_continuous_surface_plan.md`.
- On 2026-08-05, the 27-run G3-reference matrix sweep from commit `4a552c6`
  completed on six RTX 5090 jobs `32525`--`32530`. The durable optimizer
  contract is: native ABI-9 score remains a black box; flow-pulled
  fixed-geometry G3 is only a correlated search prior and must never be used
  directly as the score gradient; exact score secants determine direction sign
  and magnitude, and an exact monotone score gate accepts states. Do not resume
  G4 unless explicitly requested. All 27 configurations completed 100 steps
  from the same nfp4/nc3 `start_10` with RK4-64 and `h=0.0025`; there were zero
  accepted score drops, and 135 center repeats agreed within
  `5.68e-14`. The experiment measured 5,400 directions, required 10,800
  plus/minus endpoint scores, and actually made 18,712 black-box calls versus
  a 27,162 conservative upper bound. It consumed 37.43 GPU-hours and 7.19 h
  six-GPU wall time; all six stderr logs were empty and postflights were idle.
  The matrix sweep has no active jobs. Any later historical bullet that still
  calls pre-sweep `324xx` jobs active or queued is superseded as job-state
  metadata; its numerical-protocol and bug-fix records remain relevant.
  The best configuration was `(lr=0.03,beta1=0.7,K=2 random directions plus
  G3)`, improving `85.8396799 -> 92.6847377` with best step 87. The best
  `K=0/1/2` scores were `89.5130/92.2975/92.6847`; therefore G3 alone stalls,
  one random direction has good cost/performance, and two random directions
  raised the observed ceiling, but K=2 beat K=1 in only 6/9 paired cells and
  fixed K=4 is not justified by this sweep. Learning rate `0.003` was clearly
  too small; `0.01/0.03` were effective. Beta1 `0.7` gave the best mean and
  maximum, while `0.9` did not improve further. This is a same-start 100-step
  hyperparameter screen, not multi-seed confidence or convergence proof.
  Final plain-language acceptance is section 20 of
  `reports/qh_blackbox_gradient_exploration_report.md`; reproducible assets
  are under `reports/assets/qh_reference_direction_sweep_20260806/`, and the
  offline analyzer is `scripts/analyze_qh_reference_direction_sweep.py`.
- On 2026-08-05, branch `qh-blackbox-gradient` reached local/remote source
  commit `e601fb4`. Commit `7803aa2` is the batch-1 correctness baseline; the
  two active long optimizations `32449/32451` started from that commit and are
  unaffected by the later source update. Commit `d2545dc` removes a bounded
  but material performance waste for future runs: all smooth backtrack states
  are independently batch-1 decoded, their exact scores are evaluated in one
  parallel GPU batch, and G3 is computed only for the first score-acceptable
  candidate. The numerical mapping, backtrack order, exact gate, and accepted
  state are unchanged; related local tests still pass `26/26`, but the new
  fast path requires a remote numerical smoke after the active jobs release a
  GPU. Commit `9b1e6e4` fixes a logging-only ambiguity: older accepted-step
  rows computed `secant_center_score_delta` against the post-update score, so
  that field represented the negative accepted gain rather than the center
  repeat error. Future rows compare against the pre-update center and record
  `accepted_score_gain` separately; optimization decisions and old accepted
  scores are unaffected. The focused local suite passes `17/17`. Commit
  `e601fb4` extends the offline analyzer with explicit continuation offsets,
  reconstructed center-repeat errors, and accepted-score gains so staged runs
  can be plotted on one honest iteration axis. A standalone plain-language
  midterm status, terminology guide, algorithm walkthrough, bug ledger, and
  explicit list of supported/unsupported conclusions is appended as section
  19 of `reports/qh_blackbox_gradient_exploration_report.md`. Section 18/19,
  the clean smoke, and the batch-1 G4 reference assets are archived in commit
  `83b40cb`; final long-run results will be appended separately. Students job
  `32459` is queued with `afterok:32451` for a strict
  three-step A/B against job `32434` using the same high center, seed, and
  projected proposal; it writes
  `runs/qh_g3_subspace_batch1_batched_gate_smoke_students_3_20260805`.
  A critical numerical-protocol error was found earlier: the old
  exact same-basin reference decoded physical endpoints in batches of 32,
  while G2/G3 flow VJPs and accepted optimizer candidates use batch 1; the
  later secant optimizer likewise decoded its ten probes as one batch but
  re-evaluated proposals at batch 1. The same latent is already known to move
  the exact score by as much as about `0.0595` solely from the FP32 GEMM batch
  shape. Therefore the old batch-32 full-vs-G2/G3 cosine evidence and all
  batch-10 quadratic/secant direction-quality conclusions are superseded by
  the clean rerun recorded below. Scores of states that passed the exact batch-1
  acceptance gate remain valid. A controlled first-step replay used exactly
  the same five directions, but two random-direction slopes changed sign
  between batch 10 and batch 1; the two RMS-0.01 Adam updates had cosine only
  `0.360` and reached exact scores `87.4827` versus `86.1107`. Commit `7803aa2`
  makes every secant center,
  positive/negative endpoint, and quadratic-axis candidate use independent
  batch-1 RK4 decoding, records the independently rescored center delta, and
  parameterizes reference decode batch size/direction count; related local
  tests pass `26/26`. P107 four-GPU job `32432` completed the clean batch-1,
  32-direction, `h=0.0025` exact reference rerun for start-10 steps
  `50/89/100/120`, writing
  `runs/qh_g2_current_basin_reference_batch1_k32_20260805`. All four centers
  retained 32/32 smooth directions; 260 exact scores averaged `5.909 s` with
  `6.136 s` P95, and four-GPU postflight was clean at 2 MiB/0%. Students two-GPU
  job `32434` completed a five-step batch-1 projected-proposal smoke from the
  score-`92.6658409` step-113 center, writing
  `runs/qh_g3_subspace_batch1_projected_smoke_students_5_20260805`. Every
  independently rescored center matched the optimizer center exactly; ten
  independent flow decodes cost about `5.56 s` on the first step. No smooth
  projected proposal was accepted, while two already measured probe endpoints
  improved the score to `92.6842398`; mean step time was `78.46 s` on two
  GPUs. The accepted probes' screening and independently re-decoded exact
  scores agreed to `2e-14` and zero. Pre/postflight was clean at 2 MiB/0%.
  This validates the batch-1 protocol and exact gate, not the projected
  proposal as an optimizer. Single-GPU P107 alignment jobs
  `32438/32440/32441/32442` also completed for steps `50/89/100/120`. Their
  32-direction full/G3 cosines are `+0.238/-0.223/-0.234/+0.045`, while
  full/fixed-G4 cosines are `-0.551/-0.149/+0.315/+0.377`. Along the recorded
  Adam update, the estimated full-score slopes are
  `+30.12/-16.02/-49.13/-11.37`, G3 remains positive at
  `+23.35/+15.91/+15.88/+16.29`, and fixed-G4 is negative at
  `-8.49/-1.56/-8.70/-2.60`. Thus batch shape materially changes magnitudes
  and makes 32-direction cosine estimates noisy, but does not remove the
  post-peak G3 sign contradiction or make fixed-axis G4 production-worthy.
  P107 four-GPU job `32449` is a clean 200-step rerun from the canonical
  `start_10` latent with the historically successful seed `2026080515`, and
  writes `runs/qh_g3_subspace_batch1_start10_adam_seed15_200_20260805`.
  Students two-GPU job `32451` independently runs 120 steps from the same
  start with seed `2026080524`, writing
  `runs/qh_g3_subspace_batch1_start10_adam_seed24_120_20260805`. Both use
  corrected independent batch-1 secants, G3+four random directions,
  `h=0.0025`, Adam `(lr=0.01,beta1=0.5,beta2=0.999)`, and the monotone exact
  gate. Both passed `sbatch --test-only`, started on idle RTX 5090s with
  2 MiB/0% preflight, and produced valid monotone progress. At the latest
  recorded checkpoint, job `32449` reached `91.9611` at step 91 (best first
  reached at step 82) and job `32451` reached `91.5791` at step 72; both remain
  active and neither has accepted a score decrease.
  P107 job `32462` is queued with `afterok:32449` for an 80-step zero-momentum
  refinement from that run's `best.json` using seed `2026080525` and the
  `d2545dc` fast gate plus the `9b1e6e4` logging fix. Students job `32465` is queued with `afterok:32459` for
  a 60-step refinement from job `32451`'s `best.json` using seed `2026080526`;
  this deliberately lets the three-step fast-path A/B run first. Both
  refinement jobs passed `sbatch --test-only` and use the same bounded
  `h/lr/beta` settings as their parent runs.
- Students job `32332` completed 124 zero-momentum refinement steps from the
  old strict-control best, improving `92.3317626 -> 92.6658409` at step 113;
  this is a staged restart, not one continuous Adam history. P107 job `32336`
  was deliberately cancelled after step 13 at `92.5651` because it plateaued.
  Restart smoke `32337` failed before computation because optimizer trajectory
  JSON was not accepted as an initial case; commit `fcdf335` fixes that
  interface. Corrected projected-mode smoke `32338` improved
  `92.5885592 -> 92.6218379`. Pre-fix quadratic jobs `32416/32417` completed
  at `92.6780695/92.6688750`; quadratic-axis job `32421` completed unchanged
  at `92.6658409`, and `32422` was deliberately cancelled after five unchanged
  steps. These four jobs are retained only as batch-shape-contaminated
  diagnostics and must not be used to validate the quadratic model.
- On 2026-08-05, branch `qh-blackbox-gradient` reached local/remote source commit
  `c5e8cfc` (report/MEMORY updates remain in progress). Commits `e581f45` through `2d75a69` implement the fixed-branch G4
  oracle, exact ABI-9 acceptance gate, fused CUDA field-point VJP, and exact
  same-basin G4 reference comparison; commits `48ed335` through `ee01069` add a
  G3-informed low-rank full-score secant optimizer and make smooth Adam and
  improving discrete-branch endpoints compete by exact score. The
  field-point VJP is validated to about `2.1e-4` centered-FD relative error at
  `h=2.5e-4` and costs about `0.244 ms` for 128 points and 4096 segments. Exact
  300-direction comparisons at start-10 steps `50/89/100/120` show fixed-axis
  G4/full-score cosines `-0.536/+0.045/+0.124/+0.314`; its slope along the
  recorded Adam update remains negative at every center, including before the
  old score peak. Axis response is both omitted and large, so fixed-axis G4 is
  not a production gradient and full analytic G5 is not justified for the
  current optimizer objective. The selected practical fallback uses the
  RMS-normalized G3 direction plus four orthogonal random directions, exact
  ABI-9 centered or feasible one-sided branch secants at latent RMS `h=0.0025`, Adam
  `(lr=0.01,beta1=0.5,beta2=0.999)`, and a monotone exact-score gate. A strict
  10-step run improved `85.8397 -> 87.7132`; K=0 and K=2 controls reached only
  `86.1435` and `87.5722`. A superseded branch-endpoint-first run reached only
  `87.4434`, confirming that branch endpoints must compete with rather than
  preempt smooth proposals. P107 four-GPU A/B job `32236` completed `85.8397 ->
  87.7132`, exactly matching the strict result; at step 2 it retained the higher
  smooth candidate (`87.4501`) over the lower branch endpoint. The first
  200-step seed-14 job `32238` was deliberately cancelled after step 36: strict
  two-sided branch matching left zero usable directions on 17/36 steps and
  rejected 20/36 steps, stalling at `88.5832`. This is a valid negative result,
  not a crash. Commit `d3bdaf3` adds feasible one-sided same-branch secants and
  lets any improving probe endpoint compete with Adam; a five-step smoke from
  the formerly stuck step-30 center recovered 3/5 directions and improved
  `88.5832 -> 88.5971` without accepting a decrease. Commit `d970383` fixes the
  Slurm idle gate so stale utilization telemetry cannot override the stronger
  empty compute-process/2 MiB memory evidence. Strict-control Students job
  `32240` reached `92.3318` at step 56, then was deliberately cancelled after
  step 76 because it made no further progress for about 20 steps; its partial
  trajectory is retained as a control. Evidence at that center showed a
  same-branch probe gain of `0.0057`, so commit `8e0a043` separates the
  production thresholds: `0.001` for same-branch probes and `0.01` for
  branch-changing endpoints. P107 one-sided job `32300` was deliberately
  cancelled after step 57 at `90.8667`; it used the superseded common `0.01`
  probe threshold and had low marginal progress, so it is retained only as a
  partial control. Students two-GPU job `32332` is an active documented
  zero-momentum refinement from job 32240's fixed step-56 best and writes
  `runs/qh_g3_subspace_one_sided_refine_seed15_124_20260805`; by step 24 it
  reached `92.5886`. P107 four-GPU job `32336` is an active independent
  zero-momentum refinement from the Students step-12 snapshot (`92.5369`) and
  writes `runs/qh_g3_subspace_high_refine_p107_100_20260805`; by step 13 it
  reached `92.5651`. These are multi-stage refinements and must not be reported
  as one continuous Adam trajectory. Code audit found that coordinatewise Adam
  preconditioning maps the exact low-rank secant projection outside the
  measured subspace, particularly reducing to a sign-like update after moment
  resets. Commit `c5e8cfc` therefore adds an opt-in `projected` trust proposal
  that stays parallel to the measured projection and has RMS radius equal to
  the secant perturbation, with the same bounded exact-score gate; the two
  active jobs still use the old `adam` proposal and are controls. This new mode
  passed the local related tests (`11/11`) but has not yet had a GPU numerical
  validation. Detailed evidence is being appended to
  `reports/qh_blackbox_gradient_exploration_report.md`.
- On 2026-08-05, a source-level theoretical audit established the scope of G4.
  On a fixed regular branch (same axis candidate, selected surface/flux
  candidate, simple unclamped ray roots, fixed valid-point/index set, stable
  active extrema/quantiles, and well-conditioned ridge LS), the downstream
  score is a composition of smooth maps and implicit solves; its local total
  derivative exists and G4 is theoretically feasible. There is no fundamental
  derivative barrier in psi/flux LS, simple level roots, moving coordinates,
  weights, B, grad-B, or fixed-geometry alpha/iota. The largest missing CUDA
  primitive is the material derivative of grad-B at moving volume points,
  requiring a Biot--Savart Hessian-vector contraction without materializing a
  full Hessian. The global ABI-9 score remains nondifferentiable at axis or
  surface candidate switches, hard rejection thresholds, root changes/clamps,
  valid-mask/CUB-compaction changes, integer counts, and score kinks. Therefore
  G4 must return a branch-conditional gradient plus branch margins; it cannot
  replace exact-score acceptance/backtracking. G4a should first add psi/flux
  implicit VJPs and simple-root response; G4b should then add moving points,
  coordinate bases, weights, and the field Hessian-vector path. Axis response
  remains separate G5, and surface-trace differentiation is optional pending
  evidence. Full derivation, difficulty assessment, and validation gates are in
  section 15 of `reports/qh_blackbox_gradient_exploration_report.md`. The
  report and this conclusion are committed as `1a30454` and synchronized to
  the remote `qh-blackbox-gradient` worktree.
- On 2026-08-04, the post-peak G2 reversal investigation completed. The earlier
  200-direction reference did not sample the current `start_10` basin: it
  covered nfp6/nc2 steps 0/200/400 and a separate nfp4/nc2 step 50, whereas the
  reversing trajectory is nfp4/nc3. Do not cite the old high cosines as
  same-basin evidence. Three independent closures now establish that G2 is the
  correct derivative of its frozen-front scalar, G3 correctly adds the
  fixed-geometry alpha/iota FP32-QR response, and the FP32 RK4-64 flow VJP does
  not reverse or attach the cotangent to the wrong point. The actual method
  error is treating this frozen partial derivative as the complete ABI-9 score
  derivative: omitted psi/flux/moving-surface/volume-point/weight response
  reverses even the volume-QS derivative after the peak. A separate confirmed
  optimizer bug accepts the first `status=ok` candidate without requiring an
  exact-score improvement, so the biased direction accumulates for dozens of
  steps. Detailed evidence and countermeasures are in report section 14.
  Validated report, analysis code, and curated assets are archived in commit
  `f409daa`, synchronized to the remote `qh-blackbox-gradient` worktree. At
  finalization `squeue` contained no project job, no score/reference worker
  process remained, and all four GPU postflight rows were idle at 2 MiB/0%.
  Commit `25d0bc3` implements the opt-in frozen-front closure oracle, current
  physical-gradient trajectory loading, component-gradient reference output,
  and Slurm launchers; local Python/reference tests pass `10/10`. P107 smoke
  job `31985` completed `0:0` in `00:02:38` for nfp4/nc3 step 120 with 8
  random directions and four latent RMS scales. It built isolated gradient
  library SHA-256 `2f3c48...0162`. The frozen scalar reproduced the center
  score to `4.8e-7`; frozen-FD versus native physical G2 cosines were
  `0.9973--0.9980`, physical-secant versus flow-VJP predictions were
  `0.999997--0.9999999`, and frozen-FD versus latent predictions were
  `0.9973--0.9980`. Along the actual step-120 Adam tangent, G2 predicted
  `+16.82`, the frozen scalar measured `+15.91--+16.43`, and its frozen
  volume-QS component measured about `+38.0`, whereas the independently
  measured complete-score slope was negative. This rules out a G2 sign error
  and a flow-VJP sign/attachment error at step 120; the contradiction is now
  localized to dependencies recomputed by the complete score but frozen by
  G2. Artifacts are in
  `runs/qh_g2_fixed_front_closure_smoke_20260804/`. P107
  jobs `31996--31999` completed `0:0` for steps `50/89/100/120`, respectively,
  using 32 random directions each and the pinned diagnostic-library hash above.
  At the smallest scale, frozen-FD/native-G2 cosines were
  `1.0000/1.0000/0.9995/0.9960`, while physical-secant/flow-VJP closure stayed
  above `0.999995`; actual-Adam frozen-score slopes were all positive at about
  `+22.03/+16.28/+16.30/+16.43`. Outputs are
  `runs/qh_g2_fixed_front_closure_step{0050,0089,0100,0120}_20260804/`.
  Four-GPU P107 job `32001` completed the same four centers and a complete
  300-direction, two-scale (`0.005/0.0025`) exact-score/component reference;
  output is `runs/qh_g2_current_basin_reference_20260804/`. At `h=0.0025`, all
  four centers retained 300/300 same-branch directions. Full/G2 cosines at
  steps `50/89/100/120` were `+0.052/-0.118/-0.180/-0.305`; full/G3 cosines
  were `+0.238/-0.110/-0.160/-0.272`. Full-score projections on the actual Adam
  direction were `+37.60/-4.95/-20.99/-24.60`, while G2/G3 remained positive
  near `+16--23`. At `h=0.005`, step 100 independently remained negative;
  step 120 retained 298/300 branches and was not force-reconstructed. The
  4804 evaluations averaged `5.909 s` with `6.194 s` P95; pre/postflight was
  clean at 2 MiB/0% on all four RTX 5090s. Curated reference summary and NPZ
  SHA-256 values are `0984c0...60ba` and `8b47bd...27c` under
  `reports/assets/qh_g2_same_basin_diagnosis_20260804/`.
  Commit `8f8d39a` adds a second diagnostic-only scalar that freezes geometry,
  psi, volume points and weights but refits alpha/iota with the production FP32
  QR for every query; its center derivative should equal cumulative G1+G2+G3.
  It is synchronized remotely. Students job `32007` completed `0:0` in
  `00:02:26`; isolated library SHA-256 is `97933e...a38c`. At step 120 its
  eight-direction G3-frozen/native-G3 cosines were `0.9972--0.9983` and the
  physical-secant/flow-VJP cosines exceeded `0.999997`. Along the actual Adam
  tangent G3 predicted `+16.29`, the mixed scalar measured `+15.39--+15.90`,
  and fixed-geometry/refitted-alpha volume-QS measured about `+39.25`; hence
  the alpha/iota QR response does not explain the complete volume-QS slope
  `-28.46`. Artifacts are in
  `runs/qh_g3_fixed_geometry_closure_smoke_20260804/`. Students jobs
  `32009--32012` completed `0:0` for 32-direction G2/G3 closures at steps
  `50/89/100/120`. At the smallest scale, G3-frozen/native-G3 cosines were
  `1.0000/1.0000/0.9995/0.9958`; physical-secant/flow-VJP closure stayed above
  `0.999995`. Actual-Adam G3-frozen slopes remained positive at about
  `+23.36/+15.91/+15.89/+15.91`. Thus neither G2, G3, nor the flow VJP develops
  a sign reversal across the pre-peak/peak/post-peak trajectory.
  Students job `32027` completed `0:0` for the missing step-50 small-scale
  exact-score directional probe. Along the actual Adam tangent, complete-score
  slopes were positive at `+89.38/+44.17/+26.69` for scales
  `0.0003125/0.000625/0.00125`, providing the pre-peak contrast to the robustly
  negative post-peak step-120 slopes.
- On 2026-08-04, deterministic-bias investigation completed after the accepted
  G2-Adam sweep. Commit `8d43791` adds the diagnostic code and commit `61b5cd6`
  archives the final report, saved-trajectory analysis, figures, and short
  exact-score G2/G3 probe results. Existing trajectory evidence
  already proves this is not a dirty-gradient event: for the three complete
  lr=0.003 runs, from the score peak to the first surface-level change, every
  saved G2 dot actual-Adam-update is positive, while exact score decreases on
  `25/35`, `30/39`, and `30/39` transitions. Median consecutive-gradient
  cosines are `0.983/0.988/0.986`, gradient RMS stays roughly `0.082--0.089`,
  and exact score loses `0.380/0.426/0.422` before any branch change. Across
  the same interval axis, coordinate, and volume-QS all fall while surface and
  coil improve; the RK4-64 QH residual worsens smoothly from `0.01434` to
  `0.01737`. This supersedes any explanation that attributes the long smooth
  reversal only to dirty gradients or the later discrete surface drop. The
  code audit has found no global sign inversion in Adam, q-score derivatives,
  or flow VJP; the definite protocol error is that the optimizer accepted the
  first `status=ok` candidate without the exact-score acceptance/backtracking
  required by the original G2 design. G2 is also structurally biased because
  it omits axis/psi/surface/coordinate derivatives and freezes psi, volume
  points, weights, and fitted iota in the direct QS VJP. Short P107 jobs
  `31938` (step 89), `31937` (step 100), and `31939` (step 120) probe exact
  antithetic score along G2, cumulative G3, and the actual Adam tangent at
  three small latent RMS scales. They use the original experiment library
  SHA-256 `fdf142...0f8d4`, did not continue optimization, and wrote under
  `runs/qh_physical_gradient_bias_probe_20260804/`. All three completed `0:0`
  in `2:55/2:58/2:06`; all 54 endpoints were OK and stayed on the same branch,
  with clean 2 MiB/0% GPU postflights. G2/G3 latent cosines were
  `0.9845/0.9869/0.9893`, so G3 does not supply a material correction. At step
  120 the exact centered slopes along the actual Adam tangent were
  `-10.16/-8.23/-3.90` for latent RMS scales
  `0.0003125/0.000625/0.00125`, while G2/G3 predicted `+16.82/+16.29`.
  Therefore the missing smooth G4 geometry dependencies, not G3 alone, are the
  next physical-gradient target. Detailed evidence and the countermeasure
  ranking are in report section 13. Immediate correctness priority is to
  restore the intended contract: G2 only proposes, exact ABI-9 score applies a
  noise-aware trust/acceptance gate and always preserves best state. Preferred
  short-term correction is antithetic verification along the proposal plus a
  low-rank control-variate correction in the span of that direction and a few
  random orthogonal directions; K=4 SPSA remains the fallback. Do not use
  unselected zero-mean normal noise as a claimed debiasing method, because its
  expected direction remains the biased G2 field. No jobs remain active.
- On 2026-08-04, the physical-gradient Adam sweep was accepted without any
  rerun or new job. All formal jobs `31855--31869` are terminal: only the three
  `lr=0.003` jobs completed 200 steps; the other 12 failed after 8--71 recorded
  steps because the old G2 Python/C ABI treated a valid non-OK complete score
  (`no_axis`, `no_surface`, and related statuses) as a backend exception before
  the optimizer could backtrack. Historical scores before each failure remain
  exact ABI-9 score evaluations, but these runs must not be described as
  completed 200-step experiments. The highest new historical score was
  `90.6927323` at step 26 for RK4-256/lr=0.01, still `2.4728275` below the old
  four-direction baseline `93.1655597`. RK4-64/128/256 with lr=0.003 peaked at
  `90.6273/90.6453/90.6456` and then all collapsed to about `80.85`; each best
  used `surface_level=0.25`, while each final state used `0.08`. This validates
  G2 + flow VJP as a fast short-range direction but also validates the expected
  fixed-front limitation: it does not differentiate axis/psi/surface branch
  reselection and is unsafe for long unconstrained Adam trajectories without
  an exact-score acceptance/trust mechanism. RK4 step count had little effect
  on peak score; learning rate dominated, with lr=0.01 giving the best short
  trajectory. Completed per-step means were `7.24/8.09/10.65 s` for
  RK4-64/128/256, or `3.68x/3.30x/2.50x` faster than the old 26.65 s baseline.
  All postflight snapshots were clean at 2 MiB and 0--2% GPU utilization; no
  jobs or zombie processes remain. Full evidence is report section 12 and
  `reports/assets/qh_physical_gradient_adam_start10_sweep_20260804/`; every raw
  trajectory is also preserved remotely under the same run name. Local commit
  `41b0a1d` changes G2 so valid non-OK scores return to the caller with zeroed
  gradients and passed `11/11` local tests, but it has not been synced, rebuilt,
  or validated remotely and did not contribute to any accepted number. Do not
  claim that this interface fix solves the observed post-peak drift. No full
  physical evaluation was run because the new maximum did not beat the already
  fully evaluated score-93 baseline and the user explicitly requested no new
  jobs. Acceptance report, analysis script, aggregate tables, and figures are
  archived in local commit `e813f01`; the branch has not been synced to the
  remote after this acceptance.
- On 2026-08-04 (historical submission record, superseded by the acceptance
  entry above), the new physical-gradient Adam sweep was active on branch
  `qh-blackbox-gradient` at commit `8aca1dc53`. The fixed comparison object is
  the main-document nfp4/nc3 score-93 case: the original IID `start_10` latent
  in `reports/assets/qh_score_adam_start_panel_29960/start_10.json`, source case
  2912. Its canonical float32 latent SHA-256 is
  `bf3c7b5ed577f6e5690d83e685a7873150522060268096abb46cd1cc1c4700ad`;
  use this logical hash rather than the JSON file hash, which differs between
  CRLF and LF worktrees. The historical job-31058 baseline started from this
  latent with zero Adam moments and four antithetic black-box directions under
  FP32 RK4-256, improving `85.8832483 -> best 93.1655597 at step 197 -> final
  93.1601644` in 200 steps. The new optimizer always restarts from the same
  original latent, not the score-93 endpoint.
  Commits `1579044/8aca1dc` add a provider-based retained-activation flow VJP,
  `scripts/optimize_qh_physical_gradient_adam.py`, partition-specific one-GPU
  launchers, and the reproducible sweep submitter. Each state performs one
  compiled FP32 RK4 decode, calls the native fixed-front G2 physical gradient
  while the graph is retained, and VJPs that cotangent to latent space. The
  native forward score remains the exact ABI-9 C++/CUDA score; G2 is explicitly
  an approximate gradient, not an exact derivative of every score dependency.
  Every step saves latent, physical coil tokens, all score components and key
  diagnostics, physical/latent gradients, Adam moments, trials, and timing.
  Invalid candidates have bounded backtracking `[0.5,0.25,0.125]`; there is no
  score-monotonicity acceptance gate. The pinned checkpoint is SHA-256
  `39a3293a...`, and the active experimental gradient library is SHA-256
  `fdf142aad0f0e0739c61cf61fde9d7195a688079a26016a9c010571d9440f8d4`.
  Local VJP/reference tests passed `9/9` before submission.
  P107 smoke job `31854` completed `0:0` in `00:03:10` with clean 2 MiB/0%
  postflight. Under RK4-64 it improved `85.8605792 -> 87.3911770` in two
  accepted steps; steady iteration walls were `9.55/8.70 s`. Its complete
  artifacts are under `runs/qh_physical_gradient_adam_smoke/`.
  Formal jobs `31855--31869` cover RK4 steps `{64,128,256}` crossed with learning
  rates `{0.003,0.01,0.03,0.05,0.1}`, each for 200 steps. The authoritative job
  mapping is
  `runs/qh_physical_gradient_adam_start10_sweep_20260804/submitted_jobs.tsv`.
  At handoff, jobs `31855,31856` occupied the two allowed Students GPUs and
  `31860--31863` occupied four P107 GPUs; the other nine were queued behind the
  expected per-user GPU/job limits. All six running jobs had entered numerical
  iterations and written valid trajectories. This is intentionally an async
  handoff: do not resubmit while these jobs remain active. On the user's next
  acceptance request, collect all summaries, preserve every trajectory, plot
  all 15 score/component curves together, compare the RK4-256 runs directly
  with job 31058, append conclusions to
  `reports/qh_blackbox_gradient_exploration_report.md`, and only then decide
  whether a full physical evaluation is warranted.
- On 2026-08-04, the user's correction about RK4 step selection superseded the
  earlier blanket conclusion "keep RK4-256". Different step counts define
  different discrete maps `F_N` and therefore different self-consistent
  objectives `S(F_N(z))`; cross-step distance to RK4-256 tests interchangeability,
  not whether the VJP of `F_N` is attached to the wrong point. The VJP directly
  differentiates the actual discrete `F_N` graph and does not reverse-integrate
  before taking the gradient. Branch `qh-blackbox-gradient` commits
  `932cc91/cc57773` added the same-step closure benchmark and made every probe
  denominator use its own discrete map; commit `7bf84eb` added the validated
  assets and report section. Preliminary P107 job `31847` reused RK4-256 probe displacements
  only as the physical ratio denominator; its absolute closures remain valid,
  but its ratios are superseded and must not be reported. Replacement job
  `31850` recomputed all 200 antithetic probes separately under each candidate
  `F_N` and completed `0:0` in `00:01:07` on one idle RTX 5090, with 2 MiB/0%
  pre/postflight. Across the four saved optimization
  centers, worst `z -> F_N(z) -> B_N(F_N(z))` RMS relative to the formal
  `h=0.00125` probe was `1091/92.5/1.51/0.327/0.110%` for RK4
  `16/32/64/128/256`; worst physical `x -> B_N(x) -> F_N(B_N(x))` curve RMS
  was `903.8/74.64/1.137/0.301/0.0514 um`, or
  `169.8/14.0/0.2140/0.0566/0.0182%` of each center's same-step probe displacement.
  Historical three-QUASR closure results independently show the same sharp
  improvement from 32 to 64 steps. Operational conclusion: reject RK4-16/32;
  use FP32 RK4-64 for a self-consistent latent optimization, RK4-128 for more
  conservative physical-data inversion/round trips, and RK4-256 only for old
  artifact interchangeability or continuous-ODE reference comparisons. A
  trajectory must record and retain one `rk4_steps` value for inversion,
  optimization, persistence, and final decoding; mixing step counts really
  does switch objectives. The old strict cross-step measurements remain valid
  evidence only for interchangeability. Detailed evidence and plot are in
  report section 11 and `reports/assets/qh_flow_reconstruction_31850/`.
  Validated SHA-256 values are `419885d3...ad2d488` for `summary.json` and
  `2a89b6d9...f157a3b` for `same_step_reconstruction.png`.
- On 2026-08-04, branch `qh-blackbox-gradient` commits
  `90c789e/a595d80/700a4b3/9d1219e/2656285/0390ae5`
  added an opt-in flow-VJP performance experiment without changing the existing
  checkpointed default. It compares FP32 RK4-32/64/128/256 against RK4-256 at
  four saved optimization stages, validates physical geometry, current native
  G2 score/components/branch, fixed-cotangent and end-to-end latent VJPs, and
  measures checkpointed versus retained-activation latency and peak allocated
  memory. The launcher supports either four GPUs in one wave or two GPUs in two
  waves while keeping one process per GPU. Students job `31791` completed
  `0:0` in `00:09:50`; about five minutes were one-time profiler CPU event
  aggregation. Retaining all activations used at most about `0.47 GiB`
  incremental allocated memory and reduced RK4-256 flow forward+VJP median
  latency from `14.18 s` to `6.71 s` with checkpoint/retained gradients
  numerically consistent. RK4-128 retained excellent direction agreement but
  failed the strict forward-accuracy gate: worst position error was `18.4%` of
  the formal `h=0.00125` optimizer perturbation, score/component errors reached
  `0.0212/0.1145`, and worst end-to-end VJP cosine was `0.999281`. Output is
  `runs/qh_flow_vjp_benchmark_31791/` and local report assets are under the
  same basename. P107 four-GPU refinement job `31806` failed in 27 seconds
  before numerical work because Slurm parsed the commas in
  `--export=ALL,RK4_STEPS=160,192,224,256` as variable separators and passed
  only RK4-160; all workers correctly rejected the missing RK4-256 reference.
  It has no numerical result. Replacement `31809` completed `0:0` in
  `00:03:02` and showed that FP32 integration error is non-monotone: RK4-192
  was closer to RK4-256 than RK4-224 at all four centers, but no tested step
  count below 256 passed the strict geometry/score/component/VJP gate. Keep
  RK4-256; do not infer acceptable steps from fourth-order asymptotics alone.
  Job `31815` then failed because the model's per-call tensor-valued nfp check
  blocked full-graph compile. The integrators now validate nfp once and use the
  identical unchecked mathematical body; 13 local flow/model/VJP tests pass.
  This structural fix had no measurable eager speed benefit and is not claimed
  as one. First compiled replacement `31822` failed because
  `mode=reduce-overhead` CUDAGraph storage overwrote outputs still needed by
  RK4; it has no numerical result. Default Inductor replacement `31824`
  completed `0:0` in `00:02:27`, and same-Students eager control `31833`
  completed `0:0` in `00:02:07`. On RK4-256, eager checkpoint/retained medians
  were `17.36/8.06 s`, while compiled checkpoint/retained were `9.12/4.39 s`.
  Retained activations use at most about `0.50 GiB` incremental allocated
  memory and are now the default for the batch-1 `decode_physical_vjp`; the
  generic integrator remains checkpointed by default. Compiled versus eager
  latent-gradient minimum cosine was `0.9999999996`, maximum relative L2 was
  `3.0e-5`, and maximum score difference was `0.0186`; compile remains opt-in
  through `compile_flow_transformer()` / `--compile-flow-model` because its
  one-time compilation must be amortized and it is not bitwise equivalent.
  Same-condition steady-state speedup is `2.22x` from retained activations,
  `1.84x` further from compile, and about `3.95x` combined. One-time profiling
  observed 299,583 CUDA events, 2,854,825 total profiler events, and about
  `1.77 s` summed self-device operator time, confirming launch/framework
  overhead; event aggregation itself took about five minutes and must not be a
  routine benchmark. All successful jobs ended with every allocated GPU at
  2 MiB and 0% utilization. Detailed evidence is report section 10 and local
  assets
  `reports/assets/qh_flow_vjp_benchmark_{31791,31809,31821,31824,31833,31836}`.
  Final public-helper regression job `31836` completed `0:0` in `00:02:00`:
  its four center gradients were bitwise identical to the private-wrapper
  compiled benchmark, score/component differences were at `1e-14`, retained
  median was `4.38 s`, and both postflight GPUs were 2 MiB/0%. Commit
  `20ac604` is therefore the validated public execution interface.
  Report section 10 and all curated benchmark assets were committed as
  `b872936` and synchronized to the remote worktree.
- On 2026-08-04, the user clarified that this cluster's `sbatch --test-only`
  predicted start time is generally inaccurate and jobs commonly start
  immediately after real submission. Use `--test-only` only to validate request
  syntax/resources; use the real job's `squeue/sacct` state for scheduling
  decisions, and do not change partitions solely because of its start estimate.
- On 2026-08-04, commits `1f9e121/cddb07f` add a proposal-level acceptance
  experiment for the black-box-gradient branch. It normalizes the already
  validated G2 and G3b latent VJPs to unit RMS, decodes antithetic proposals at
  latent RMS steps `0.0025/0.005/0.01` with FP32 RK4-256, and evaluates all 48
  proposals with the pinned production ABI-9 score library. The ordinary score
  path remains unchanged. Students job `31754` completed all 48 cases: every
  result was `status=ok`, and the positive-gradient endpoint beat its
  antithetic negative endpoint in all 24 method/scale pairs. However, only
  12/24 positive endpoints beat the center; the mature step-400 center lost
  score on both sides even at RMS step `0.0025`, showing that direction quality
  and an acceptable trust radius are separate questions. Both postflight GPUs
  were 2 MiB/0%. Output is
  `runs/qh_blackbox_gradient_proposal_31754/`. Commit `92cd8c6` parameterizes
  the trust scales. Students backtracking job `31758` completed `0:0` in
  `00:03:09` at scales `0.0003125/0.000625/0.00125`: all 48 candidates were
  `ok` and same-branch, all 24 positive endpoints beat their antithetic
  negatives, and 23/24 beat the center. Step-400 recovered gains up to
  `+0.0957`, proving its first-run loss was excessive step size rather than a
  reversed gradient. The only negative was a `-0.0150` G2 score at step-200,
  scale `0.000625`, between positive adjacent scales and therefore consistent
  with a finite-scale score wrinkle. Both GPUs again ended at 2 MiB/0%.
  Future physical-VJP optimization must use true ABI-9 acceptance and trust-
  radius backtracking; no fixed latent step is valid across stages. These jobs
  verify exact score gain and branch retention; they do not run an optimizer.
- On 2026-08-04, the decision-grade 200-direction study is complete and
  supersedes all earlier 4-direction gradient impressions. At `h=0.005`, G2
  cosine across nfp6 steps 0/200/400 and nfp4 step 50 is
  `0.854/0.425/0.863/0.418`; G3b is `0.854/0.455/0.866/0.479`. Both greatly
  exceed same-bank random K=4 cosine near `0.13`; G2/G3b are equivalent to
  roughly K=64 or more. G3b's gain over G2 is only `0.0005/0.030/0.0033/0.061`,
  so G2 is the preferred experimental direction and G3b is not a justified
  default. G1 alone is weak. One latent physical VJP step costs about
  `17.4--19.8 s`; native reverse is only `0.12--0.16 s`, while FP32 RK4-256
  flow backward is `8.5--9.5 s`. Therefore pure VJP has no current wall-clock
  advantage over four-direction Adam. Full evidence and plots are in
  `reports/qh_blackbox_gradient_exploration_report.md`. Experimental APIs
  remain opt-in; production `sgpu_score_coils` and ABI-9 are unchanged.
- On 2026-08-04, a source audit clarified the apparent flow-VJP bottleneck.
  The validated path is batch 1 with two coil tokens through a 30.33M-parameter
  Transformer and RK4-256, hence 1024 strictly sequential velocity-model calls.
  It checkpoints every 8 ODE steps (32 chunks), so backward recomputes all
  1024 forwards and then propagates their activation VJPs. Parameters are
  frozen; no weight gradients are computed. The observed `8.5--9.5 s`
  backward versus `4.5--5.1 s` single-sample forward is therefore expected;
  the poor latency comes from tiny GEMMs/kernel launches and checkpoint
  recomputation, not saturated RTX 5090 arithmetic. Earlier fast decode numbers
  were batched throughput: 48 candidates decoded in `11.70 s`, while the old
  4096-latent pool took `38.34 s` total. Optimization priority is gradient-
  validated RK4-64/128, a no-checkpoint batch-1 benchmark if memory permits,
  then same-condition batched multi-start VJP and CUDA-graph/compile launch
  reduction. Detailed explanation is report section 9.
- On 2026-08-04, commit `ac99266` adds the opt-in cumulative G3 gradient path.
  G3 keeps the production `sgpu_score_coils` ABI/path unchanged and, only for
  `sgpu_score_coils_g3_gradient`, caches the existing FP32 augmented-QR factors
  and differentiates the alpha/iota ridge least-squares solution implicitly.
  It adds the resulting iota/QS, alpha-fit-residual, and normal-field adjoints
  to G1+G2 before the existing CUDA Biot-Savart and Fourier/current map. Eight
  local independent math/flow-VJP tests pass, including scaled ridge-LS,
  normalized alpha-weight, and alpha field-preprocess adjoints. At this
  intermediate point the CUDA path was not yet numerically accepted; the
  completed acceptance result is recorded above. Students job `31720`
  completed `0:0` in
  `00:04:03` on one idle-checked RTX 5090 with 12 CPUs. CUDA 13/sm120 build SHA
  is `ab8c069c...`; ordinary and all three gradient-entry forwards agree with
  the pinned production score/component values within `2.85e-14`. Median G3
  wall time was `5.2260 s` versus `5.0735 s` for the experimental ordinary
  forward (`+3.01%`); total G3 reverse was `0.1653 s`. However, on the same
  four preliminary complete-score directions, G2/G3 cosines were
  `0.568/-0.137`, and cumulative gradient RMS jumped from `22.75` to `396.65`.
  Four directions are not a final statistical comparison, but this is a hard
  diagnostic stop: do not use G3 for optimization until component-isolated
  and formal 200-direction checks determine whether the cause is a VJP bug or
  cancellation by frozen flux/psi/point-motion dependencies. Postflight had
  2 MiB allocated and no compute process; artifacts are under
  `runs/qh_native_g1_validation_31720/`.
  Component-isolated Students jobs `31722/31724/31726/31728` then located the
  explosion: coordinate-only G3 RMS was `3893.87`, almost entirely the
  normal-field term (`3882.84`), whereas alpha-residual-only was `11.94` and
  the iota-to-volume-QS increment only `0.0732`. The normal-field formula
  itself passes the independent autograd VJP; the bad proxy arises because its
  frozen-psi partial is amplified near zero normal field while the omitted
  psi/surface motion supplies the physical cancellation. Commit `2f446f7`
  therefore defers this geometry-covariant term as an inseparable G4 bundle.
  Replacement job `31731` completed `0:0` in `00:03:59`; build SHA is
  `fdf142aa...`, all forward values remain within `2.85e-14`, G3 RMS is now
  `23.11` versus G2 `22.75`, and four-direction G2/G3 cosines are
  `0.568/0.558`. This removed the numerical pathology but showed no preliminary
  G3 benefit; the formal 200-direction result is now complete above. G3b median
  wall/reverse times are `5.2165/0.1685 s`, and postflight was 2 MiB at 0%.
  Commit `d236a1f` parameterizes latent center/scale/direction selection in the
  common launcher and is synchronized remotely. Commits `74c3de4/38a1a33` add
  explicit safe-subspace reconstruction: a single branch-changing direction
  no longer turns an otherwise useful 200-direction reference into an
  unexplained NaN. It also bootstraps random-K subsets from the frozen
  orthogonal bank, explicitly labeled as bank reuse rather than independently
  scored random blocks.
- On 2026-08-04, the user granted the active black-box-gradient exploration
  continuing access to the `Students` partition's additional 2 RTX 5090 GPUs
  and 24 CPUs until the user explicitly revokes that permission. This is in
  addition to the existing P107 allowance. Foreground validation may therefore
  use Students without waiting behind P107 work, while all scheduler/account,
  one-GPU-per-process, idle-GPU timing, and cleanup requirements still apply.
- Formal reference job `31640` completed `0:0` in `02:21:23` on four P107 RTX
  5090 GPUs: all 6404 cases finished, mean/P95 score latency was
  `4.905/5.213 s`, and postflight was 2 MiB/0% on all cards. Dependent latent
  VJP jobs `31738/31740/31742/31744` also completed `0:0`; RK4-256 re-decoding
  relative L2 was `4.8e-14--1.13e-7`. The formal analysis and compact assets
  are preserved locally under `reports/assets/qh_blackbox_gradient_*` and
  `reports/assets/qh_native_g1_validation_3173*/3174*`.
- On 2026-08-04, branch `qh-blackbox-gradient` commit `0e877ae` adds a
  restartable multi-scale black-box reference-gradient driver, a four-GPU
  Slurm launcher, and a one-direction GPU smoke launcher. The production
  ABI-9 forward interface is unchanged. Reference preparation freezes latent
  centers, independent RMS-orthogonal banks, RK4-256 decoded physical coils,
  scales, source/checkpoint/library hashes, and branch fingerprints; four
  score shards append each result independently and can resume. The first
  formal batch is configured for nfp6 steps 0/200/400 and nfp4 step 50, all
  200 directions and $h=0.01/0.005/0.0025/0.00125$ (6400 endpoints plus four
  centers). Remote smoke job `31638` is currently submitted on P107 using
  checkpoint SHA `39a3293a...` and the validated corrected ABI-9 library SHA
  `40dca742...`. The main repository's current same-named build has SHA
  `d2cfcab1...` and is explicitly not used; the validated library is pinned at
  `~/local_surface_evaluator_worktrees/qh-volume-qs-g-fix/gpu_backend/build_native_score/libstellarator_gpu.so`.
  Smoke job `31638` completed `0:0` in `00:01:06`: all three cases were
  `ok`, the endpoint pair retained the center fingerprint, score latency was
  `6.213 s` mean / `8.007 s` P95, and the RK4-256 re-decoded center scored
  `74.42519` versus historical recorded `74.43583`. Formal four-GPU job
  `31640` is now submitted for the 6404-case batch; its output directory is
  `~/local_surface_evaluator_worktrees/qh-blackbox-gradient/runs/qh_blackbox_gradient_reference_31640/`.
  Experimental commits `85c5a43` and `6b7afe7` add separate opt-in G1 and
  cumulative G1+G2 APIs without changing `sgpu_score_coils` or either ABI-9
  struct. G1 analytically differentiates the active coil-engineering metrics.
  G2 freezes axis/psi/selected surface/volume points/weights/fitted iota and
  discrete branches, then applies a pointwise QH/QA/QP score VJP, a CUDA
  segment-level Biot-Savart VJP, and an analytical segment-to-Fourier/current
  map. These implementations are not yet numerically validated. P107 job
  `31642` was cancelled while still pending with zero allocated resources,
  after the continuing Students allowance made migration preferable. The first
  Students replacement `31655` failed before compilation in three seconds
  because the shared runtime venv has no `pytest`; no numerical or CUDA check
  ran. Commit `a442e13` separates native and latent validation, follows the
  allocated CPU count when compiling, and enforces an idle-GPU preflight. The
  launcher now invokes the five dependency-free test functions directly, so
  the missing optional test runner cannot block the actual Release CUDA build,
  24-direction G1 check, old/new forward benchmark, G2 timing, and four true-
  score direction checks. Latent validation remains disabled until formal
  reference job `31640` has completed. Replacement Students job `31659`
  passed all five standalone math/VJP checks and reached CUDA compilation, then
  failed in 43 seconds because `fmax(abs(nfp), 1)` selected a host-only integer
  overload inside a device kernel. This is an implementation compile bug, not
  numerical evidence; it is fixed by an explicit device-safe double-valued
  conditional. Students job `31662` then completed `0:0` in `00:03:51`: all
  five standalone checks and the Release CUDA build passed, and G1's 24-
  direction component check achieved median/P95 relative errors
  `4.36e-9/3.60e-8`. Its timing, forward-regression, and four-direction G2
  results are invalid for final comparison because the validation launcher
  accidentally built the experimental library for `sm_52`, while the pinned
  RTX 5090 baseline was built for `sm_120`. This explains the observed 49%
  forward slowdown and likely the small `0.00579` score drift. The launcher is
  corrected to the repository's established `CMAKE_CUDA_ARCHITECTURES=120`
  setting; only the same-architecture rerun may establish forward overhead or
  G2 agreement. Artifacts from the invalid-architecture run are retained under
  `runs/qh_native_g1_validation_31662/` and must remain labeled accordingly.
  A first `sm_120` replacement, Students job `31666`, failed during compilation
  in `00:01:04` because the launcher still selected system NVCC 12.0, which
  does not support `compute_120`; no numerical validation ran. The validated
  production build uses `/public/app/cuda/13.0/bin/nvcc`. The launcher now
  matches `scripts/slurm_build_native_score.sh` by fixing CUDA 13.0 compiler,
  toolkit, runtime, and architecture, and uses a fresh
  `gpu_backend/build_gradient_sm120/` directory so the old `sm_52` CMake cache
  cannot leak into the rerun.
  Corrected Students job `31672` completed `0:0` in `00:03:05` on an idle RTX
  5090. The experimental `sm_120` library SHA-256 is `d5a41836...`. Against
  the pinned baseline, the same input's score differed by only `1.42e-14` and
  all seven components agreed at floating-point roundoff. Median pure-forward
  times were `5.15026/5.15044 s` (baseline/experimental), so the measured
  normal-path overhead is `0.0036%`; G1 and cumulative G2 calls cost only
  `0.8%/2.44%` above the experimental forward. G2's explicit reverse phase was
  `0.1273 s`, including point/field/parameter-map phases of
  `0.00018/0.03039/0.05237 s`. The four-direction full-score smoke gave cosine
  `0.568` but only 50% sign agreement, which is too small a sample for a
  conclusion; formal 200-direction job `31640` remains authoritative. Future
  validation now hard-fails if status differs or score/component drift exceeds
  `1e-10`. Artifacts are under `runs/qh_native_g1_validation_31672/`.
  Two additional Students cross-case smokes then completed concurrently:
  `31680` tested nfp6 step 0 in `00:02:50`, and `31681` tested nfp4 step 50 in
  `00:02:29`. Both retained production forward results within `8e-14`, and G1
  component median relative errors remained `1.12e-8/2.25e-9`. Their four-
  direction G2-versus-full-score cosines were respectively `-0.278/0.819`,
  with only 50% sign agreement in each tiny sample. These results are
  preliminary variance diagnostics, not reference gradients: they show that
  fixed-front G2 usefulness can vary strongly by optimization stage and make
  the alpha/iota G3 dependency worth testing, while job `31640` remains the
  decision-grade 200-direction comparison. Artifacts are under
  `runs/qh_native_g1_validation_31680/` and `..._31681/`.
- On 2026-08-04, branch `qh-blackbox-gradient` was created from
  `qh-small-condition-adam` commit `4b14658`. Commit `c45c42e` adds the
  plan-only report `reports/qh_blackbox_gradient_exploration_plan.md`; no
  gradient implementation, compilation, remote job, or numerical experiment
  has started. The plan introduces cumulative gradient groups from score/cache
  instrumentation and coil engineering through fixed-front volume-QS,
  alpha/iota LS, s/psi geometry, and finally conditional axis dependence. Its
  primary validation set uses saved random-start Adam states across early,
  middle, mature, and feasibility-boundary scales on the nfp6/nc2 and
  nfp4/nc2 trajectories, rather than relying on inverse-QUASR near-optima.
  Follow-up commit d0034ee makes the metric protocol explicit. Smooth
  two-coil anchors use all 200 RMS-orthogonal antithetic directions to build a
  finite-scale black-box reference gradient, against which cumulative
  G1/G2/G3 physical gradients and repeated random-K estimates are compared by
  cosine. Hard branch cliffs and all-ok scale-inconsistent cliffs are detected
  separately; symmetric perturbations backtrack only to a recorded minimum
  scale, then persistent cliffs become active constraints rather than zero or
  silently discarded derivatives. Boundary points remain in a separate suite
  evaluated by safe-subspace cosine, feasible-step rate, backtracking, and
  exact score gain. Random direction counts K=1--64 use repeated independent
  blocks to report mean/std/quantiles and an equivalent black-box direction
  count with bootstrap uncertainty. No experiment has yet been run.
- On 2026-08-03, root `README.md` was comprehensively rewritten on branch
  `qh-small-condition-adam` by commit `fcab3ee`. It now treats ABI-9 native
  C++/CUDA scoring and sample-specific full physical evaluation as the two
  official paths; documents the current `coils -> axis -> s -> psi ->
  (alpha,iota)` split, native build/single/batch interfaces, fixed
  `evaluation/full_physical/` entrypoints, flow/low-momentum Adam usage,
  validated score/optimization results, and current feasibility-boundary
  limitation. The old Python CLI is explicitly retained only as a legacy
  research path. All 10 local Markdown links and 11 referenced command paths
  exist, code fences are balanced, `git diff --check` passes, and the fixed
  full-evaluation preflight passes under WSL with 20 files validated. Direct
  Windows execution of that Linux/Slurm preflight fails only because its Bash
  subprocess receives a Windows backslash path; this is not a README or
  production Linux workflow failure.
- On 2026-08-03, P107 four-GPU continuation job `31401` completed `0:0` in
  `01:44:16`, exactly extending the corrected ABI-9 `nfp=6`, two-base-coil
  state from iteration 400 to 700 under
  `~/local_surface_evaluator/runs/qh_nfp6_nc2_adam_continue700_20260803/`.
  The score improved from `86.1233491` at resume to a best `86.6414447` at
  iteration 491, then ended at `85.9452858`. Only 110 of the 300 continuation
  rounds applied an Adam update; 190 were skipped. After iteration 491 the
  antithetic endpoints increasingly crossed the `no_surface` boundary, and
  the final 67 rounds applied no update at all. Thus the run is operationally
  complete but does not prove a smooth-objective optimum: it reached a
  feasibility-boundary lock under the current `skip_entire_step` policy.
  The best native components axis/psi/surface/coordinate/volume-QS/iota/coil
  are `92.018/98.353/88.679/80.192/85.444/100/60.385`; its per-helicity QH
  error is `0.00539963`, iota `1.97317`, and selected native surface level
  `0.16`. Source state/best, flow checkpoint, and score-library SHA-256 values
  remain respectively `d490cf38...` / `34156733...`, `39a3293a...`, and
  `40dca742...`. Complete physical acceptance of iteration 491 is under
  `~/local_surface_evaluator_worktrees/qh-small-condition-adam/runs/qh_nfp6_nc2_adam491_full_eval_20260803/`.
  Sample-specific FP32 GPU source-psi jobs `31457--31460` completed `0:0`;
  `a=0.08` was selected for maximum tested coverage with validation RMS and
  angle-P95 `6.1418e-4/1.2636e-4`. Standard alpha+nu plus LS/Newton jobs
  `31465--31468` accepted a continuous, volume-increasing branch through
  `s=0.12/0.24/0.36/0.49`. At selected `s=0.49`, `|V|=0.0706051 m^3`,
  `iota=2.13315`, dense relative residual `8.6456e-6`, normal-field P95
  `1.1955e-5`, and face QA/QH/QP errors are
  `8.3154e-3/1.88224e-4/8.4723e-3`. Outer jobs `31472/31473` failed
  explicitly in the nu-coordinate invertibility check at `s=0.56/0.64`
  (minimum Jacobian `-0.002586/-0.410405`), establishing the nearest tested
  outer boundary without a CPU point-cloud fallback. Fixed-surface full
  evaluation job `31477` completed `0:0` in `00:11:02` on selected `s=0.49`.
  The direct vacuum field has `|B|=0.80877--1.10608 T` (mean `0.93200 T`),
  and all eight Poincare lines produced 29 hits at all four sections. DESC used
  the allowed CPU backend, stayed nested, and reduced normalized force
  mean/P95/max from `1.31003/2.18179/796.284` to
  `6.4162e-4/1.5648e-3/8.2683e-3`. It reached the 50-iteration cap, so
  `optimizer_success=false` must remain explicit despite the useful final
  residual. Relative to step 400, the step-491 quick native score is `+0.5181`
  higher and its volume-QH component is better, but independent accepted
  volume is `2.48%` smaller, face QH squared error is `6.33%` worse, and DESC
  final residuals are essentially tied. Therefore step 491 is physically
  valid but not independently better than step 400. Selected-surface and DESC
  equilibrium SHA-256 values are `464c1e73...` and `ef2f3366...`. Complete
  evidence and all eight DESC figures are in
  `reports/qh_small_condition_adam_report.md` section 13 and
  `reports/assets/qh_small_condition_adam_nfp6_nc2_continue700_20260803/`.
  Feature-branch delivery commit is `dd6e3af`.
- On 2026-08-03, P107 four-GPU continuation job `31330` completed `0:0` in
  `01:08:54`. It exactly resumed the corrected ABI-9 `nfp=6`, two-base-coil
  Adam state at iteration 200 and reached its best/final score `86.1233491` at
  iteration 400, versus `83.4688735` at resume and `74.4358335` before Adam.
  The final native components axis/psi/surface/coordinate/volume-QS/iota/coil
  are `92.170/98.394/88.371/80.471/83.685/100/62.937`; per-helicity QH fell
  from `0.01028325` at step 200 to `0.00657843` at step 400. The last 50-step
  running-best gain is still `0.7803`, so the trajectory improved but did not
  demonstrate saturation. All 400 history rows and 401 complete trajectory
  cases are preserved at
  `~/local_surface_evaluator/runs/qh_nfp6_nc2_adam_continue400_20260803/` and
  locally under
  `reports/assets/qh_small_condition_adam_nfp6_nc2_continue400_20260803/`.
  The rolling temporal guard rejected the step-390 RMS-`176.743` gradient
  without changing parameters or moments. Two large current-score drawdowns
  near steps 297 and 310 were instead discrete scorer branch switches from
  selected `surface_level=0.16` to `0.08`; the running best remained intact.
  The immutable score library SHA remains
  `40dca7422995a91eab0a58285d9ced59a8e3be04a96b2b37686effbe6f1abff5`.
  Complete physical evaluation of the step-400 best is finished under
  `~/local_surface_evaluator_worktrees/qh-small-condition-adam/runs/qh_nfp6_nc2_adam400_full_eval_20260803/`.
  Source-psi jobs `31354--31357` all completed `0:0`; sample-specific `a=0.08`
  was selected for maximal tested coverage with 389,440-point FP32 GPU-QR
  validation RMS/angle-P95 `6.8323e-4/1.2609e-4`. P107 candidate jobs
  `31362--31365` completed `0:0`: standard alpha+nu plus LS/Newton accepted
  the continuous, volume-increasing `s=0.12/0.20/0.30/0.36` branch, reaching
  `|V|=0.0588090 m^3`, `iota=2.11238`, dense relative residual `1.4419e-5`,
  and face QH error `1.3137e-4` at `s=0.36`. Outer jobs `31371/31372`
  completed `0:0` and retained the branch at `s=0.42/0.49`; selected `s=0.49`
  has `|V|=0.0723990 m^3`, `iota=2.11837`, dense relative residual
  `1.9568e-5`, normal-field P95 `2.6870e-5`, and face QA/QH/QP errors
  `8.3676e-3/1.7702e-4/8.6257e-3`. Outer jobs `31373/31374` failed
  explicitly at `s=0.56/0.64` because the alpha-derived toroidal correction
  became non-invertible (minimum Jacobian `-0.07924/-0.14037`), establishing
  the nearest tested outer boundary. Fixed-surface full-evaluation job `31382`
  completed `0:0` in `00:05:51` on selected `s=0.49`. All eight Poincare
  lines produced 29 hits at all four sections. DESC used the allowed CPU
  backend, remained nested, converged successfully by `xtol` in 27 iterations,
  and reduced normalized force mean/P95/max from
  `1.03490/2.33249/272.860` to
  `6.8506e-4/1.5589e-3/8.3121e-3`. The direct surface QH squared error is
  `1.7702e-4`, or `1.33%` RMS amplitude; visible contour curvature therefore
  remains real and is not contradicted by its approximately sevenfold RMS
  advantage over QA/QP. Selected-surface and DESC-equilibrium SHA-256 values
  are `3d9b9de26cab4f01e8bd0c3d550b87b72c59fa6a778893947c3af6515e0f451a`
  and `859a2a308c516c7a30b69eab602a4405770572ffbb31610ed8918250f25ffe3b`.
  Output is under
  `.../qh_nfp6_nc2_adam400_full_eval_20260803/selected_s0p49_full/`; evidence
  and all eight DESC figures are in `reports/qh_small_condition_adam_report.md`
  section 12 and the local continuation asset directory. Feature-branch
  delivery commit is `414f59e`. Local launcher commits are
  `5ee403a` and `f92dcff`; corresponding remote commits are `d33f57d` and
  `502b226`.
- The apparent mismatch between the step-200 `nfp=6` direct Boozer contour plot and its
  surface QH metric is mostly a scale-reading issue, not evidence that the plot
  is already high-quality QH. `helical_qs_metric` reports an area-weighted
  relative *squared* error. Thus QH `2.3357e-4` corresponds to `1.53%` RMS
  amplitude, while QA/QP `6.1462e-3/6.5015e-3` correspond to `7.84%/8.06%`;
  QH is about five times better in RMS amplitude, not orders of magnitude. The
  dominant contour slope follows the expected
  $\theta-N_{\rm FP}\phi=\mathrm{const}$ direction, but visible curvature and
  closed islands remain physically meaningful non-QH content. This agrees
  with the native volume-QS component being only `78.777/100`. Step 400
  improves the corresponding face QH squared error to `1.7702e-4` (`1.33%`
  RMS) and native volume-QS to `83.685/100`, but the improvement remains
  finite and does not make the contours ideally straight.
- On 2026-08-03, complete physical evaluation of the corrected ABI-9
  `nfp=6`, two-base-coil Adam best case was completed on branch
  `qh-small-condition-adam`. The immutable input is
  `runs/qh_nfp6_nc2_screen128_adam200_20260803/seed_2026080360/adam/best.json`
  with SHA-256
  `59c1efd068ecdf0e339f882b1a055c55c86e35ea75ba5aa26cbe1d321ddc4f0`.
  Source-psi jobs `31262/31264/31266/31268` for sample-specific
  `a=0.04/0.05/0.06/0.08` all completed `0:0`; `a=0.08` was selected because
  it provides the largest tested physical coverage while retaining FP32-GPU-QR
  validation RMS/angle-P95 `1.1602e-3/1.9009e-4` over 389,440 training points.
  Standard alpha+nu plus LS/Newton candidates at
  `s=0.12/0.16/0.20/0.24/0.30/0.36` all passed on a continuous,
  volume-increasing branch. Initial outer `s=0.49/0.64` jobs
  `31294/31296` stopped at the fixed 180,000-point budget with only
  160,480/126,576 valid GPU-ray points; their failed outputs were preserved
  under explicit `*_os1p25_failed` names. GPU-ray-only oversampling 2.0 jobs
  `31298/31300` then established the physical boundary without changing the
  LS budget: `s=0.49` produced a standard rejected summary and collapsed to
  mean fitted `s=0.0408`, while `s=0.64` failed the toroidal-coordinate
  invertibility check with minimum Jacobian `-0.08353`. The largest accepted
  surface is therefore sample-specific `s=0.36`, with
  `|V|=0.0625544 m^3`, `iota=2.3003895`, dense relative residual
  `4.0804e-5`, normal-field P95 `5.3022e-5`, and face QA/QH/QP errors
  `6.1462e-3/2.3357e-4/6.5015e-3`. Its alpha+nu initial relative residual was
  still `8.36e-2`; standard Simsopt LS, not alpha+nu alone, supplied the final
  high accuracy. Fixed CPU-P107 downstream job `31302` completed `0:0` in
  `00:05:26`; Poincare has 29 hits for each of eight lines at all four plotted
  sections, and DESC stayed nested while reducing normalized force
  mean/P95/max from `1.02814/2.24897/157.661` to
  `1.09736e-3/2.61703e-3/1.04420e-2`. DESC converged by `xtol` with
  `success=true`, cost `1.86165e-4`, and optimality `1.0690e-8`. Selected
  surface SHA-256 is
  `b0239e29c3b8cd73d89b8e355811a7878dce6b908a74f3c0f4c93cb1e50e9886`;
  equilibrium SHA-256 is
  `557168e9c14dcdc204599146389cd52edec203928fccfc07dfcaa0b97091d957`.
  Evidence is in `reports/qh_small_condition_adam_report.md` section 11 and
  `reports/assets/qh_small_condition_adam_nfp6_nc2_20260803/`; delivery
  validation references all eight successful DESC PNGs. Remote outputs remain
  at
  `~/local_surface_evaluator_worktrees/qh-small-condition-adam/runs/qh_nfp6_nc2_full_eval_20260803/`.
  Feature-branch delivery commits are `ad38905` (`Add nfp6 complete physical
  evaluation`) and `176c023` (`Record nfp6 evaluation delivery`). The same
  content is integrated on local `main` by commits `848fb5e` and `31497e0`.
- On 2026-08-03, local `main` contains non-fast-forward merge commit
  `2e83d21` (`Merge native QH scoring and latent optimization`). The merge
  preserves the original Simsopt LS/Newton and DESC route while adding the
  ABI-9 native QH score, flow-matching tooling, robust latent Adam, complete-
  evaluation entrypoints, and the methodology/future-direction documents.
  Post-merge interface inspection confirms score ABI 9, corrected
  $G=\mu_0I_{\rm link}/(2\pi)$, explicit per-helicity diagnostics, physical
  volume weights $Rr_b^2$, and the fixed 100000-point volume budget. The nfp6/
  nc2 result and its report are applied to `main` at `9910ec9`; the corrected
  ABI-9 landscape report/assets are applied at `4dd2db2`. Feature branch
  `qh-small-condition-adam` records the same landscape delivery at `4cbe76c`.
  The full merged test suite passes with `134 passed`.
- On 2026-08-03, commits `701b6e7`, `7fb5214`, and `96453fb` prepared
  the current paradigm for mainline delivery. The landscape launcher now pins
  the validated 30k physical-loss flow checkpoint and corrected ABI-9 score
  library by SHA-256, records both hashes and the score definition, and labels
  plots as corrected ABI-9 rather than the ambiguous old `score v3`; its seven
  focused tests and shell syntax check pass, and remote `sbatch --test-only`
  accepts the four-GPU job. `docs/QH原生评分与潜空间优化方法.md` is the
  rigorous Chinese methodology/experiments document for the native score,
  alpha+nu full-evaluation branch, flow matching, and latent Adam.
  `reports/qh_future_directions_feasibility.md` separately analyzes the
  validated proxy results, conditional Reflow, and component-wise approximate
  gradient/VJP feasibility. README links these documents while preserving the
  original LS/Newton/DESC route. The nfp6/nc2 and corrected ABI-9 landscape
  sections are now both filled from their formal runs.
- On 2026-08-03, P107 job `31233` completed `0:0` in `00:29:21` after completed
  dependency `31227_0`, so the two formal four-GPU jobs did not overlap. It
  reran the
  three-reference/four-direction landscape with FP32
  RK4-256, the validated checkpoint SHA
  `39a3293a459e248a0d1ec062607a1a467128b14d8ca973aadd82e113532ab99f`,
  and corrected ABI-9 library SHA
  `40dca7422995a91eab0a58285d9ced59a8e3be04a96b2b37686effbe6f1abff5`.
  Its output root is
  `~/local_surface_evaluator/runs/qh_flow_landscape_abi9_20260803`. All 1095
  unique scores and 1128 logical points are complete. Relative to independent
  random data-space directions, the latent drop-5 coordinate/physical-radius
  width ratios have medians `8.630/3.427`, both wider in 12/12 directions;
  the second-derivative RMS ratio median is `0.257`, smoother in 11/12.
  Relative to the matched flow-Jacobian tangent, corresponding ratios are
  `0.951/0.973` and roughness ratio `1.0005`, so the benefit is learned
  correlated directions rather than coordinate scaling. Latent/tangent/random
  `status=ok` rates are `74.73/75.54/52.69%`. FP32 RK4-256 round-trip position
  RMS is `2.26e-8`--`4.57e-8 m`; QH reconstruction differences are below
  `1e-8`. Four GPUs were 0%, 2 MiB with no compute process before and after.
  Evidence is in `reports/qh_flow_landscape_report.md` section 10 and
  `reports/assets/qh_flow_landscape_abi9_31233/`.
- On 2026-08-03, commit `b3b5223` on branch `qh-small-condition-adam`
  fixes the all-`ok` cross-step dirty-gradient failure in standard latent Adam.
  The optimizer now defaults to a rolling, scale-invariant median/MAD guard
  over the latest 20 accepted gradient and actual-update RMS values after a
  20-step warmup. A candidate exceeding either adaptive limit is rejected
  before center decoding/scoring; parameters, Adam step, first moment, and
  second moment are all left unchanged. The guard never uses a fixed absolute
  cap and can be explicitly disabled only with `--no-temporal-scale-guard`.
  Causal replay of the saved two-coil history preserves the beneficial step
  184 and rejects step 185 (`gradient RMS 337.109 > 39.320`, proposed update
  RMS `0.042175 > 0.017850`). It also identifies the small score-losing step
  170 as a marginal gradient-scale outlier. The correct anomalous antithetic
  pair at step 185 is `85.3298124/72.1562425`; the previously recorded pairing
  with `85.7874` was wrong. Full local validation is `134 passed`.
- On 2026-08-03, formal P107 job `31227_0` completed `0:0` in `01:10:42`.
  With `nfp=6`, two base coils, 128 IID starts and the corrected ABI-9 score,
  the selected start rescored at `74.43583` and 200 low-momentum Adam rounds
  reached their best/final `83.46887` at step 200. QH per-helicity error fell
  from `0.0381162` to `0.0102832`, iota remained `2.08426`, volume-QS rose
  `55.8315 -> 78.7771`, and coil rose `58.5193 -> 60.0925`; the gain is not a
  low-iota or size cheat. The new temporal guard rejected exactly one
  multi-direction contaminated step 167 before center decoding
  (`gradient RMS 182.669 > 18.464`, proposed update RMS
  `0.05473 > 0.03326`) while 199 normal updates applied. All 200 history rows
  and 201 trajectory cases are complete; four GPUs were 0%, 2 MiB with no
  compute process both before and after. End-to-end time was `4232.75 s`.
  No full alpha+nu/Simsopt/DESC evaluation was run for this case. Evidence is
  in `reports/qh_small_condition_adam_report.md` section 10 and
  `reports/assets/qh_small_condition_adam_nfp6_nc2_20260803/`.
- On 2026-08-03, four-GPU smoke job `31223` completed `0:0` in `00:06:53`
  and verified the new temporal-guard artifact schema and cleanup on the remote
  branch tip `fcc5297`. Its deliberately small eight-case screen found no
  `status=ok` start, so all 22 Adam rounds were safely skipped; this is control-
  flow evidence only, not optimization evidence; the formal evidence is job
  `31227_0` summarized immediately above.
- On 2026-08-03, branch `qh-small-condition-adam` completed the requested single
  smaller-condition experiment at `nfp=4`, two base coils. Implementation
  commits `dd62ab9`, `988d115`, and `eb6901e` respectively preserve every Adam
  step's complete coil/noise/native-score state, plot the true coil-score--QH
  trajectory, and expose GPU-ray candidate oversampling while preserving the
  old default 1.25 exactly. Delivery commit `312c735` versions the report and
  all 201 trajectory snapshots plus complete-evaluation artifacts. Four-GPU
  job `31148` completed `0:0` in `01:00:08`: the 128-IID screen selected case
  43 at corrected score `78.83857`; low-momentum Adam re-scored it at
  `78.84175`, reached `85.77307` at step 184, and ended at `85.10663`, with
  199 applied updates. Candidate/Adam/end-to-end times were
  `138.72/3460.78/3599.50 s`. Best axis/psi/surface/coordinate/volume-QS/iota/
  coil components are `93.435/97.531/97.235/87.599/77.069/100/72.799`; native
  `iota=1.47284`, and QH/QA/QP per-helicity errors are
  `0.0136752/0.131211/0.0297222`. All four GPUs were idle before formal timing
  and returned to 0%, 2 MiB with no compute process after it.
  A post-delivery audit confirms a dirty-gradient event when producing step
  185 from the step-184 optimum. One antithetic pair scored `85.3298/72.1562`
  and produced directional delta `13.1736`, accounting for 95.5% of the four
  deltas' squared energy. Gradient RMS jumped to `337.109` (67.1x the preceding
  20-step median), update RMS to `0.042175` (18.3x local median), and score fell
  by `1.06890`. The within-step median/MAD limit rose to `15.666`, so it did not
  flag the direction; all points remained status-ok, so invalid-center
  backtracking also did not apply. Momentum carried the damage into steps
  186--187, and step 200 remained `0.66643` below the best. Saved data cannot
  distinguish a score evaluation burr from a true local cliff, but either is
  an unreliable finite-difference gradient. Future robust Adam should use a
  rolling cross-step median/MAD guard on gradient/update scale and roll back
  both moments when rejecting such a step; never use a fixed gradient cap.
  Complete evaluation independently selected sample-specific source `a=0.08`
  (389,440-point FP32 GPU QR, validation RMS/angle-P95
  `6.828e-4/1.166e-4`). Standard alpha+nu plus LS/Newton accepted the continuous
  `s=0.12--0.49` sequence, selecting `s=0.49` with
  `|V|=0.0671413 m^3`, `iota=1.5270221`, dense relative residual
  `3.8360e-5`, normal-field P95 `5.7532e-5`, and surface QA/QH/QP errors
  `5.4361e-3/4.6577e-5/5.4660e-3`. Initial `s=0.64` attempts exposed that
  `grid_xy` is irrelevant to the GPU-ray backend; job `31186` used the fixed
  oversampling control at 1.6 and solved numerically, but was correctly rejected
  because volume decreased to `0.0531968 m^3` and fitted mean `s` collapsed to
  0.383, proving an inner-branch jump. CPU-DESC job `31188` completed `0:0` in
  5:30, remained nested, and reduced normalized force mean/P95/max from
  `1.6651/2.7813/5.3482` to `0.0039577/0.0086370/0.024939`; optimizer
  `success=false` means only the 50-iteration cap. Selected-surface SHA-256 is
  `8b1f4b3e43918f7a6d6f0c187a23ac669fcd4fbdf79be7696c5f0cf246854eed`;
  DESC-equilibrium SHA-256 is
  `3b89c6b3056966128ecaff4684a53af41431c8b2bef1810b547af9d0665655e0`.
  Relative to the prior three-base-coil score-93.166 solution, coil score
  improves by 7.48 and accepted volume by 4.92%, while native/face QH errors
  are about 5.95x/7.14x worse. Thus two coils give a validated, more
  engineering-friendly QH solution but not a higher total score in this run.
  Full evidence is in `reports/qh_small_condition_adam_report.md` and
  `reports/assets/qh_small_condition_adam_nfp4_nc2_20260803/`. Delivery
  validation references all eight successful DESC PNGs; all 201 trajectory
  schemas pass explicit coefficient/noise/score checks; local suite is
  `132 passed`.
- On 2026-08-03, commit `07deab9` added the corrected-score Adam
  `score-QH` landscape and fixed an overly strict complete-evaluation sampling
  gate without changing production native-score defaults. The final plot
  overlays all 583 QUASR and 465 random-flow `status=ok` calibration cases with
  six complete 200-step trajectories: score, iota, QA, QP, QA/QH, and QP/QH
  versus QH. Do not restore the incomplete endpoint-only coil/surface panels;
  history did not preserve per-step components, latents, or decoded coils.
  The best score/QH point is `93.16556/0.00230032`, with stable `|iota|` near
  1.6. Among status-ok QUASR cases its score, QH empirical CDF, coil, and surface
  percentiles are P88.3, P47.5, P1.5, and P96.7, so the high score carries a
  clear coil-engineering cost. Cases rejected before volume-QS have no honest
  QH coordinate and are omitted from these scatter panels, while remaining in
  the full status-distribution figure.
- The same sample's complete physical evaluation is accepted. It found that
  `s=0.49/0.64` had 209,413/181,980 valid GPU-ray candidates, both enough for
  the fixed 180,000 alpha train plus validation budget, but the generic
  production sampler's separate 95% gate rejected them before LS/Newton.
  `VolumeQSConfig` retains 0.95 by default; only the maintained full-evaluation
  launcher sets the extra fraction to zero and still requires the complete
  fixed point budget. It records candidate count/fraction and never falls back
  to `legacy-cartesian`. Uniform-code jobs `31119/31121/31123/31125` completed
  `0:0` in 3:21--3:25 with clean idle GPU pre/postflights. Standard LS/Newton
  accepted the continuous `s=0.24/0.36/0.49` sequence; `s=0.64` formally
  solved but was rejected as an inner-branch jump because volume decreased.
  Selected `s=0.49` has `|V|=0.06399216 m^3`, `iota=1.68782777`, dense relative
  residual `2.6514e-5`, normal-field P95 `4.2959e-5`, and surface QH error
  `6.5239e-6`; Poincare passed. CPU-DESC job `31135` completed `0:0` in 4:46,
  stayed nested, and reached final normalized force mean/P95/max
  `0.0023306/0.0048506/0.0174893` at its 50-iteration limit. Selected-surface
  SHA-256 is `794751c7dec47ce021d273cef4a6d700e06d71949c80683426b7b596d26e53a5`;
  DESC-equilibrium SHA-256 is
  `2b0993a7576498d95f9483e2794e83f3d21799beaa31cb28c4159707fe753c1a`.
  Detailed evidence and all eight DESC figures are in
  `reports/qh_differential_qs_metric_investigation.md` section 13 and
  `reports/assets/qh_corrected_adam_93p166_full_eval_20260803/`. Local
  validation is `129 passed`; final report/artifact delivery commit is
  `7ea28d6`.
- On 2026-08-02 the user permanently stopped routine latent-score collection
  because the shared $G$ convention bug invalidates the old score calibration
  and prior proxy experiments did not justify further accumulation. Known
  collectors `30594` (Students) and `30859` (P107) were explicitly cancelled
  and confirmed `CANCELLED` with `0:0` after elapsed times `21:26:18` and
  `04:00:33`. Do not launch or report corpus collectors again unless the user
  explicitly reverses this instruction. Ignore unrelated future Student jobs;
  project GPU work should use the four P107 RTX 5090 GPUs only.
- Historical completed branch: `qh-volume-qs-g-fix`, created from
  `qh-flow-screened-adam` at `d5e5689` on 2026-08-02. It owns the versioned
  correction of the differential volume-QS convention, fixed 1024+1024 score
  calibration, same-start 200-step Adam comparison, and complete physical
  evaluation requested after the audit.
- On 2026-08-02, a focused audit of the differential volume-QS metric found a
  definitive shared Python/C++/CUDA convention bug. The volume pipeline uses
  radian angles and toroidal flux divided by $2\pi$, so the Boozer covariant
  current function must be
  $G=\mu_0 I_{\mathrm{link}}/(2\pi)$. Both
  `stellarator_eval/volume_qs.py` and `gpu_backend/src/score_pipeline.cu`
  instead use $G=\mu_0 I_{\mathrm{link}}$, i.e. the normalized-turn value is
  too large by exactly $2\pi$. This inflates QA and QH while leaving QP
  unchanged. Human-facing outputs also mix helicity normalization: target QH
  is raw, QA has unit norm, and QP is already divided by `nfp`. Exact offline
  reconstruction under the current constant-iota configuration changes the
  score-61.339 sample's QA/QH-raw/QH-per-helicity errors to
  `0.119169/0.020415/0.004951`, and the score-63.691 sample's to
  `0.120479/0.012113/0.002938`; normalized QH is respectively about 6.0x and
  10.3x lower than QP. Independent accepted Boozer surfaces also put QH 26--94x
  below QA/QP. Therefore old volume-QS components, QH competitor gates, and
  total-score physical calibration are superseded pending a versioned fix and
  threshold recalibration; old geometries and independent full evaluations
  remain valid. The audit-only state is now superseded by the active
  `qh-volume-qs-g-fix` branch: Python and CUDA use
  $G=\mu_0I_{\mathrm{link}}/(2\pi)$, score ABI is 9, explicit raw and
  per-helicity diagnostics are exposed, and score composition consumes only
  the explicit per-helicity fields. The complete local suite passes with
  `128 passed`; remote compilation and numerical acceptance are complete.
  Collector rows preserve `nfp`, `n_base_coils`, and every coil's complete
  decoded `x[33],y[33],z[33],current_A` token, so all collected cases can be
  exactly re-scored with a corrected versioned CUDA library without re-running
  flow decoding. The saved scalar diagnostics alone cannot reconstruct the
  corrected total score because edge QA/QP moments and their covariance were
  not stored; rescoring from the preserved decoded tokens is required.
  Full evidence and the correction plan are in
  `reports/qh_differential_qs_metric_investigation.md`.
- Corrected ABI-9 CUDA build job `30990` completed `0:0` in 46 seconds. The
  accepted library SHA-256 is
  `40dca7422995a91eab0a58285d9ced59a8e3be04a96b2b37686effbe6f1abff5`.
  Four-GPU smoke job `30992` completed `0:0`; its two known full-evaluation
  cases reproduced the audit's algebraic predictions to displayed precision:
  old-score 61.339 became score `88.9614871` with corrected
  QH-per-helicity/QA/QP `0.0049513402/0.119168541/0.029613844`, and old-score
  63.691 became `90.9812895` with
  `0.0029377588/0.120479481/0.030117924`. Both report
  $|G|=1.1051014\ \mathrm{T\,m}$. Smoke timing is not accepted because two
  startup utilization counters were transiently 100% despite 2 MiB memory and
  no compute PID. Commit `412cd4b` therefore requires three consecutive fully
  idle probes. Formal matched 1024 held-out QUASR QH plus 1024 FP32 RK4-256
  random-flow calibration job `30994` completed `0:0` in `44:07`; all four GPUs
  were 0%, 2 MiB with no process both before and after. QUASR versus random-flow
  all-sample score mean/median/max are `48.019/75.520/95.262` versus
  `24.087/0.372/87.362`; status-ok rates are `56.93%` versus `45.41%`.
  Score at least 80 occurs in `443/1024` QUASR and `17/1024` random cases, a
  26.1x enrichment. Among status-ok cases, median QH error per helicity is
  `0.002545` versus `0.04918`. Eight score workers sustained `0.7882` samples/s
  for 2050 cases; random-flow decode took 17.00 s. Frozen summary and plot are
  in `reports/assets/qh_corrected_score_calibration_30994/`; full remote rows
  are in `runs/corrected_score_calibration_1024x2_20260802/results`. All
  recovery results append to `reports/qh_differential_qs_metric_investigation.md`,
  not a new report. First Adam submission `31051` is invalid launch evidence:
  Slurm comma-separated `--export` truncated the center-backtracking sequence
  to `[0.5]`; it was cancelled after two iterations and must not enter the
  comparison. Replacement job `31058` completed `0:0` in `01:29:37` under the
  exact old-job-30662 settings: same `start_10`, seed `20260804`, 200
  iterations, eta `0.01`, beta1/beta2 `0.5/0.999`, perturbation `0.005`, four
  directions, FP32 RK4-256, robust whole-step skipping, and center backtracking
  `[0.5,0.25,0.125]`; only the score library/objective changed to ABI 9. It
  improved `85.88325 -> 93.16556` (best at iteration 197; final `93.16016`),
  applied all 200 updates with zero invalid pair endpoint, and had maximum
  drawdown `0.4084`. The best native components axis/psi/surface/coordinate/
  volume-QS/iota/coil are `97.910/98.532/97.874/89.436/94.202/100/65.318`;
  iota is `1.64627` and QH/QA/QP errors per helicity are
  `0.0023003/0.115880/0.0289945`. Four-GPU postflight was 0%, 2 MiB on every
  card. Python emitted only harmless duplicate semaphore cleanup warnings at
  interpreter shutdown; all 200 history rows and artifacts are complete.
  Frozen artifacts and same-start comparison plot are in
  `reports/assets/qh_corrected_score_adam_start10_200_31058/`. The next active
  stage is the maintained complete physical evaluation of this `best.json`;
  source `a` and surface `s` must be selected for this sample, not copied from
  earlier cases.
- Previous completed branch: `qh-flow-screened-adam`, created from
  `qh-flow-score-regression-proxy` at `53c95a00041ce0b9082d6e1b0b177dc41ba66741`
  on 2026-08-02. The active experiment uses the familiar `nfp=4`, three-base-
  coil condition and, for each independent seed, decodes and native-scores 128
  IID Gaussian flow latents on CUDA, selects the highest current-production
  score, then runs 50 steps of the validated robust low-momentum Adam policy
  (`eta=0.01`, `beta1=0.5`, `beta2=0.999`, perturbation `0.005`, four
  antithetic directions, FP32 RK4-256). It records candidate-selection, Adam,
  and exact end-to-end wall times. No full physical evaluation belongs to this
  experiment unless the user explicitly requests one. Implementation commit
  `afd6db8a61ea802dda8ef2392df7d3729cd5a498` and metadata-recovery correction
  `cf57f34486e6d0b1ec8c3bae82fa145f09aebb0c` are synchronized to the remote
  worktree. Full local validation is `126 passed`. Four-GPU jobs `30788_0` and
  `30790_[1-7]` completed all eight candidate seeds `2026080200--2026080207`.
  Six of eight runs reached score 40, none reached 50, and best scores had
  min/median/mean/max `32.9074/44.3686/43.1026/49.5427`; median gain was
  `20.7762`. Mean candidate-screen and complete end-to-end times were
  `178.35 s` and `1375.26 s`. Seed `2026080203` applied only 25 of 50 Adam
  updates because dirty endpoints correctly skipped the other rounds. The
  first job's numerical run completed 50/50 steps but its original summary
  assertion rejected the `0.0585` FP32 batch-versus-single decode score
  discrepancy; its summary was recovered from unchanged artifacts without a
  rerun. The other seven jobs completed `0:0`. No full evaluation was run.
  Combined score-history analysis shows that all runs except seed `2026080203`
  achieved their running best at step 50. Last-10-step best-score gains for
  seed suffixes `00--07` were
  `1.038/3.964/0.679/0/0.607/1.527/4.140/3.609`; thus 50 steps truncated most
  trajectories rather than demonstrating convergence. Seed `03` is the
  exception: all final ten rounds were skipped by the invalid-endpoint gate,
  so its flat tail is feasibility-boundary stalling, not ordinary saturation.
  Detailed evidence is in `reports/qh_screened_start_adam_report.md`, local
  assets under `reports/assets/qh_screened_start_adam_20260802/`, and remote
  results under
  `~/local_surface_evaluator/runs/qh_screened_start_adam_20260802_nfp4_nc3`.
  Report and frozen local-asset delivery commit
  `2bcbd79bd2bb22acac3e19a5cb47c9b8244dccb9` is synchronized to the remote
  worktree. Combined eight-seed score curves, tail-progress metrics, and raw
  histories were added in commit
  `32856ae129c162558a0a4bf4e9e07c1e5c6f92f5`, also synchronized remotely.
  Final report/artifact delivery metadata, including the 70,724-row corpus
  snapshot, is commit `b8ef79b67c8bb2a66bcf3627b7e8b699f3de9319`; this
  commit is included in both local and remote worktrees.
  Low-priority P107 collector `30859` was restored after foreground completion,
  retains `Nice=10000`, and atomically wrote one 64-row shard on each of four
  ranks; independent Student collector `30594` remained active throughout.

- Previous completed branch: `qh-flow-score-regression-proxy`, created from
  `qh-flow-latent-proxy` at `da734e4a89a883237e0a65177a7b40795174e312`
  on 2026-08-02. This branch owns the completed frozen-corpus
  latent-to-native-score regression experiment. Delivery commit
  `a56b160213ed70b3540cc7ac4dabcec651400b6e` is synchronized to the remote
  worktree. Query `git rev-parse HEAD` at session start for any later tip.
- On 2026-08-01, the current validated production native-score library became
  SHA-256 `4bf7a12ea3dbdef9faf6de3ce4dc1840ecf48847ba795267500dd4179f730708`.
  It uses the strict mathematical elliptic-axis condition, a continuous
  topology-quality margin, the original fixed maximum of six high-precision
  surface candidates, and preserved diagnostics for the closest rejected
  long-trace candidate. The old library `0b7342db...aa427` is archived and must
  not be the default for new optimization or collection. All launch wrappers
  are pinned to the new hash. The complete local suite has 122 passing tests.
  Detailed numerical evidence is in
  `reports/qh_random_start_score_adam_report.md` section 11.
- On 2026-08-02, complete physical evaluation of the topology-fixed Adam
  sample was accepted. The frozen input SHA-256 is
  `63de73980ad07d457e79c3eaa9b2ef34d731e36622d06dad7f06413afd531539`;
  its current-production native score is `61.33896330666827` with components
  axis/psi/surface/coordinate/volume-QS/iota/coil =
  `99.2312/97.2853/83.0218/85.2201/38.7846/100/62.7895`.
  Sample-specific source fitting selected `a=0.06`. Standard LS/Newton plus
  independent dense validation accepted `s=0.24,0.36,0.49,0.64` and rejected
  adjacent `s=0.81`, so `s=0.64` was selected. The surface has
  `|V|=0.0412330184 m^3`, `iota=1.94668607`, dense relative residual
  `2.71927e-5`, normal-field P95 `4.63098e-5`, and surface QH error
  `1.33268e-4`; Poincare passed. CPU DESC stayed nested and reached final
  normalized force mean/P95/max `3.127e-3/6.993e-3/1.647e-2`, but hit its
  50-iteration limit. Selected-surface SHA-256 is
  `06420743e7f812ced6c7b5538f303e1976bdb5f373d6bb891f4fe30ea2a71df4`;
  DESC-equilibrium SHA-256 is
  `96e021104c225002a09170bbe587613ba386ae888e4e77b530a84193556add2e`.
  Full evidence and all eight DESC figures are in
  `reports/qh_random_start_score_adam_report.md` section 12 and
  `reports/assets/qh_adam_topology_fixed_61p339_full_eval_20260801/`.
- Full-evaluation orchestration was corrected on 2026-08-02. Surface candidates
  now default to parallel one-GPU/four-CPU jobs; `SERIAL_CANDIDATES=1` is only
  an explicit resource-limited override. Current remote JAX has no CUDA
  backend, so DESC uses `DESC_BACKEND=cpu-p107`, requesting 16 P107 CPUs and no
  GPU. Strict GPU attempt `30642` is invalid infrastructure evidence only;
  CPU-DESC job `30645` completed `0:0` in 5:46. Do not silently request a GPU
  and let JAX fall back to CPU.
- Alpha preprocessing now defaults to equal-area `gpu-ray` sampling,
  vectorized C++/CUDA FP32 field evaluation and flux calibration, and PyTorch
  CUDA FP32 QR. First remote comparison job `30651` failed before nu because
  the accelerated coordinate adapter omitted `grad_psi`; commit
  `1b1b1a2d01de9f268774fa4f963700ddeb674d1a` fixed the interface and added a
  regression test. Same-surface jobs `30653` and `30655` then completed and
  matched the legacy final standard surface to about `3e-6` in iota and
  `1e-9` in independent residual/QH metrics. Flux calibration fell from
  `54.87 s` to `0.56 s`; alpha total fell from `140.22 s` to about `105 s`.
  The remaining roughly `62.6 s` volume-sampling stage is 1,574-mode fitted-psi
  basis construction/evaluation over about 226k ray candidates; a Horner
  rewrite did not provide a measurable end-to-end gain. This is outside the
  ten-second native-score path and needs a separately validated dedicated GPU
  polynomial evaluator if optimized further. Current complete local suite:
  118 passing tests.
- On 2026-08-02 the user clarified the performance boundary: DESC is allowed
  to run on CPU. The strict GPU-throughput requirement applies to the native
  C++/CUDA coils-to-score path; CPU DESC should request no GPU, while native
  score code must not acquire accidental Python/CPU fallbacks or avoidable
  serial work.
- On 2026-08-02 the user added a hard full-evaluation fallback rule. For any
  stage that is naturally batch-parallel on CUDA and whose CPU implementation
  is roughly one to two orders of magnitude slower, do not autonomously fall
  back to the CPU or an old slow backend for any reason. If a CUDA path can be
  added or repaired with a simple code change, implement and validate it;
  otherwise stop at that stage, preserve the evidence, and ask the user to
  choose between accepting the slow CPU path and designing a GPU algorithm.
  The `legacy-cartesian` alpha fallback used during the 63.69 full evaluation
  is a one-time explicitly forgiven exception and must not become precedent.
  This rule does not supersede the user's explicit permission for DESC itself
  to run on CPU.
- Slurm job `30662`, the requested 200-step low-momentum Adam run from the
  original IID `start_10`, completed `0:0` in 1:32:00. With $\eta=0.01$,
  $\beta_1=0.5$, $\beta_2=0.999$, perturbation `0.005`, four antithetic
  directions, FP32 RK4-256, robust whole-step skipping, and production score
  library `4bf7a12e...`, it improved `38.6590225 -> 63.6914797` (best at step
  195; final `63.6786003`). It applied 184 updates, skipped 16 dirty-endpoint
  rounds, had no invalid center or center rollback, and reduced maximum
  drawdown to `0.7625`. The old same-IID 200-step package reached only
  `59.97998` with drawdown `4.2007`; this is a package-level comparison, not a
  beta1-only ablation. Frozen optimizer evidence is in
  `reports/assets/qh_adam_low_momentum_start10_200_30662/`.
- Complete physical evaluation of the `63.6914797` sample was accepted on
  2026-08-02. Frozen input SHA-256 is
  `3e1843b2b8ae2a603bf1150daa0de6bdc16d9c8c7e5ce1805711a05cb04f4693`.
  Sample-specific source fitting selected `a=0.08`; nested standard surfaces
  passed at `s=0.24,0.36`, guarded `s=0.49` failed, and formal Newton successes
  at `s=0.64,0.81` were rejected as inner-branch jumps because enclosed volume
  decreased and fitted psi collapsed inward. The selected `s=0.36` surface has
  `|V|=0.0491435318 m^3`, `iota=1.94508971`, dense residual `2.87594e-5`,
  normal-field P95 `4.78143e-5`, and surface QH error `4.41025e-5`; Poincare
  passed. CPU DESC job `30745` completed `0:0` in 4:35, remained nested, and
  reached normalized force mean/P95/max
  `2.725e-3/6.124e-3/1.484e-2` at its 50-iteration limit. Relative to the
  61.339 sample, volume is 19.2% larger, surface QH error is 66.9% lower, and
  all three final DESC force summaries improve by about 10--13%. Selected
  surface SHA-256 is
  `c5d9b6eb12c57637c5c61831cf5c046fb592c7046d29976d2c28c44666e9e279`;
  DESC equilibrium SHA-256 is
  `a5115b395cd39c83b47e9c38698e23427b81a329b8cbb09e4629c352565ff05d`.
  Full evidence and all eight DESC figures are in report section 13 and
  `reports/assets/qh_adam_low_momentum_63p691_full_eval_20260802/`.
- Full-evaluation branch selection now rejects formal solver successes whose
  absolute enclosed volume does not increase with outward target `s`, while
  preserving the raw `solver_accepted` state and not imposing an arbitrary
  initial-distance threshold. Implementation commit `fab3751` includes the
  regression test. Full-evaluation submitters now default to the validated
  base-repository DESC environment rather than a nonexistent worktree-local
  venv; failed job `30742` exited in one second before numerical work and is
  invalid infrastructure evidence only.
- Background collection is active through Student job `30594` and low-priority
  P107 job `30859`. Metadata-only recount job `30889` completed `0:0` on
  2026-08-02: the unified append-only corpus contained exactly 70,724 completed
  samples in 1,107 shards from 36 streams, with `ok=30553`, `no_axis=18927`,
  `no_surface=5064`, `drift_rejected=15635`, and `flux_rejected=545`. Refresh
  this count at every later delivery because both collectors continue to append
  shards. Earlier recount job `30780` is invalid launch-only evidence (`127:0`, zero
  seconds, no numerical work): Slurm `--wrap` used `/bin/sh`, where `source`
  was unavailable; use POSIX `. /path/to/activate` in future metadata wraps.
- The 2026-08-02 latent-score regression experiment is complete. It used the
  current production native score divided by 100, sigmoid output, and MSE on a
  frozen 43,584-row current-library snapshot with disjoint deterministic
  34,868/4,358/4,358 train/validation/test splits. The final 717,415-parameter
  model explicitly conditions on `nfp` and `n_coils`, starts exactly from the
  train-set `(nfp,n_coils)` score-mean baseline, and learns only a latent
  residual. Job `30769` completed `0:0` in 1:40; it selected step 75, continued
  through step 2,475 and all four LR reductions, and observed a persistent
  validation rise. Its checkpoint SHA-256 is
  `73a523acb34635fd95f630d44eab48c79d51917b05be8b435d2dfe9f5ed201e3`.
  On independent test it reached RMSE/MAE/R2/Pearson/Spearman
  `4.0255/3.0882/0.1573/0.3971/0.3975`, only slightly better than the condition
  baseline `4.0467/3.1057/0.1484/0.3854/0.3808`. Its prediction range is only
  `0.450--7.226`, so it emitted no prediction above 10 despite 34 actual test
  scores above 20 and seven above 30. The model is valid negative evidence for
  absolute high-score regression, not an accepted high-score proxy. Full
  frozen-corpus distribution, convergence evidence, test plots, and tail
  analysis are in `reports/qh_latent_score_regression_proxy_report.md` and
  `reports/assets/qh_score_regression_proxy_30767/`, `30768/`, and `30769/`.
- Foreground four-GPU score-regression job `30767` completed `0:0` on
  `anode01` in 2:55. It froze 43,584 current-library rows from 681 shards and
  excluded 15,556 old-library rows; the deterministic split is
  34,868/4,358/4,358. The first reused-classifier regressor selected step 175
  after continuing to step 2,575 and observing a clear validation-loss rise.
  On test it reached RMSE `4.0891`, Pearson `0.3737`, Spearman `0.3810`, and
  R2 `0.1305`, which does not beat the simple `(nfp,n_coils)` train-mean
  baseline (`4.0467/0.3854/0.3808/0.1484`). It emitted no test prediction above
  10 even though test contains 34 actual scores above 20 and seven above 30;
  this first checkpoint SHA-256 `57e0a2a8...e286e22` is valid negative evidence,
  not an accepted screening proxy. Controlled job `30768` added explicit
  `n_coils` and reduced capacity; it completed `0:0` in 1:30 and improved test
  RMSE to `4.0593` but still did not beat the condition-mean baseline. Job
  `30769` is the final baseline-anchored result recorded above. Low-priority
  P107 collector `30747` was intentionally cancelled after 1:12:18 to release
  the four GPUs and must be restored after foreground acceptance; Student
  collector `30594` remains running. Jobs `30765` and `30766` are invalid
  launch-only failures (`128:0`, zero or one second, no numerical work): the
  first was submitted outside the project so its relative log directory did
  not exist, and the second changed Slurm's working directory without fixing
  `SLURM_SUBMIT_DIR`. Future submissions must use
  `scripts/submit_qh_score_regressor.sh`, which pins both `--chdir` and the
  exported `PROJECT` path.
- Delivery code fixes rank-based calibration bins so tied predictions cannot
  create empty quantile bins, and accepts step 0 as the valid selected
  checkpoint when no learned residual beats the condition baseline. The full
  local suite passes: `122 passed`.
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
- Native score changes must not increase ordinary latency or introduce even a
  rare long-latency tail. If a correctness change cannot be made with
  essentially unchanged bounded cost, keep the production score unchanged,
  present the measured tradeoff to the user, and wait for a decision. A
  diagnostic implementation that is slower must be marked non-production and
  reverted after the audit.
- DESC may run on CPU and is not part of the native C++/CUDA throughput
  requirement. In the current environment it must explicitly use the 16-CPU,
  zero-GPU `DESC_BACKEND=cpu-p107` path. Optimization and timing audits should
  focus on the native coils-to-score chain, which must remain C++/CUDA and use
  available GPU parallelism.
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
- Reports must use plain language as the primary narrative, not dense internal
  shorthand. Define every necessary technical term on first use, explain what
  each quantity or algorithm does in the workflow, and do not require the user
  to infer meanings from names such as G3/G4, VJP, secant, branch, or smoke.
  Clearly separate verified results from hypotheses and cite the relevant plots
  and raw summaries.
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
  remains independent and running. In replacement array `30543`, beta1=0.9
  and 0.5 with robust filtering at perturbation 0.01 completed by safely
  skipping every post-best dirty step, but froze at their step-2 best scores.
  The perturbation-0.005 element then found a separate failure: an all-`ok`,
  ordinary-scale gradient proposed an updated center with `no_axis`. Element 5
  and pending element 6 were cancelled. The next corrected version retains the
  skip policy and additionally rolls back parameters, both moments, and Adam
  step count whenever an updated center is non-`ok`; proposal diagnostics are
  preserved rather than hidden.
- On 2026-08-01, independent proposal audit jobs `30551` and `30555` proved a
  native axis-topology bug. Iterations 1--4 replayed with exactly zero noise and
  update RMS error, RK4-256/RK4-512 agreed, and all four GPUs repeated the same
  state. The old implementation incorrectly used
  `abs(trace)/sqrt(det) < 2 - axis_topology_margin`, so the default margin 0.02
  changed mathematical existence into a hard threshold at 1.98. The exact
  proposal has residual `1.366e-8` and normalized absolute trace `1.980828 < 2`:
  it is a strict elliptic axis. Production now uses strict `<2` for existence;
  the 0.02 margin only defines candidate preference and a continuous quality
  scale. The exact proposal is then `status=ok` on all GPUs. This supersedes the
  earlier instruction to keep the buggy score definition for corpus
  comparability; old rows remain immutable and are distinguished by their
  recorded library hash.
- Exact replay of the five historical invalid Adam endpoints found that the
  topology fix restores two old `no_axis` probes to scores `57.7851` and
  `59.6424`, reducing their directional score jumps by factors 131 and 1768.
  A third old `no_axis` becomes the accurate `drift_rejected`; the other two
  historical drift rejections remain. All three remaining cases have closed
  elliptic axes with residuals `1e-9`--`5e-8` and psi-angle P95 near
  `1.2e-4`--`1.4e-4`. They pass the 5% drift criterion through 8 periods and
  require about `5.070%`, `7.063%`, and `7.574%` at 16 periods. Thus
  `drift_rejected` means the fitted-psi surface seeds did not pass the bounded
  long-trace screen; it does not mean no axis and does not prove that full
  LS/Newton cannot find a magnetic surface.
- An experimental all-candidate surface verifier was built only for diagnosis.
  It tried all 9--10 one-period candidates and restored none of the three drift
  endpoints, while the same audit slowed from 81 seconds to 97 seconds. This
  change was reverted before production. SHA values
  `15278af22326655eeb91473ff2b344c2ffc7b543c525f3d8ba5c211f90cd81f0`
  and `53de3ff55954174fcc629b7c03de025f3819becc051aa491ac22565f22d080bd`
  are diagnostic-only and must not be deployed. Final bounded audit job `30589`
  completed `0:0` in 82 seconds and preserves the failed candidate's already
  computed level, one-period drift, long drift, and crossing period without
  adding traces.
- The corrected robust Adam policy skips the whole gradient/moment/parameter
  step if any directional endpoint is non-`ok`, uses only scale-adaptive
  median/MAD winsorization for valid directional outliers, and rolls back
  parameters, both moments, and Adam step count if an updated center remains
  invalid after bounded feasibility backtracking. Topology-fixed short job
  `30569` completed all 16 steps from `59.97998` to `61.33896` with no
  `no_axis` proposal or long drawdown. This validates the immediate fix but is
  not a long-run beta optimum claim.
- Old-library Student collector `30399` was intentionally cancelled after
  `06:10:27` so the score binary could be replaced cleanly. New-library
  collectors `30594` (Student, two GPUs) and `30595` (P107, four GPUs, low
  priority) started independently from commit `e16402e`; both launchers
  validated the production library hash before entering their collection loop.
  On 2026-08-01, `30595` was deliberately cancelled after `00:37:12` to release
  P107 for complete physical evaluation of the topology-fixed Adam sample;
  `30594` remains running and must not be disturbed. The evaluated sample is
  frozen from `runs/qh_adam_topology_fixed_short_20260801/best.json`. Its
  recorded score `61.3389633067` came from intermediate native-library SHA-256
  `13966f7a...`; the complete report must also re-score the unchanged sample
  with current production library SHA-256 `4bf7a12e...` and distinguish the two.
  The frozen input SHA-256 is
  `63de73980ad07d457e79c3eaa9b2ef34d731e36622d06dad7f06413afd531539`.
  Complete-evaluation root is
  `~/local_surface_evaluator_worktrees/qh-flow-zo-adam/runs/qh_adam_topology_fixed_61p339_full_eval_20260801`.
  Source-psi jobs `30602/30604/30606/30608` are running for
  sample-specific `a=0.04/0.05/0.06/0.08`; all four were confirmed on idle
  RTX 5090 allocations. Select `a` from their measured diagnostics rather than
  reusing an earlier sample's value.
  All four completed `0:0` in 27--69 seconds. The selected source is `a=0.06`:
  validation RMS/angle-P95 are `8.039e-4/1.062e-4`, its cheap screen reaches
  mean radius `0.04903 m` at `s=0.64` and fails at `s=0.81`; `a=0.08` gives
  virtually no extra physical coverage but materially worse fit error.
  Serialized standard-surface jobs `30611/30613/30615/30617/30619` test
  `s=0.24/0.36/0.49/0.64/0.81`. Current-production one-case rescore job `30621`
  is also submitted; it is pending only on the 16-CPU P107 QOS limit while the
  first surface job runs.
  This serial policy was not a user requirement: it was introduced by agent
  commit `9a3eb43` without performance evidence and conflicts with the durable
  requirement to use available GPUs. Pending jobs `30613/30615/30617/30619`
  were cancelled before startup. A first replacement attempt showed that an
  environment-only CPU override did not beat the implementation script's
  `#SBATCH --cpus-per-task=16`; `30624` therefore started with 16 CPUs and the
  remaining `30626/30628/30630` stayed CPU-QOS blocked. The fixed entrypoint now
  defaults to parallel candidates, adds an explicit command-line
  `--cpus-per-task=4`, and retains `SERIAL_CANDIDATES=1` only as an explicit
  resource-limited option. Do not restore serial evaluation as the default.
  The first legacy candidate exposed the next bottleneck: alpha took `152.65 s`
  while its GPU QR took only `3.79 s`; flux calibration took `52.88 s`, and
  roughly `96 s` remained in oversized CPU Cartesian-lattice filtering and
  coordinate setup plus field sampling. `alpha_clebsch_ls_experiment.py` now
  defaults to `gpu-ray`, reusing the already tested `volume_qs` equal-area ray
  sampler, GPU psi/gradient evaluation, one batched GPU field evaluation, and
  vectorized GPU flux calibration. `--sampling-backend legacy-cartesian`
  remains for controlled comparison. The complete local suite has 115 passing
  tests. A same-surface remote speed and physics comparison is still required
  before treating the new default as numerically accepted.
  A metadata-only recount after 24 minutes of the new jobs found exactly 17,092
  completed samples in 269 atomic shards from 20 streams (`ok=7389`,
  `no_axis=4788`, `no_surface=1148`, `drift_rejected=3645`,
  `flux_rejected=122`). This is a dated snapshot; refresh it from shard
  metadata at every later delivery.

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
