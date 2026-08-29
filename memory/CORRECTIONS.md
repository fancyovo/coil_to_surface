# Correction Ledger

Last reviewed: 2026-08-28 (Asia/Shanghai).

This ledger is append-only at the entry level. Record both model-discovered and
user-reported errors. Keep the erroneous artifact for provenance, add a visible
supersession notice where it may be reused, and state which conclusions survive.
An open critical correction blocks promotion and external reporting.

## CORR-20260825-01 - Coordinate protocol was misidentified

- Severity/status: critical / contained.
- Reported by: user, then verified from machine artifacts.
- Error: the 32-case Flow-versus-data control was discussed as evidence about
  the current 64-direction, 200-step method. The actual rerun used 2 directions
  and 100 steps. Historical and current methods were conflated.
- Evidence: run configuration and trajectory artifacts; analysis summarized in
  `reports/qh_data_space_large_scale_validation_20260825.md`.
- Corrected conclusion: retract the 32-case coordinate-causality claim. Retain
  the independent 48-condition initialization evidence. The 309-pair corpus is
  valid evidence for the two complete recipes it actually ran, not a pure
  coordinate ablation.
- Containment: correction banners were added to the 2026-08-24 and 2026-08-25
  reports; 2D launchers are inert; current protocol identity is machine-readable.
- Verification: the 309 manifest specifies 64 directions and 200 steps, and
  3,955,200 aggregate directions equals `309 * 200 * 64`.

## CORR-20260828-01 - A historical 2D CLI default remained executable

- Severity/status: critical / contained.
- Discovered by: model during the user-requested mainline audit.
- Error: the private generic latent optimizer still defaulted to 2 directions,
  200 steps, and learning rate 0.01 after the accepted/public workflow had moved
  to 64 directions, 200 steps, and learning rate 0.02. Several historical shell
  launchers could also submit 2D runs. This could silently create new data under
  an obsolete method.
- Impact audit: this does not invalidate the 309 corpus; its manifest and count
  prove 64 directions. Post-protocol 2D experiments exist, including the
  invalidated 32-case control, and must be interpreted only as history.
- Corrected state: one shared module defines the current defaults; canonical and
  compatibility CLIs use it; exactly 2 is rejected; old shell launchers exit 64;
  resume rejects legacy, 2D, or mismatched protocol state.
- Affected prose: stale current/default wording was found in the main README,
  current method document, score-throughput report, score-compression report,
  direct-alpha feasibility report, and the 2026-08-24 coordinate report. Current
  documents now state 64D/200; historical reports carry top-level and local
  deprecation labels while preserving their recorded numbers.
- Verification: `tests/test_qh_optimization_defaults.py` checks values, protocol
  metadata, parser rejection, and every tracked shell file containing a 2D token.

## CORR-20260828-02 - Historical score maximum was presented as current

- Severity/status: high / contained.
- Error: memory retained the step-4341 score `93.3672653` as the highest fully
  evaluated sample after a current-objective sample scoring `94.6368682` had
  already been fully evaluated.
- Corrected conclusion: `94.6368682` is the verified current cubic-iota score for
  `p107_37034_3_000018_step0150`. The step-4341 sample remains a valid historical
  constant-iota reference, with scores not directly comparable across objectives.
- Evidence: `reports/qh_min_face_qh_full_evaluation_20260819.md` and
  `reports/qh_score_throughput_and_continuous_surface_plan.md`.
- Containment: root memory now separates the two objective definitions.

## CORR-20260828-03 - Untracked inventory was understated

- Severity/status: low / contained.
- Discovered by: model during final mainline verification.
- Error: the restructured root memory described the pre-existing untracked
  inventory as hundreds of files. `git ls-files --others --exclude-standard`
  reported 12,230 files on 2026-08-28.
- Impact: no source, protocol, experiment, or numerical conclusion is affected;
  all untracked files remained unstaged and untouched.
- Corrected state: root memory now says "many thousands" and requires a live
  count when the exact inventory matters, avoiding another stale snapshot.
- Verification: tracked status remained clean after the consolidation commit.

## CORR-20260828-04 - Zero-hit Wilson lower endpoint retained roundoff

- Severity/status: low / contained before experiment submission.
- Discovered by: model through the new random-survey unit tests.
- Error: the first implementation of the score-tail Wilson interval returned a
  lower endpoint of `2.17e-19` for zero successes in 1000 trials. The exact
  probability boundary for a zero-hit report must be 0.
- Impact: no remote survey, report, numerical conclusion, or saved artifact used
  the faulty formatter. The error existed only in uncommitted code and was
  caught by the zero-success regression test.
- Containment: zero successes now force the lower endpoint to 0, all successes
  force the upper endpoint to 1, and the focused test suite passes.

## CORR-20260828-05 - Build gate did not activate its CMake environment

- Severity/status: low / contained; replacement validation still required.
- Discovered by: model from Slurm job `46832` stderr.
- Error: the first survey build launcher called `cmake` before activating the
  pinned virtual environment that provides CMake on Students compute nodes.
- Impact: job `46832` exited before configuration and produced no library,
  validation artifact, random sample, or numerical conclusion.
- Containment: the launcher now activates the pinned environment before the
  configure step. A new job ID and a passing hash/ABI/reference-score gate are
  required before any survey worker can run.

## CORR-20260828-06 - Rebuild validation assumed a stable linker hash

- Severity/status: medium / contained in code; replacement validation pending.
- Discovered by: model from build jobs `46845` and `46850`.
- Error: the second gate tried to compare a newly linked library against the
  hash from the preceding build. Reconfiguration/relinking changes the GNU
  build ID, so identical source and toolchain need not reproduce identical
  binary bytes across separate jobs.
- Impact: `46845` and `46850` stopped before native scoring. They produced no
  accepted validation, random sample, or numerical conclusion.
- Containment: one job now builds once, records that file's actual SHA-256, and
  immediately validates the same bytes against ABI 10 and the frozen
  `94.6368681663` reference score. The survey manifest must then pin the hash
  recorded by that validation artifact; it may not trigger another rebuild.

## CORR-20260828-07 - Reference gate mixed standalone and optimizer score modes

- Severity/status: high / contained in code; replacement validation pending.
- Discovered by: model from job `46853` component diagnostics and the frozen
  2026-08-19 scoring script.
- Error: the first numerical gate compared a reference created with standalone
  library defaults against a new call using optimizer center-score overrides
  (`surface_selection_mode=1`, theta 128, trace 400). The resulting `93.2054`
  was therefore compared with incompatible `94.6369` evidence.
- Impact: job `46853` was rejected and no random survey ran. Its score is a
  configuration diagnostic, not evidence of a source or evaluator regression.
- Corrected fact: the frozen reference used `batch_native_score.py` with no
  overrides: continuous standalone surface mode, theta 256, trace 800, and two
  confidence periods. Main contains the reference commit's GPU source.
- Containment: reference validation and random global workers now share one
  standalone helper that passes no config overrides. The deferred Adam200 stage
  remains separately labeled and uses its own frozen optimizer configuration.

## CORR-20260828-08 - Frozen standalone reference included warmup and solver pins

- Severity/status: high / contained in code; replacement validation pending.
- Discovered by: model from job `46861`, the frozen native-score `job.sh`, and
  `batch_native_score.py` argument defaults.
- Error: correction `CORR-20260828-07` described the frozen reference as having
  no Python overrides and omitted `--warmup`. The batch script always passed
  `psi_solver_mode=2` and `alpha_solver_mode=2`, then discarded one evaluation
  of the same case before writing its measured result.
- Impact: production-library gate job `46861` evaluated the frozen case cold.
  It returned status `ok`, ABI 10, and score `94.6254148`, so the gate rejected
  it against the warmed reference `94.6368682`. No random survey ran.
- Corrected fact: the two solver pins equal the current library defaults. The
  warmed and cold calls differ slightly in the psi/surface/coordinate numerical
  path, so the exact historical scoring state includes one discarded warmup.
- Containment: the survey now records the warmup-case hash, warms each worker
  process once, pins exactly the two solver modes, and leaves all other fields
  at library defaults. The replacement gate requires two consecutive warmed
  evaluations to match the frozen reference and each other within `1e-5`.

## CORR-20260828-09 - Warmup did not explain the reference-score drift

- Severity/status: high / contained for the survey; historical runtime cause
  remains unresolved.
- Discovered by: model from repeated-reference job `46865` and an exact array
  comparison between the frozen batch loader and the survey helper.
- Error: correction `CORR-20260828-08` attributed the score difference to a
  discarded warmup. Job `46865` returned `94.6254147736` identically on the
  first, second, and third call, with status `ok` and ABI 10 throughout.
- Evidence: frozen input and library hashes match; the helper passes coefficient
  and current arrays identical to `batch_native_score.py`; the current wrapper
  and GPU backend files have no diff from recorded commit `2ff16a6`. The old
  artifact did not preserve enough driver/CUDA runtime metadata to isolate the
  remaining `0.0114534` difference.
- Corrected fact: `94.6368681663` is the recorded 2026-08-19 observation.
  `94.6254147736` is the reproducible 2026-08-28 current-runtime reference for
  the same input and library. The full physical conclusions are unaffected by
  this small screening-score drift.
- Containment: every survey worker now treats the frozen case as a score
  reference preflight, requires the current value within `1e-5`, and records
  GPU UUID plus driver version. The six-GPU pilot will detect device-level
  differences before the formal sample count is calibrated.
- Verification: current-runtime gate job `46869` passed with three values equal
  to `94.6254147736` within `1.42e-14`, ABI 10, production library SHA-256
  `565c3207...c729`, and driver `580.173.02`.

## CORR-20260828-10 - Slurm export truncated calibrated worker counts

- Severity/status: medium / contained before formal sampling.
- Discovered by: model from prepare job `46890` stdout and its generated
  manifest, before any formal GPU worker submission.
- Error: the calibrated list `223,231,233,234,238,240` was passed inside
  Slurm's comma-delimited `--export` argument. Slurm assigned only `223` to the
  value and parsed the remaining numbers as separate export entries, so the
  prepare script expanded a uniform count of 223 to all six workers.
- Impact: job `46890` created a 44,154-sample manifest with six equal 7,359
  targets. It produced no global evaluation, candidate, clustering result, or
  numerical conclusion. The run directory is retained and classified invalid.
- Corrected fact: the pilot calibration requires per-condition counts
  `[223,231,233,234,238,240]`, whose sum is 1,399 and whose 33-condition total
  is 46,167 samples.
- Containment: the parser now accepts colon-delimited counts for Slurm export,
  and `prepare` requires an independent expected total sample count before it
  creates a run directory. The replacement formal run uses a new `v2` path.

## CORR-20260828-11 - Per-condition cluster counts were described as a global count

- Severity/status: medium / contained before the clustering report.
- Discovered by: user review of the first clustering status interpretation.
- Error: an interim update stated that the selected cluster count ranged from
  2 to 3 without naming its per-`(nfp,n_base_coils)` scope. That wording could
  be read as a global taxonomy and ignored the physical distinction among five
  base-coil counts.
- Primary evidence: `atlas_summary.json` contains 33 separately analyzed
  `(nfp,n_base_coils)` groups. Summing each group's `selected_k` gives 72 leaf
  partitions. The leaf totals for `n_base_coils=1..5` are
  `[11,15,14,17,15]`.
- Corrected fact: the atlas is hierarchical. Level 1 contains 33 hard physical
  condition groups; level 2 contains 2--3 geometric partitions per hard group;
  the complete atlas contains 72 leaves. Samples with different base-coil
  counts never enter the same group or leaf.
- Scope and retained conclusions: clustering assignments and numerical output
  remain valid. Only the interim verbal interpretation was wrong. The strong
  within-group concentration result also remains valid.
- Containment: future atlas summaries explicitly record hard-group count,
  total leaf count, leaf count by base-coil count, and the per-hard-group range.
  The overview figure names its colors as within-group partition counts and
  prints both global totals. End-to-end tests assert the hierarchy fields.
- Reporting blocker: resolved. Strict atlas job `46913` regenerated the summary
  and overview with explicit 33-hard-group/72-leaf fields. The report
  `reports/quasr_qh_structure_atlas_20260829.md` uses the two-level terminology
  and adds a separate five-hard-group/10-leaf cross-`nfp` atlas.

## CORR-20260829-12 - Low-band Adam controls were not required to be valid starts

- Severity/status: medium / contained before Adam200 execution.
- Discovered by: model while preparing the user-authorized three-hour follow-up.
- Error: the first follow-up selector sampled four controls uniformly from
  `[0,20)` without requiring evaluator status `ok`. All four selected controls
  have status `no_axis`, while the frozen local-gradient optimizer rejects any
  initial center whose status is not `ok`. The provisional 4.67 GPU-hour cost
  also treated all 14 selected records as full Adam200 trajectories.
- Primary evidence: all four low-band entries in
  `reports/assets/qh_data_gaussian_global_survey_20260829/adam_followup_selection.json`
  record `status=no_axis`; `scripts/optimize_flow_latent.py` raises on an
  invalid initial center before its first update.
- Corrected fact and scope: the immutable 14-record selection remains the
  pre-registered provenance. Its four invalid controls are reproducible
  optimizer-ineligible events. The post-acceptance extension adds 38
  `score < 10,status=ok` controls stratified by base-coil count and includes all
  34 `score >= 10` samples, yielding 72 executable Adam200 trajectories.
- Impact: the 46,167 global scores, score-tail rates, retained candidates, and
  clustering conclusions are unchanged. No Adam result requires invalidation
  because Adam200 had not started when the flaw was found.
- Containment: the new preparation command freezes source hashes, verifies
  deterministic sample reconstruction, preserves original membership, records
  inclusion probabilities, requires exactly 76 selected and 72 eligible
  records, and tests that random starts do not reuse a recorded axis hint.
- Promotion/reporting blocker: resolved for submission; final basin-rate claims
  remain blocked until all six workers and the weighted summary are accepted.

## CORR-20260829-13 - Consolidated optimizer referenced an undefined root variable

- Severity/status: high / resolved in code and verified at optimizer entry.
- Discovered by: model during acceptance of Slurm arrays `47488` and `47489`.
- Error: `scripts/optimize_flow_latent.py` defined `REPO_ROOT` but called
  `repository_provenance(PROJECT_ROOT)`. The compatibility wrapper defined its
  own `PROJECT_ROOT`, yet the imported `main()` retained the canonical module's
  globals, so every optimizer subprocess raised `NameError` before loading its
  initial case or starting iteration 1.
- Primary evidence: all 72 failure records under
  `qh_data_gaussian_adam200_stratified_3h_20260829/failures/` have the same
  traceback at optimizer line 696. Prepare job `47487` passed with 76 selected,
  72 eligible, and `[12,12,12,12,12,12]` eligible cases per worker. Each worker
  also reproduced the ABI-10 reference score `94.6254147736`.
- Corrected fact and scope: arrays `47488` and `47489` contain no Adam update,
  score improvement, or convergence evidence. Their four
  `ineligible_survey_status` outcomes remain valid diagnostics; the replacement
  run will recreate the same deterministic selection in a new frozen root.
- Impact: global-survey, QUASR-clustering, current-default, and earlier 309-case
  conclusions are unchanged. The failure exposes an execution regression in
  the consolidated canonical optimizer and compatibility entry point.
- Containment: the optimizer now passes its defined `REPO_ROOT`; a regression
  test inspects the executable `main()` binding. Follow-up workers stop after
  three identical failures and return nonzero for every runtime failure;
  `--allow-partial` applies only to a clean wall-time cutoff.
- Replacement evidence: preparation job `47902` passed on corrected commit
  `6f084a5a9b6799c01d57d4fe4ae0945573765d25`; worker arrays `47903` and
  `47904` passed the former failure point and entered the gradient estimator.
- Promotion/reporting blocker: the undefined-root defect is resolved. The
  independent library-interface defect is tracked by `CORR-20260829-14`.

## CORR-20260829-14 - Standalone score library lacked the optimizer batch API

- Severity/status: high / contained; corrected dual-library smoke passed.
- Discovered by: model during startup acceptance of arrays `47903` and `47904`.
- Error: the follow-up manifest froze production ABI-10 library SHA-256
  `565c3207...c729`, which reproduces standalone global scores but exports no
  `sgpu_create_field_batch_f32`. The local-gradient estimator requires that
  symbol and ten related query-batch symbols. Preparation and worker reference
  preflights checked only the standalone scoring contract.
- Primary evidence: every attempted optimizer subprocess in the `v2` run root
  raised `AttributeError: undefined symbol: sgpu_create_field_batch_f32` while
  binding `BatchCoilFieldGpu`. Each of six workers stopped after three identical
  failures. The frozen outcomes contain 18 runtime failures, four expected
  `ineligible_survey_status` diagnostics, and no completed optimizer result.
- Corrected fact and scope: arrays `47903` and `47904` contain zero Adam updates
  and no basin-rate evidence. The global survey and its production-library
  scores remain valid because they use only the standalone ABI-10 interface.
  QUASR clustering and all earlier accepted conclusions are unchanged.
- Containment: preparation and worker startup now load the gradient library and
  require the full query-batch symbol set before entering any sample loop; the
  required names are frozen in the run manifest and covered by regression
  tests. Build job `47911` produced batch library `b6697f54...48d6` and showed
  three deterministic ABI-10 scores of `94.6368686721`; validation job `47933`
  passed against that independent gradient-library reference. The interface
  keeps formal library `565c3207...c729` for every reportable score and assigns
  the batch library only to local gradients. Smoke job `47937` completed one
  64-direction Adam step with all 128 endpoints `ok`; its optimizer-formal score
  improved from `48.5949558` to `53.4776601` and both hashes matched.
- Promotion/reporting blocker: no random-start optimizability conclusion may be
  reported until the smoke, complete third run, and strict weighted summary pass.

## CORR-20260829-15 - Current optimizer combined incompatible library contracts

- Severity/status: high / dual-library execution verified on the exploration
  branch; default launches remain blocked pending mainline promotion.
- Discovered by: model while containing `CORR-20260829-14`.
- Error: the 2026-08-28 consolidation described the 64D/200 optimizer and the
  current compressed formal evaluator as one executable default. The canonical
  optimizer used a single `--lib` for formal center scores and its query-batched
  local-gradient oracle. Current formal library `565c3207...c729` implements the
  former contract and lacks the latter API.
- Primary evidence: `nm -D` finds no `sgpu_create_field_batch_f32` in
  `565c3207...c729`; arrays `47903/47904` fail at that exact binding. The
  historical 309-trajectory report records batch-capable library
  `7834a88d...9a3`. Both that library and current-source build
  `b6697f54...48d6` reproduce `94.6368686721`, while the compressed formal
  library reproduces `94.6254147736` under the same runtime.
- Corrected fact and scope: the 309-pair settings, direction count, trajectories,
  and within-corpus Flow/data comparison remain valid under their frozen batch
  library. They do not demonstrate that the later compressed formal library can
  serve as the same optimizer binary. The current method definition remains
  32-screen/64D/200; its executable artifact pairing required correction.
- Containment: `scripts/optimize_flow_latent.py` now accepts a separately hashed
  `--gradient-lib`, records formal and gradient roles independently, and checks
  both hashes on resume. The random-start follow-up freezes both dependencies,
  validates the batch API before sample work, and accepts success metrics only
  from formal-library scores. Regression tests cover the split interface.
  Smoke job `47937` verified one complete gradient/update/formal-rescore cycle.
- Promotion/reporting blocker: the complete follow-up must pass before this
  pairing supports random-basin conclusions. Promotion of the corrected default
  execution contract to `main` remains a separate required step.

## Required Entry Template

- ID, title, date, severity, status, and reporter/discoverer.
- Exact incorrect claim or behavior.
- Primary evidence and reproduction path.
- Corrected fact and scope.
- Affected reports, code, experiments, and downstream conclusions.
- Conclusions that remain valid.
- Containment change and regression check.
- Promotion/reporting blocker status.

Never resolve an entry merely by deleting the old wording. Resolution requires
an auditable corrected fact and a guard against recurrence.
