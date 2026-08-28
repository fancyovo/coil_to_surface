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
