# Correction Ledger

Last reviewed: 2026-09-01 (Asia/Shanghai).

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

## CORR-20260830-33 - Vacuum G counted coils that did not link the selected axis

- Severity/status: critical / resolved and promoted to `main`.
- Reported by: user while reviewing special unlinked-coil geometries.
- Error: native ABI-10 and Python volume-QS computed
  `G = sign(flux)*2e-7*2*nfp*sum(abs(base currents))`. This assigned every
  physical coil the conventional unit linking number. Simsopt surface setup
  used the corresponding all-coil current sum as its initial Boozer `G`.
- Primary evidence: `G` enters
  `f_C=(M*iota-N)A-(M*G+N*I)C`, so the shortcut changed target and competitor
  QS errors, the volume-QS component, helicity gates, and total score. Exact
  linking found 468/1276 unlinked physical coils in 21/38 Adam200 cases and
  76/284 in four of six Adam2000 cases.
- Corrected fact: ABI-11 uses
  `sign(flux)*abs(integral_axis(B dl))/(2*pi)`, equivalent to
  `sign(flux)*mu0*abs(sum_j(I_j*Lk_j))/(2*pi)` for signed physical currents and
  oriented linking integers.
- Affected conclusions: ABI-10 scores for geometries violating its linking
  assumption are superseded as physical scores. ABI-10 trajectories remain
  frozen evidence for the objective that generated them. Population rates,
  score thresholds, rankings, and gains measured under ABI-10 remain historical
  until an ABI-11 experiment measures them.
- Retained conclusions: coil geometry, Biot-Savart fields, exact linking
  audits, magnetic surfaces, Poincare plots, DESC equilibria, resource costs,
  frozen trajectory integrity, and geometry-only novelty classifications do
  not use the shortcut. Converged standard Simsopt LS/Newton runs optimized
  `G` after initialization and are not invalidated solely by the old initial
  guess.
- Containment and verification: ABI-11 computes magnetic-axis circulation in
  formal native, fixed-front, query-batch, Python volume-QS, saved-QS, and
  Simsopt-initialization paths. ABI versioning, protocol identity, actual
  library-hash classification, resume checks, and active launchers block ABI-10.
  The canonical `main` library SHA-256 is
  `1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed`.
  The earlier exploration-worktree build SHA-256 is
  `921a51683ba6b2d17ef16daa63207d47f91c4dd55faea918675542c05d5d1668`;
  both builds produced the same 44 scores and components within `1.85e-13`.
  Forty-four strict-axis replays matched independent topology-predicted `G`
  within `1.0921e-8`; all 38 Adam200 endpoints remained at score 50 or above.
  See `reports/abi11_default_promotion_20260830.md`.
- Promotion/reporting blocker: resolved by the user-accepted ABI-11 mainline
  promotion. ABI-10 remains reproducible only through its frozen historical
  manifest and library outside current run state.

## CORR-20260831-40 - Acceptance report was delivered only from a hidden worktree

- Severity/status: medium / resolved during delivery; no numerical impact.
- Reported by: user after the acceptance response.
- Error: the model said the report was available locally and linked to the
  experiment's hidden `.worktrees/qh-online-rwcfm-zero` checkout while the
  primary user-visible checkout remained on an unrelated exploration branch.
  The primary checkout's `reports/` directory therefore did not contain the
  report, and dot-directory hiding made the linked location difficult to find.
- Corrected fact and scope: the report existed and was committed at `f38c7e9`
  on `codex/qh-online-rwcfm-zero`; only discoverability and checkout state were
  wrong. Audit evidence, figures, numerical conclusions, and protocol state
  remain valid.
- Containment and verification: move the experiment branch to the primary
  checkout, mirror the report and assets into the branch-independent
  `_shared_reports/` directory, open that mirrored file, and require future
  explorations to switch the primary checkout unless a preserved-state
  exception is stated explicitly.
- Promotion/reporting blocker: none after the mirror and checkout verification.

## CORR-20260901-44 - QUASR marginal standardization was treated as a suitable geometry prior

- Severity/status: high / corrected interpretation; replacement prior under
  direct measurement.
- Reported by: user after inspecting high-scoring Adam2000 coil geometry and
  the absence of Online RWCFM enrichment.
- Error: the zero-start experiments treated an independent standard Gaussian
  in QUASR marginally standardized Fourier coordinates as a broad neutral
  starting distribution. The inverse normalizer restores featurewise means and
  standard deviations without restoring cross-mode, XYZ, intercoil, topology,
  curvature, or torsion structure.
- Primary evidence: `flow_matching/data.py` defines the featurewise transform;
  the completed q0-q7 and q0-q9 fixed-prefix audits show no target enrichment.
  The user's geometry observation motivates the mechanism; those runs did not
  isolate it causally. Canonical failure report: branch
  `codex/qh-online-validity-rwcfm-zero`, commit `9079fa3`, file
  `reports/qh_online_rwcfm_failed_20260901.md`.
- Corrected fact and scope: both Online RWCFM runs evaluate learning from this
  specific QUASR-marginal prior. They do not establish that RL fails with a
  structured physical prior. Their completed ABI-11 abundance measurements
  remain valid.
- Containment and verification: jobs `50256` and `50431` are no longer active;
  the replacement protocol uses a hierarchical analytic reference-axis,
  winding-surface, and circle-valued-field construction and measures the coil
  component before RL.
- Promotion/reporting blocker: q8, q10, and their QUASR-marginal prior cannot be
  promoted or reused as the default start distribution.

## CORR-20260901-45 - Exploration branch was first created from the closed experiment

- Severity/status: medium / resolved before new source edits or submission.
- Discovered by: model during branch-baseline verification.
- Error: `codex/axis-surface-prior` was initially created while the visible
  checkout still pointed at the closed Online RWCFM branch. The intended base
  was consolidated `main`; the separate `main` worktree made a direct switch
  fail and the follow-up branch command inherited the wrong HEAD.
- Corrected fact and scope: the mistaken branch was renamed and deleted without
  force after switching to its identical source branch. A new
  `codex/axis-surface-prior` was created from
  `main@de75f6d9637a8d728f27b17260fa5959d209257d`. No new-prior edit, result,
  report, job, or scientific conclusion existed on the mistaken branch.
- Containment and verification: branch, HEAD, `main`, and merge-base were all
  checked against `de75f6d` before implementation. Future branch creation after
  a failed checkout must verify `git rev-parse HEAD` and the intended merge-base
  before the first edit.
- Promotion/reporting blocker: none.

## CORR-20260901-46 - First analytic-prior analysis request exceeded Students memory QOS

- Severity/status: low / resolved before submission; no remote-state or
  scientific impact.
- Discovered by: Slurm `sbatch --test-only` during the six-GPU experiment gate.
- Error: the dependent CPU analysis task requested `24 GB` under
  `qos_stu_default`, which returned `QOSMaxMemoryPerUser`.
- Corrected fact and scope: the generator and both GPU worker requests passed
  their preflights. The atomic launcher runs all three preflights before making
  the run directory or submitting a worker, so the failed gate created no job,
  run artifact, sample, score, or partial experiment.
- Containment and verification: the analysis task now uses the previously
  validated `2 CPU / 8 GB` Students summary envelope. The launcher must pass a
  fresh complete preflight before formal submission. Scheduler-predicted start
  times are not used as runtime evidence.
- Promotion/reporting blocker: none after the fresh preflight passes.

## CORR-20260901-47 - Tokamak-like analytic prior was incorrectly passed by the visual gate

- Severity/status: high / v1 stopped and invalidated; replacement under
  geometry-only review.
- Reported by: user after viewing the generated full-coil previews.
- Error: the model declared the visual check successful because the coils were
  smooth, linked, and low in high-mode energy. The check ignored the core
  morphology requirement: the reference axis remained almost circular, the
  winding tube had little toroidal variation, and the contours formed a nearly
  uniform tokamak TF-coil family.
- Primary cause: axis Fourier amplitudes scaled as `nfp^-2`; surface
  corrugation and ellipticity were small perturbations of a circular tube; the
  contour warp budget was additionally proportional to the already small
  intercoil level spacing. These three contractions compounded.
- Affected evidence: protocol
  `qh-axis-surface-contour-prior-score-abi11-v1`, jobs `51581`, `51582`, and
  `51583`, and 953 partial rows under run root
  `axis_surface_prior_20260901_6028de3`. Their code and ABI-11 outputs remain
  mechanically reproducible, but the rows do not sample the requested prior
  and support no intended-prior abundance conclusion.
- Corrected requirement: the reference axis must have visible stellarator
  bending; the winding surface must follow it; cross-section size, elongation,
  orientation, or triangularity must vary over a field period; and the coil
  family must inherit this global shaping while remaining locally smooth.
- Containment and verification: all six GPU shards and the dependent analysis
  job were cancelled and fully left the queue. The remote run has
  `termination.json` with status `aborted_visual_gate_failed` and per-shard
  counts. Future prototypes must display axis, inner reference surface,
  winding surface, and coils together and receive explicit user approval before
  any batch score submission.
- Promotion/reporting blocker: v1 is invalidated and cannot be resumed or
  promoted.

## CORR-20260901-48 - V2 representative analysis retained a hardcoded v1 family list

- Severity/status: medium / resolved and verified.
- Discovered by: model during v2 result acceptance.
- Error: `representative_rows()` still iterated over `near_circular`,
  `balanced`, and `helical` after the rest of the analyzer had been generalized.
  The v2 run contains only `balanced_stellarator`, so the dependent analysis
  failed on `max()` over an empty group.
- Affected evidence: job `51616` and the first analysis pass under
  `axis_surface_prior_balanced_v2_20260901_1dcfd18`. The failure occurred after
  `summary.json`, `condition_summary.csv`, and both distribution PNGs were
  written. Those four artifacts remain valid; only `representative_samples.json`
  and its HTML files were absent.
- Fix and verification: commit `5cbb97a` derives the family set from input rows
  and tests the single v2 family explicitly. CPU-only job `51627` reran analysis
  against the frozen 6000 scored rows and produced
  `representative_samples.json` plus both requested coil HTML files. The test
  suite passed 35 tests after the fix.
- Promotion/reporting blocker: resolved. The aggregate artifacts and repaired
  representative artifacts are delivered together in the canonical report.

## CORR-20260901-49 - Three.js representative views clipped coils on narrow screens

- Severity/status: low / resolved before report delivery.
- Discovered by: model during the required rendered-deliverable audit.
- Error: the representative-coil HTML used a camera distance fixed from the
  three-dimensional bounding-box diagonal. Narrow viewports reduced horizontal
  field of view without increasing that distance, clipping the right edge of
  the coil set.
- Affected evidence: only the responsive framing of the two v2 representative
  HTML files. Coil coordinates, scores, tables, figures, and conclusions are
  unchanged.
- Fix and verification: the Three.js writer now fits a bounding sphere using
  the smaller of the vertical and horizontal half-field angles, repeats that
  fit on resize, and wraps the overlay label. A regression test checks the
  generated responsive-camera logic; desktop and 390x844 rendered views are
  checked before delivery.
- Promotion/reporting blocker: resolved after both delivered HTML files pass
  the responsive rendering check.

## CORR-20260901-50 - Consolidated loader lost standardized-data priority

- Severity/status: high / fixed before the analytic-prior Adam200 launch.
- Discovered by: model while auditing the direct-data optimizer entry point for
  the user-requested six-GPU experiment.
- Error: consolidation commit `c7438c0` replaced the former optimizer-specific
  loader with `scripts.flow_runtime.load_initial_noise`, whose section list did
  not include `data_prior_screening`. A payload retaining both standardized
  data parameters and Flow screening metadata would therefore load the Flow
  latent as normalized coil coordinates.
- Affected scope: no accepted default Flow-latent run, balanced-v2 score, or
  historical frozen result is changed. The bug would affect a new current-code
  direct-data launch using such a mixed payload; no such launch is accepted as
  evidence.
- Fix and verification: the shared loader now reads
  `data_prior_screening.normalized_coil_tokens` first and restores the complete
  optimizer-output fallback list. A regression test covers a payload containing
  both standardized-data and Flow metadata. The new analytic-prior experiment
  additionally uses an exact, unclipped start mode and checks step-0 roundtrip
  error before formal submission.
- Promotion/reporting blocker: resolved by the exact-start preflight, ABI-11
  manifest checks, passed smoke `51678`, and six live formal trajectories.

## CORR-20260901-51 - Optimizer provenance check used an undefined root name

- Severity/status: medium / fixed before formal Adam200 sampling.
- Discovered by: model through the mandatory three-step GPU smoke job `51640`.
- Error: `scripts/optimize_flow_latent.py` defined the repository root as
  `REPO_ROOT` but passed an undefined `PROJECT_ROOT` to
  `repository_provenance()`. Static compilation succeeded because the name was
  resolved only when `main()` reached the provenance check.
- Affected scope: smoke job `51640` exited before evaluator construction or
  any optimization update. Dependent arrays `51641` and `51642` never started
  and were cancelled with analysis job `51643`. No score, trajectory, accepted
  conclusion, or current default is affected. Prepared selection manifest from
  job `51639` remains a valid record of its frozen failed-launch attempt.
- Fix and verification: the call now uses `REPO_ROOT`. An AST regression test
  rejects any loaded `PROJECT_ROOT` name in the optimizer entry point, in
  addition to the existing compile and protocol tests. A new frozen commit and
  a new smoke-gated job chain are required for the formal experiment.
- Promotion/reporting blocker: resolved by final smoke job `51678`, which
  reproduced the selected initial score within 0.0313 and completed three
  updates with zero parameter roundtrip error.

## CORR-20260901-52 - Score-only records were treated as axis-continuation records

- Severity/status: medium / fixed before formal Adam200 sampling.
- Discovered by: model through replacement smoke job `51653`.
- Error: the shared optimizer passed any recorded native score through the
  strict axis-continuation path. Balanced-v2 score-only rows intentionally omit
  `axis_R` and `axis_Z`, so initial evaluation raised `KeyError` before calling
  the evaluator.
- Affected scope: smoke job `51653` stopped before the initial native call or
  any Adam update. Dependent arrays `51654` and `51655` never started and were
  cancelled with analysis job `51656`. Existing Flow starts with complete axis
  diagnostics and all accepted historical trajectories are unchanged.
- Corrected behavior: a recorded result supplies the initial strict hint only
  when both axis coordinates exist and are finite. Otherwise the initial
  ABI-11 evaluation performs its standalone axis search. Every subsequently
  accepted center still supplies a complete strict continuation hint.
- Verification: a regression test covers absent, partial, nonfinite, and valid
  recorded diagnostics. A third frozen launch and successful three-update GPU
  smoke are required before six-card optimization begins.
- Promotion/reporting blocker: resolved by final smoke `51678` and formal
  arrays `51679`/`51680`; every first trajectory produced live iteration
  history.

## CORR-20260901-53 - Per-trajectory cap was shorter than observed nc=4 Adam200

- Severity/status: high / fixed before accepting a formal trajectory.
- Discovered by: model during the required six-worker live-progress audit of
  arrays `51664` and `51665`.
- Error: the worker allowed five hours overall but passed a fixed 2400-second
  cap to every optimizer. The first `nc=4,nfp=8` case took about 21 seconds per
  update, projecting roughly 70 minutes for Adam200, so the inner cap would
  have converted a scientifically valid slow trajectory into a runtime failure.
  The prior p90 runtime estimate did not cover this balanced-v2 cost tail.
- Affected scope: arrays `51664` and `51665` were cancelled about four minutes
  after launch, before any 200-step trajectory was accepted. Their incomplete
  directories remain operational evidence only. Smoke job `51663` remains
  valid because it completed all three requested steps and passed every gate.
- Corrected behavior: each trajectory may use at most 7200 seconds, further
  bounded by the worker's remaining 17400-second budget minus 300 seconds for
  cleanup. The worker still stops opening new cases when less than its
  2400-second reserve remains. A unit test covers both the two-hour cap and the
  shrinking remaining-budget case.
- Promotion/reporting blocker: resolved at launch. Arrays `51679`/`51680`
  showed all six first trajectories advancing from 3 to 28 updates with
  7200-second trajectory limits and no failure artifact.

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
