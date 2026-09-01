# Correction Ledger

Last reviewed: 2026-08-30 (Asia/Shanghai).

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
