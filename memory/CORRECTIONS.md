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
