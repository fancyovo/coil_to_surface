# Correction Ledger

Last reviewed: 2026-09-04 (Asia/Shanghai).

This ledger is intentionally selective. Add an entry only when:

1. the model failed to notice an error and the user later identified it; or
2. an objectively simple mistake consumed substantial time or repeated effort
   to correct.

Implementation failures noticed immediately and fixed during the same workflow
are normal development history and do not belong here. Qualifying entries state
the corrected conclusion, retained evidence, and containment. The former
1810-line ledger, including removed operational minutiae, remains recoverable
from Git at `bf077a8:memory/CORRECTIONS.md`.

## CORR-20260825-01 - Historical 2D evidence was presented as the current method

- Qualification: user-reported after the model had conflated protocol versions.
- Error: a 32-case Flow-versus-data control using 2 directions and 100 steps was
  discussed as evidence for the 64-direction, 200-step method.
- Correction: the coordinate-causality claim from that control is retracted.
  Its 48-condition initialization evidence remains usable. The independent
  309-pair corpus really used 64 directions and 200 steps, but compares two
  complete recipes rather than isolating coordinates.
- Containment: 2D launchers are inert, current defaults are machine-readable,
  and historical reports carry explicit deprecation notices.

## CORR-20260828-01 - A historical 2D CLI default remained executable

- Qualification: user raised the stale script default during the mainline
  credibility audit; fixing it required a repository-wide protocol cleanup.
- Error: a generic optimizer and historical launchers could still start
  2-direction optimization after 64D/200 became the accepted method.
- Correction: the 309-pair corpus is not affected because its manifest and
  aggregate direction count prove 64D/200. Later 2D runs are historical only.
- Containment: shared defaults, parser guards, resume checks, inert historical
  launchers, and regression tests prevent accidental 2D execution.

## CORR-20260830-33 - Vacuum G counted coils that did not link the selected axis

- Qualification: user identified the physical risk after inspecting an
  unlinked-coil geometry.
- Error: ABI-10 used the absolute sum of all base currents for vacuum `G`,
  assigning every coil a unit linking number. This changed QS residuals and the
  total score when a coil did not link the selected magnetic axis.
- Correction: ABI-11 uses magnetic-axis circulation, equivalently the
  linking-number-weighted signed current sum. ABI-10 scores for affected
  geometries are historical and cannot be mixed with ABI-11 scores.
- Retained evidence: geometry, Biot-Savart fields, linking audits, converged
  Simsopt surfaces, Poincare plots, and DESC results do not inherit the shortcut.
- Containment: ABI-11 is the mainline default; active scoring and resume paths
  reject ABI-10. See `reports/abi11_default_promotion_20260830.md`.

## CORR-20260831-40 - A report was delivered only inside a hidden worktree

- Qualification: user reported that the promised local report was not visible.
- Error: the canonical report existed on the experiment branch, but the primary
  checkout remained elsewhere and the supplied path was inside `.worktrees/`.
- Correction: numerical results and the branch commit were valid; delivery and
  discoverability were not.
- Containment: explorations switch the visible checkout, and finalized reports
  plus assets are mirrored and link-checked in `_shared_reports/`.

## CORR-20260901-44 - QUASR marginal standardization was treated as a neutral prior

- Qualification: user identified the geometry problem after inspecting the
  high-score coils and the failed enrichment experiment.
- Error: independent Gaussian sampling in featurewise-standardized QUASR
  coordinates was treated as a broad geometry prior. It does not restore
  cross-mode, intercoil, topology, curvature, or torsion structure.
- Correction: the failed Online RWCFM runs establish failure for that specific
  prior, not for RL with structured physical priors. Their measured abundance
  and runtime data remain valid.
- Containment: the Gaussian prior is deprecated for new physical exploration;
  subsequent work uses explicit axis, winding-surface, and contour construction.

## CORR-20260901-47 - A tokamak-like prior incorrectly passed the visual gate

- Qualification: user rejected the geometry after the model had declared the
  first visual check successful.
- Error: smoothness and linking were accepted while the nearly circular axis,
  weakly varying winding surface, and uniform TF-like coil family violated the
  intended stellarator morphology.
- Correction: the 953 partial v1 rows are mechanically reproducible but do not
  estimate the requested prior. The prior needs visible axis bending, matching
  winding-surface variation, and nonuniform stellarator shaping.
- Containment: v1 was stopped and invalidated; later prototypes require explicit
  geometry review before large score batches.

## CORR-20260901-60 - Independent full-evaluation work was serialized

- Qualification: user reported the two-hour latency while six GPUs were
  available.
- Error: independent `a`, `s`, and sample evaluations were chained for
  operational convenience. The physical results were valid, but most wall time
  came from orchestration.
- Correction: candidate evaluations are independent until their selection
  barrier. The standard workflow now submits one allocation, runs candidates
  concurrently, selects both `a` and `s` in code, and continues through
  Simsopt and DESC without agent-side staging.
- Containment: `evaluation/full_physical/submit_full_evaluation.sh` is the
  routine entrypoint; staged submitters are diagnostic tools only.

## CORR-20260902-66 - Prior handedness and the QH target were mismatched

- Qualification: user noticed the reversed Boozer-contour slope and questioned
  why nominally random samples systematically reached the wrong hand.
- Error: balanced-v2 and compact-v3 generated an all-negative-iota valid
  population while scoring the positive-hand `QH_(1,+1)` target. Early
  explanations also attributed the sign directly to the contour scalar without
  isolating the construction axis.
- Correction: all 569 valid compact-v3 starts and all 120 saved trajectories
  were negative-hand. Signed rescoring and negative-target Adam reached
  `80.8523` and `84.9308`. The leading bias was the fixed-sign reference-axis
  harmonic; surface and contour contributions were not independently isolated.
- Retained evidence: frozen positive-target scores, coil metrics, surfaces,
  Poincare data, and DESC results remain valid under their recorded definitions.
- Containment: later axis-flip experiments use explicit handedness metadata and
  report signed QH targets. See
  `reports/axis_surface_prior_handedness_audit_and_negative_optimization_20260902.md`.

## CORR-20260902-67 - Remote work bypassed the documented WSL master

- Qualification: user stopped an attempt to use the wrong server connection
  route and pointed to `REMOTE_CODEX_INSTRUCTIONS.md`.
- Error: after context loss, the model tried direct SSH and a temporary local
  agent/key route instead of the authenticated WSL master.
- Correction: no credential was stored, no job was submitted, and no remote
  state changed through the incorrect route.
- Containment: remote work uses only WSL `Ubuntu` and the `ustc107` master.
  The remote document is read in a new empty conversation or after compaction;
  routine access performs only its basic connectivity check.

## CORR-20260902-73 - A simple cross-stage consistency gate was applied too late

- Qualification: simple configuration plumbing caused three invalidated
  axis-flip attempts and substantial GPU/time loss before the root mismatch was
  isolated.
- Error: successive runs lost the screened axis, changed numeric representation,
  and finally compared screening with an optimizer using different surface
  settings. The first version checked consistency only after Adam200.
- Correction: v1-v3 Adam outcomes are quarantined. Their own screening
  handedness observations remain usable. V4 scores identical representable
  tokens under one shared score configuration and validates the saved axis and
  step-0 score before the first update.
- Containment: shared score configuration, axis retention, exact representation,
  and the pre-update gate are regression-tested. V4 completed 50 valid Adam200
  trajectories with zero gate failures.

## CORR-20260903-77 - The RL plan incorrectly proposed fixing the current channel

- Qualification: user challenged the plan before implementation.
- Error: the proposal treated fixed `nfp=8,nc=3` as making current
  deterministic and described coil permutation as necessary for a Transformer
  that was already permutation-equivariant.
- Correction: q0 starts with equal currents, while Adam targets may learn
  unequal relative allocations under the fixed total-current projection.
  Random whole-coil permutation is optional representation symmetrization.
- Containment: the implemented Flow retains a trainable current channel and
  tests current round trips and permutation equivariance.

## CORR-20260903-91 - The first coil-shrink map changed coil shape

- Qualification: user clarified that each coil should undergo one similarity
  transform about one anchor point.
- Error: pointwise scaling about independently nearest axis points imported
  axis-following distortion and high-frequency refit error, so it did not
  implement the requested shape-preserving shrink.
- Correction: the pointwise scan remains labeled diagnostic. The replacement
  uses one fixed magnetic-axis anchor per base coil and preserves shape exactly
  apart from translation and uniform scale.
- Containment: tests verify proportional radius change, current preservation,
  and machine-precision Fourier reconstruction. See
  `reports/axisflip_case23_coil_shrink_results_20260903.md`.

## CORR-20260904-96 - The score-gradient Flow plan reintroduced eliminated transport variables

- Qualification: user-reported missed method error and scorer omission.
- Error: the proposed implementation rebuilt an explicit `x+epsilon*d` target,
  introduced `rho` and per-sample transport weights, and estimated the round
  cost as if 128 endpoint scores were independent single-point calls. It also
  described the scorer only as generic ABI-11 and omitted the active R04 coil
  curvature variant.
- Correction: the first-order method uses the raw Adam-compatible
  `g^T grad_x ell_theta` term directly. Its `rho` and displacement factors are
  derivation-only constants absorbed into the loss coefficient; no shifted
  target or target score is evaluated. The 64-direction `g` must use the same
  `gradient_probe`/`LocalFullGradientEstimator` path as current Adam: one
  `BatchCoilFieldGpu` query containing 128 endpoints per center, with shared
  center capture, axis tracing, psi fit, and `score_local_batch`.
- Retained evidence: no code, branch, checkpoint, or job was changed by the
  mistaken plan. Existing R012 results remain valid under their frozen
  manifest. R035 samples show `gradient_wall_s` about `5.13--5.15 s` and Adam
  iteration wall about `6.6--7.0 s` per sample on the current path.
- Containment: the new protocol must pin ABI-11 R04 library SHA-256
  `7b21e66329a23f18ef3cfab408d0a21410acea3c8d10d8b2463ccee7f5a3969f`,
  curvature p95 radius `0.04 m` (`25 m^-1`) and maximum scale `35 m^-1`,
  rather than the standard ABI-11 library. The corrected cost model and
  coordinate/path identity are required in its manifest and tests.

## CORR-20260905-101 - Full evaluation accepted a self-intersecting Boozer surface

- Qualification: user reported the visibly self-intersecting surface after the
  full physical evaluation.
- Error: the workflow treated Simsopt LS/Newton residual convergence and finite
  volume as sufficient. It selected the largest finite-volume candidate without
  a geometric injectivity/nestedness gate, and it did not preserve the pre-LS
  surface coefficients for a direct before/after check. This allowed a surface
  with residual `~1e-13` but `DESC nested_initial=false` and
  `nested_final=false` to be presented as a usable physical surface.
- Correction: the GPU path did feed the intended level-surface construction
  into Simsopt (`extract_surface_backend=gpu`; selected-level fit RMS
  `4--6e-9 m`; positive radial roots). However, the initial-best candidate
  reports `radius_max=0.08 m`, exactly the configured hard cap, so the
  complete toroidal grid may contain clipped rays; the run did not save the
  pre-LS DOFs or a clipping fraction. This makes the initial geometry only
  partially audited. The gross failure is nevertheless downstream: the
  unconstrained Simsopt penalty LS moved to a degenerate Fourier branch from
  an initial Boozer residual of about `16.5`; its exact Newton polish took zero
  iterations and did not repair the geometry. The saved initial-best section
  winds six times around the magnetic axis, and the Adam20 section has local
  angular folds.
- Retained evidence: GPU extraction and fit timing, raw residuals, final
  Fourier coefficients, DESC nesting/force diagnostics, and the cross-section
  diagnostic image remain valid as evidence of this failure. Native ABI-11
  screening scores are unaffected; the two full physical evaluations are
  quarantined and cannot support a claim of a valid nested surface.
- Containment: full evaluation must save pre-LS DOFs and reject any candidate
  with non-injective section tests, negative/near-zero surface Jacobian, or
  failed DESC nesting before selecting by volume. The diagnostic image is
  mirrored at `_shared_reports/axisflip_r012_surface_self_intersection_diagnostic.png`.

## CORR-20260905-102 - Direct raw Boozer residual was compared with alpha+nu residual

- Qualification: user-reported discrepancy after the full evaluation showed an
  initial residual around `16.5`, while the historical calibration reported
  values around `1e-3`.
- Error: the pinned R012 full-evaluation path (`95ed6cf`) fitted the GPU level
  surface directly with the geometric polar angle and measured
  `np.linalg.norm(boozer_surface_residual(...))` before alpha+nu. That is an
  unweighted raw norm over collocation points, not the normalized `1/|B|`
  alpha+nu metric used by the historical calibration. The geometric theta also
  has the opposite orientation from the Simsopt Boozer theta convention, so the
  dominant error is tangential coordinate mismatch. The formal alpha+nu route
  was therefore bypassed and the two residual definitions were treated as
  comparable.
- Correction: on the same R048 candidate, direct raw residual `16.5313` had a
  normalized value `0.2223`; its tangential part was `0.2218` and its normal
  part only `0.0150`. The correct alpha+nu route reduced the normalized
  residual from `0.14350` (alpha only) to `0.01647`, kept the nu map positive
  (`1+Dnu` in `[0.6934, 1.2661]`), and flattened local-G relative spread from
  `0.14266` to `0.000984`. Standard Simsopt LS/Newton on that alpha+nu surface
  reached dense normalized residual `0.0009976`, consistent with the historical
  `1e-3` scale. This does not indicate an alpha+nu algebra/sign bug.
- Retained evidence: the raw direct residual decomposition, alpha/nu summaries,
  and dense standard-chain summary remain diagnostic evidence only. Native
  ABI-11 score and optimization results are unaffected.
- Containment: formal full evaluation must invoke and record the alpha+nu
  preparation before Simsopt, report weighted/normalized residuals with their
  definitions, and quarantine direct geometric-point-cloud results. A future
  audit must compare residuals only after matching weighting, normalization,
  grid, and parameterization.
- Resolution: corrected jobs `54321` and `54323` reran both samples through the
  enforced alpha+nu provenance chain. Their standard LS/Newton outputs are the
  current full-evaluation results; job `54223` remains superseded diagnostic
  evidence only.

## Entry Template

```text
## CORR-YYYYMMDD-NN - Short title

- Qualification: user-reported missed error | simple error with substantial rework.
- Error:
- Correction:
- Retained evidence:
- Containment:
```

Do not add an entry merely because a command, test, job, or first implementation
failed and was promptly corrected.
