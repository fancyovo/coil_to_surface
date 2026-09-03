# Concise Project History

Last reviewed: 2026-09-03 (Asia/Shanghai). This file routes historical questions;
reports and immutable archives retain the detailed record.

## Timeline

- Through 2026-08-08: coils-to-axis, geometric surface label, flux calibration,
  straight-field-line fitting, near-Boozer initialization, native score design,
  Flow training, CEM, early finite-difference Adam, and abandoned analytic-
  gradient routes were developed. Full detail is in
  `../MEMORY_archive_20260808.md` and the reports it cites.
- 2026-08-06: early score-fast matrices and beta sweeps used two directions.
  These are deprecated historical optimizer studies.
- 2026-08-10: psi grid 48, cubic iota, and strict axis-hint mode 2 became current
  score/evaluation settings. Cubic-iota and direction-reuse optimization studies
  from this period still used two directions and remain historical.
- 2026-08-13 evidence set: the 309-pair Flow/data corpus froze 64 directions and
  200 steps. It later supported the complete-recipe comparison summarized in
  `reports/qh_data_space_large_scale_validation_20260825.md`.
- 2026-08-19: complete physical evaluation verified current-objective sample
  `p107_37034_3_000018_step0150` at score `94.6368682`. See
  `reports/qh_min_face_qh_full_evaluation_20260819.md`.
- 2026-08-24: a 32-case coordinate control ran with 2 directions and 100 steps,
  despite later being discussed as though it represented the 64D/200 protocol.
  Its coordinate conclusion is invalidated; see `CORRECTIONS.md`.
- 2026-08-25: the large-scale report corrected that mismatch. It retained Flow
  initialization evidence and interpreted the 309 pairs as a complete-recipe
  advantage, leaving pure coordinate causality unresolved.
- 2026-08-27: sanitized public `main` reached commit `89d30e9`, exposing stable
  evaluator and 32-screen/64D/200 optimization interfaces.
- 2026-08-28: the private history was consolidated onto `main`; current defaults
  were centralized, 2D execution was blocked, old reports were bannered, and
  memory was split into current, protocol, correction, decision, writing,
  history, and immutable archive layers.
- 2026-09-02: compact-flexible-v3 analytic-prior scoring completed 3600 ABI-11
  samples and measured a `15.8056%` initial-valid rate. A frozen random sample
  of 120 valid starts completed direct-data Adam200; 65 reached score 50 by
  update 50 and 97 by update 200. Full evaluation then accepted representative
  samples `axisv3_case_01341` and `axisv3_case_02832` at standard surfaces
  `s=0.24` and `s=0.36`; the latter's vacuum Poincare map retained an internal
  island-chain signature. Canonical evidence is in
  `../reports/axis_surface_prior_compact_flexible_v3_adam200_results_20260902.md`.
- 2026-09-02: a signed-helicity audit established that all 569 valid source
  samples and all 120 saved Adam200 trajectories occupied the negative-iota
  mirror branch. Explicit negative-target Adam200 raised two representative
  endpoints to `80.8523` and `84.9308`; the old same-handed abundance
  interpretation was withdrawn. A subsequent causal review identified the
  fixed-sign construction-axis harmonic as the leading bias and withdrew the
  unisolated scalar-field-only attribution. Canonical evidence is in
  `../reports/axis_surface_prior_handedness_audit_and_negative_optimization_20260902.md`.
- 2026-09-02: construction-axis-flip stream v1 was invalidated after its compact
  screening records omitted `axis_R/axis_Z`, allowing optimizer step 0 to select
  a different magnetic-axis branch. Remaining workers were canceled and all
  Adam outcomes quarantined. V2 moves the consistency gate before Adam and
  requires strict continuation from the screened axis; see `CORR-20260902-71`.
- 2026-09-02: v2 then exposed a hidden `float64` generator to `float32`
  optimizer-start conversion after screening. Its formal arrays were canceled
  with zero completed trajectories. V3 screens the exact optimizer-representable
  reconstruction and pins smoke to regression case 25; see `CORR-20260902-72`.
- 2026-09-02: the pinned v3 smoke proved the remaining score gap came mainly
  from ABI defaults in screening versus optimizer formal surface-selection
  overrides. V3 formal arrays never started. V4 shares the exact configuration
  constructor across both stages; see `CORR-20260902-73`.
- 2026-09-02: the corrected axis-flip v4 stream screened 110 cases and found 56
  valid positive-iota starts. User-directed draining left 50 complete Adam200
  trajectories and six preserved successor partials; analysis `52761` reported
  zero complete-trajectory failures. Thirty-seven complete trajectories reached
  70, nine reached 80, and the maximum was `81.8258373`. Median volume-QS score
  rose from `19.4311` to `59.1199`, while median coil-engineering score remained
  near 69. Canonical evidence is in
  `../reports/axis_surface_prior_axisflip_v4_results_20260902.md`.
- 2026-09-02: full physical evaluation selected axis-flip v4 cases 18 and 23 as
  the maximum-score and higher-engineering representatives. Both accepted
  standard surfaces at `s=0.49`, with volumes `0.076781` and `0.045235 m3` and
  direct `QH_(1,+1)` errors `0.0016255` and `0.0021736`. Both DESC boundaries
  stayed nested while the solves reached the 50-iteration cap. The report and
  machine-readable selection evidence are stored with the v4 result above.
- 2026-09-03: independent Adam2000 continuations of axis-flip v4 cases 18 and
  23 completed all requested updates in parallel. Their ABI-11 screening bests
  were `87.6821` and `89.5535`; volume-QS improved by `18.97/21.86` while the
  coil component decreased by `6.30/5.19`. The long-run supplement is appended
  to `../reports/axis_surface_prior_axisflip_v4_results_20260902.md`.
- 2026-09-03: Students-only full evaluation of case 23's Adam2000 step-1985 best
  accepted `s=0.81`, volume `0.102164 m3`, and direct `QH_(1,+1)=2.3487e-4`;
  `s=1.00` was rejected before standard LS/Newton by the fixed volume-sampling
  budget. Vacuum Poincare ordering and DESC boundary nesting passed. DESC
  reduced mean normalized force by `32.9x` but reached the 50-iteration cap at
  `0.02379`. Evidence is appended to the same v4 report.
- 2026-09-03: 200,000 fixed-condition (`nfp=8,nc=3`) compact-v4 teacher samples
  supported converged q0 Flow distillation and an independent ABI-11 audit.
  Online Adam20 reward-weighted training completed 22 reportable rounds by the
  second snapshot. Initial-score median rose `63.6751 -> 71.9023`, initial
  score-at-least-70 rose `4/64 -> 53/64`, and 1394/1408 samples were valid.
  All near-duplicate rates were zero; descriptor variance contracted `16.6%`
  while effective rank remained within `2.0%` of round 0, so the live run
  continued. See
  `../reports/axisflip_prior_online_rl_results_20260903.md`.
- 2026-09-03: the atomically complete online-RL `round_012.pt` EMA checkpoint
  was frozen by SHA-256 for a separate eight-trajectory Students experiment.
  It retains the current screen32/latent-Adam200/64D/RK4-128 mechanics at
  `nfp=8,nc=3`; later live-policy updates cannot alter the input.
- 2026-09-03: initial frozen-policy array `53046` exposed and preserved a
  QUASR-only hardcoded checkpoint-step guard before Adam. Corrected array
  `53049` completed all eight 64-direction pipelined Adam200 trajectories at
  commit `8f5d57a`. Best-score median was `78.2001`, maximum was `79.6088`, and
  every trajectory improved its volume-QS component. See
  `../reports/axisflip_rl_round12_latent_adam200_results_20260903.md` and
  `CORR-20260903-82`.

## Evidence Routes

- Current method: `../docs/QH原生评分与潜空间优化方法.md`.
- Current Flow/data interpretation:
  `../reports/qh_data_space_large_scale_validation_20260825.md`.
- Initialization evidence and corrected public summary:
  `../reports/summary1/技术报告.md`.
- Score compression and historical 2D studies:
  `../reports/qh_score_evaluation_compression_20260810.md`.
- Historical constant-iota long run:
  `../reports/qh_score_throughput_and_continuous_surface_plan.md`.
- Full physical evaluation contract: `../docs/精简线圈评估流程.md` and
  `../evaluation/full_physical/README.md`.
- Early DESC handoff:
  `archive/CODEX_HANDOFF_pre_restructure_20260828.md`.
