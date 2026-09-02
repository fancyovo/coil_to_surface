# Concise Project History

Last reviewed: 2026-09-02 (Asia/Shanghai). This file routes historical questions;
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
