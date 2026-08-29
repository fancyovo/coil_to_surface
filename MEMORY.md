# Current Project Memory

> Current truth, verified 2026-08-29 (Asia/Shanghai). This is a compact routing
> and safety file, not a work log. Older material is indexed under `memory/`.

## Baseline

- The authoritative private development baseline is `main` after the
  2026-08-28 consolidation. Verify the actual checkout and HEAD with Git at the
  start of repository work; do not infer them from this file.
- The active checkout is the exploration branch
  `codex/qh-basin-atlas-random-survey`, created from
  `main@4c446b6d3c923d2ace38a6be7b6e57b31cf6285a`. It studies QUASR geometric
  structure and independent standardized-data random starts; it does not
  replace the current QH default.
- The sanitized open-source sibling `../opensource_staging` was verified clean
  on `main` at `89d30e92b7b05687637f2589f649b8def8d3c8b7`. The private baseline
  carries the current public screening and optimization interfaces while
  retaining private research evidence.
- Before remote work, inspect the scheduler and read
  `REMOTE_CODEX_INSTRUCTIONS.md`. Old logs and run directories never establish
  current job state; only job IDs recorded under Active Exploration do.
- Many thousands of pre-existing untracked audit, bundle, run, and generated
  files are present. Preserve them and stage source changes explicitly; verify
  the live count instead of treating a recorded count as stable.

## Current QH Default

- Protocol ID: `qh-flow-screen32-adam200-64d-v1`.
- Screen 32 random Flow starts. Optimize the selected latent with 200 Adam
  updates, 64 fresh random-orthogonal centered directions per update,
  perturbation `0.005`, learning rate `0.02`, beta `(0.7, 0.999)`, and FP32
  RK4-128 decoding.
- Canonical entry points are `scripts/screen_flow_starts.py` and
  `scripts/optimize_flow_latent.py`. Shared defaults and protocol metadata live
  in `flow_matching/optimization.py`. Compatibility entry points must resolve
  to the same values.
- Exactly two directions is deprecated historical evidence and is hard-blocked
  in current Python entry points. Historical shell launchers exit before their
  old 2D settings can run.
- Resume requires an exact machine protocol match and matching repository
  commit/dirty state. Legacy or unclassified manifests cannot enter current run
  state through `--resume`.
- A nondefault experiment must identify itself as experimental in its manifest.
  An explicit future request to study two directions requires a new protocol,
  branch, launcher, and review of the guard; historical 2D files remain inert.
- Full rules, manifest fields, and the historical protocol registry are in
  `memory/PROTOCOLS.md`.

## Active Exploration

- Registered experimental protocol `qh-data-gaussian-global-survey-v1` draws
  independent `N(0,1)` values in standardized coil-data coordinates and scores
  them with independent current ABI-10 standalone global evaluations. Each GPU
  process must first reproduce the current frozen-case reference
  `94.6254147736` within `1e-5`; formal calls pin `psi_solver_mode=2` and
  `alpha_solver_mode=2`, with all remaining fields at library defaults. All 33
  supported `(nfp,n_base_coils)` groups are sampled exactly evenly across six
  workers.
- This survey measures score-tail prevalence. It preserves the `score >= 20`
  tail and deterministic reconstruction metadata for every sample. A separate
  stratified 64-direction, 200-step data-space Adam follow-up is required to
  estimate the optimizable-basin rate and its additional accelerator cost;
  score thresholds alone are not that rate. The immutable original selection
  contains all 10 samples at score at least 20 plus four uniform low-band
  controls; all four controls have `no_axis` status and cannot enter the frozen
  optimizer. The user-authorized 2026-08-29 extension preserves those four as
  diagnostics, takes a census of all 34 `score >= 10` samples, and samples 38
  additional `score < 10,status=ok` controls by base-coil count. It therefore
  contains 76 records and 72 actual Adam200 trajectories, with 12 trajectories
  per GPU across six workers. The estimated cost is 17.59 serial GPU-hours and
  2.93 parallel hours from the prior 309-case mean. Preparation job `47487`
  validated the full 76/72 selection, reconstruction, and hashes. First worker
  arrays `47488` and `47489` are invalid infrastructure attempts: all 72 Adam
  cases exited before iteration 1 because the consolidated optimizer referenced
  undefined `PROJECT_ROOT`; four expected `no_axis` diagnostics were recorded.
  No optimization result exists from those arrays. A clean replacement run is
  required after `CORR-20260829-13`.
- QUASR structure analysis uses all 170,755 QH entries, condition-stratified
  permutation/rotation/parameterization-robust geometric descriptors, stable
  partitions, nearest-neighbor density, medoids, and a reusable novelty
  reference. The hierarchy first hard-splits 33 supported
  `(nfp,n_base_coils)` groups, then selects 2--3 geometric partitions inside
  each group. The 2026-08-28 atlas therefore has 72 leaf partitions; its leaf
  counts for `n_base_coils=1..5` are `[11,15,14,17,15]`. Different base-coil
  counts never share a group or leaf. The sample-weighted within-group effective
  count is 1.599 and the sample-weighted largest-partition share is 81.28%.
  A second aggressive atlas hard-splits only the five `n_base_coils` values and
  permits geometrically similar samples from different `nfp` values to share a
  leaf. Its fit is balanced across `nfp`; every leaf records `nfp` composition
  and each hard group records cluster/`nfp` normalized mutual information. The
  full aggressive result selects two leaves for each base-coil count, for 10
  leaves total. All 10 contain 5--7 `nfp` values; sample-weighted cluster/`nfp`
  normalized mutual information is 0.0932. Strict and aggressive references
  are complementary novelty checks. Full results and artifact hashes are in
  `reports/quasr_qh_structure_atlas_20260829.md`.
- Current-runtime validation job `46869` established the frozen-case reference
  `94.6254147736` on driver `580.173.02` with production-library SHA-256
  `565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729`.
  Failed setup/gate attempts and the invalid 44,154-sample preparation are
  contained and recorded in `memory/CORRECTIONS.md`; none produced formal data.
- Formal run `qh_data_gaussian_global_formal_10h_v2_20260828` completed
  46,167/46,167 evaluations on jobs `46898`, `46901`, and `46905`. Every one of
  the 33 conditions has 1,399 samples. Status counts are 33,107 `no_axis`,
  10,182 `no_surface`, 1,887 `drift_rejected`, 104 `flux_rejected`, and 887
  `ok`. Scores at least 10/20/30/40/50 occurred 34/10/5/3/0 times; maximum score
  is `47.2301782`. The score-at-least-50 rate has 95% Wilson upper bound
  `8.3201e-5`. Six workers consumed 58.17 GPU-hours over 10.02 hours elapsed.
  Summary job `47462` completed with empty stderr; all 10 retained candidate
  hashes match. Full evidence and limitations are in
  `reports/qh_data_gaussian_global_survey_acceptance_20260829.md`.
- Aggregating the frozen survey by `n_base_coils` gives sample counts
  `[6995,9793,9793,9793,9793]` and evaluator-`ok` rates
  `[4.3746%,2.3180%,1.6951%,1.1539%,0.7659%]`. Score-at-least-20 counts are
  `[7,1,1,1,0]`; their small-count Wilson intervals and all other thresholds
  are recorded in the survey acceptance report and `n_coils_breakdown.json`.

## Current Evaluator And Physical Contract

- The production native evaluator is C++/CUDA ABI 10. Current score-library
  SHA-256 is
  `565c32073b145d97a1f2244705fb06e4b3458ce798cd74d0c97ee4e0129dc729`.
  Intentional rebuilds require fresh numerical validation before promotion.
- Current evaluator defaults include psi grid 48, cubic
  `iota(u)` with `u=psi/psi_edge`, strict axis-hint mode 2, continuous surface
  confidence, and history-independent standalone/corpus axis search. Optimizer
  continuation alone may use the previous validated axis as a strict hint.
- Native score is a bounded screening objective, not a Boozer-surface or MHD
  certificate. Selected candidates require the fixed workflow in
  `docs/精简线圈评估流程.md` and `evaluation/full_physical/README.md`.
- Surface-fit radius `a` and level `s` are sample-specific results, never global
  defaults. Acceptance requires independent dense residuals, standard
  Simsopt LS/Newton, Poincare nesting, face QA/QH/QP, and DESC diagnostics.

## Accepted Numerical Conclusions

- Flow provides reliable initialization evidence and is the current practical
  high-score optimization method. Direct standardized-data optimization is a
  valid, faster baseline, but the current 309-pair evidence compares complete
  recipes rather than isolating coordinate causality.
- The 309-pair corpus actually used 64 directions and 200 steps. Its manifest
  and aggregate count of 3,955,200 orthogonal directions support this. Latent
  optimization won 201/309 pairs; median paired best-score advantage was 0.997
  with 95% interval `[0.799, 1.312]`; score-at-least-92 rates were 23.0% versus
  0.32%. Latent/data settings differed in learning rate and perturbation, so
  this result does not prove an intrinsic coordinate effect.
- The 32-case 2026-08-24 coordinate control used 2 directions and 100 steps.
  Its coordinate-causality conclusion is retracted. Its separate 48-condition
  initialization evidence remains usable.
- With the same ABI-10 library and frozen input, fully evaluated sample
  `p107_37034_3_000018_step0150` recorded `94.6368682` on 2026-08-19 and
  reproducibly scores `94.6254148` under the 2026-08-28 runtime. Input SHA-256
  is `6ee6f8e1f0290ec49093596a5f95b7f2aac98c61d51af3cad59410a771b7e8c1`.
  See `reports/qh_min_face_qh_full_evaluation_20260819.md` and
  `CORR-20260828-09`.
- The 10,000-step run whose best was `93.3672653` at step 4341 remains valid
  evidence for its historical constant-iota objective and for stagnation under
  that frozen recipe. Its score is not the current maximum and is not directly
  comparable with the current cubic-iota objective.

## Invalid Or Historical High-Risk Material

- Every 2-direction optimization result and launcher is historical. Labels such
  as "default", "production", or "standard" inside an old artifact describe
  its old local context and have no authority over the current method.
- `reports/qh_flow_initialization_vs_optimization_control_20260824.md` preserves
  the erroneous 2D/100 coordinate-control history. Only its explicitly retained
  initialization evidence may be cited.
- ABI-9 and earlier score results that predate current-sign, linked-current
  scale, physical-volume weighting, fixed-point budget, topology, or continuous
  surface corrections cannot be numerically mixed with the current score.
- Old CEM, hybrid Adam, analytic-gradient G2-G5, BFGS, proxy, trust-region,
  reduced-zero-tail, and constant-iota routes are research history, not current
  defaults. Their artifacts remain available for provenance.
- `CODEX_HANDOFF.md` is a tombstone for a July DESC handoff, not current task
  state. Its exact former contents are archived.

## Work Governance

- Start new exploration from the consolidated `main`. Use a `codex/` branch
  when an experiment changes methods, defaults, or shared code.
- User acceptance of a method as the default means promotion to `main` and
  coordinated updates to implementation defaults, protocol metadata, launchers,
  tests, current docs, `memory/DECISIONS.md`, and any affected correction entry.
- Preserve frozen experiment manifests. Record observed metadata separately
  from current re-evaluation; never reconstruct settings from prose or CLI
  defaults when a manifest exists.
- Record every discovered error in `memory/CORRECTIONS.md` in the same turn,
  including impact, retained conclusions, evidence, containment, and status.
- External documents and figures must pass `memory/WRITING.md`. Use direct,
  affirmative explanations; remove defensive contrast, ambiguous references,
  version leakage, and irrelevant claims after generation.

## Memory Map

- `memory/README.md`: loading and maintenance architecture.
- `memory/PROTOCOLS.md`: current protocol, run gates, and deprecated registry.
- `memory/CORRECTIONS.md`: append-only error and correction ledger.
- `memory/DECISIONS.md`: active decisions and promotion semantics.
- `memory/WRITING.md`: external document and multimodal review rules.
- `memory/HISTORY.md`: concise chronology and evidence pointers.
- `memory/archive/`: immutable pre-restructure snapshots; never read by default.
- `MEMORY_archive_20260808.md`: legacy full-history archive; historical only.

Update this file only when current truth or routing changes. Put chronology in
`memory/HISTORY.md`, errors in `memory/CORRECTIONS.md`, and detailed evidence in
reports. Never store credentials.
