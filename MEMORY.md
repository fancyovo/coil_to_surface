# Current Project Memory

> Current truth, verified 2026-09-01 (Asia/Shanghai). This is a compact routing
> and safety file, not a work log. Older material is indexed under `memory/`.

## Baseline

- The authoritative private development baseline is `main` after the
  2026-08-28 consolidation. Verify the actual checkout and HEAD with Git at the
  start of repository work; do not infer them from this file.
- The sanitized open-source sibling `../opensource_staging` was verified clean
  on `main` at `89d30e92b7b05687637f2589f649b8def8d3c8b7`. The private baseline
  carries the current public screening and optimization interfaces while
  retaining private research evidence.
- Active exploration branch: `codex/axis-surface-prior`, based directly on
  `main@de75f6d`. Protocol
  `qh-axis-surface-contour-prior-score-abi11-v1` failed its visual morphology
  gate and is invalidated. The accepted balanced-v2 experiment completed 6000
  ABI-11 score-only samples under arrays `51614/51615`; repaired analysis job
  `51627` produced the final report and representative coil views.
- Many thousands of pre-existing untracked audit, bundle, run, and generated
  files are present. Preserve them and stage source changes explicitly; verify
  the live count instead of treating a recorded count as stable.

## Current QH Default

- Protocol ID: `qh-flow-screen32-adam200-64d-abi11-v1`.
- Screen 32 random Flow starts. Optimize the selected latent with 200 Adam
  updates, 64 fresh random-orthogonal centered directions per update,
  perturbation `0.005`, learning rate `0.02`, beta `(0.7, 0.999)`, and FP32
  RK4-128 decoding.
- Canonical entry points are `scripts/screen_flow_starts.py` and
  `scripts/optimize_flow_latent.py`. Shared defaults and protocol metadata live
  in `flow_matching/optimization.py`. Compatibility entry points must resolve
  to the same values.
- The default evaluator is ABI 11, score-library SHA-256
  `1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed`.
  Protocol classification includes the actual library hash. ABI 10 cannot
  launch or resume from current entry points.
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

## Current Evaluator And Physical Contract

- The production native evaluator is C++/CUDA ABI 11. Current score-library
  SHA-256 is
  `1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed`.
  Intentional rebuilds require fresh numerical validation before promotion.
- Vacuum `G` uses the selected magnetic axis Ampere circulation. Equivalently,
  it sums signed physical-coil currents weighted by their oriented linking
  numbers. Unlinked coils make no contribution to that axis current.
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
- The frozen ABI-10 309-pair corpus actually used 64 directions and 200 steps.
  Its manifest
  and aggregate count of 3,955,200 orthogonal directions support this. Latent
  optimization won 201/309 pairs; median paired best-score advantage was 0.997
  with 95% interval `[0.799, 1.312]`; score-at-least-92 rates were 23.0% versus
  0.32%. Latent/data settings differed in learning rate and perturbation, so
  this result does not prove an intrinsic coordinate effect or ABI-11 outcomes.
- The 32-case 2026-08-24 coordinate control used 2 directions and 100 steps.
  Its coordinate-causality conclusion is retracted. Its separate 48-condition
  initialization evidence remains usable.
- ABI-11 strict-axis replay validated 44 linked and unlinked samples against
  independent linking-number current sums; maximum absolute `G` discrepancy
  was `1.0921e-8`. All 38 replayed Adam200 endpoints remained at score 50 or
  above. See `reports/abi11_default_promotion_20260830.md`.
- The 10,000-step run whose best was `93.3672653` at step 4341 remains valid
  evidence for its historical constant-iota objective and for stagnation under
  that frozen recipe. Its score is not the current maximum and is not directly
  comparable with the current cubic-iota objective.

## Invalid Or Historical High-Risk Material

- Every 2-direction optimization result and launcher is historical. Labels such
  as "default", "production", or "standard" inside an old artifact describe
  its old local context and have no authority over the current method.
- ABI 10 is historical. Its all-current vacuum-`G` shortcut is invalid for
  geometries containing coils that do not link the selected magnetic axis.
  ABI-10 trajectories remain frozen evidence for their original objective and
  cannot resume through current entry points. ABI-10 scores cannot be mixed
  numerically with ABI-11 scores.
- `reports/qh_flow_initialization_vs_optimization_control_20260824.md` preserves
  the erroneous 2D/100 coordinate-control history. Only its explicitly retained
  initialization evidence may be cited.
- ABI-9 and earlier score results that predate current-sign, linked-current
  scale, physical-volume weighting, fixed-point budget, topology, or continuous
  surface corrections cannot be numerically mixed with the current score.
- Old CEM, hybrid Adam, analytic-gradient G2-G5, BFGS, proxy, trust-region,
  reduced-zero-tail, and constant-iota routes are research history, not current
  defaults. Their artifacts remain available for provenance.
- Fully evaluated sample `p107_37034_3_000018_step0150` recorded ABI-10 score
  `94.6368682`. Its physical-evaluation artifacts remain valid; the score is a
  historical ABI-10 value.
- `CODEX_HANDOFF.md` is a tombstone for a July DESC handoff, not current task
  state. Its exact former contents are archived.

## Work Governance

- Start new exploration from the consolidated `main`. Use a `codex/` branch
  when an experiment changes methods, defaults, or shared code, and switch the
  primary user-visible checkout to that branch before substantive work. A
  secondary worktree is an explicit exception for preserved or concurrent work.
- Keep each report's canonical copy on its owning branch and mirror finalized
  reports plus local assets to the primary checkout's Git-excluded
  `_shared_reports/` directory. The mirror remains visible across branch
  switches; branch artifacts and commits remain the provenance source.
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

## Active Analytic-Prior Exploration

- The failed Online RWCFM evidence is specific to an independent Gaussian in
  QUASR featurewise-standardized Fourier coordinates. It does not establish an
  RL limitation for structured physical priors. The canonical failure report
  is on `codex/qh-online-validity-rwcfm-zero@9079fa3` and mirrored in
  `_shared_reports/qh_online_rwcfm_failed_20260901.md`.
- Analytic-prior v1 contracted axis, surface, and contour amplitudes until the
  result resembled circular-axis tokamak coils. Its partial ABI-11 scores cannot
  establish the requested prior's abundance.
- The user accepted only the `balanced_stellarator` v2 morphology for a short
  scoring batch. Protocol
  `qh-axis-surface-contour-balanced-score-abi11-v2` measured 6000 samples over
  26 `nc<=4` conditions with no optimization. Its maximum ABI-11 total score was
  `16.4393`; 56 samples reached 10, and none reached 20 or 50. The coil-component
  median was `72.4114`, with 22.12% at 80 or above. See
  `reports/axis_surface_prior_balanced_v2_results_20260901.md`.
- Scoring commit `1dcfd18` and arrays `51614/51615` produced all 6000 rows with
  zero scoring errors. Analysis `51616` wrote valid aggregate artifacts before
  its v1-family selector failed; commit `5cbb97a` and CPU job `51627` repaired
  representative selection against the unchanged rows. `CORR-20260901-48`
  records the contained analysis error.
- The v2 inner surface is a geometric reference, not a computed magnetic
  surface, and the construction axis is not a verified magnetic axis. The
  registered experiment does not modify the current QH default.
- Registered follow-up
  `qh-axis-surface-contour-balanced-random-ok-adam200-64d-abi11-v1` samples 84
  of the 2250 initial-`ok` v2 rows without replacement and runs exact-start
  direct-data Adam200 on six GPUs. It uses 64 directions, `h=0.0025`, learning
  rate `0.01`, and no Flow calls. Conditional score-at-least-50 abundance must
  be multiplied by the measured `0.375` initial-valid rate. Launch state is
  recorded in the frozen runtime manifest rather than inferred from this file.

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
