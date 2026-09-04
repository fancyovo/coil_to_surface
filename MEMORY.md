# Current Project Memory

> Current truth, verified 2026-09-04 (Asia/Shanghai). This is a compact routing
> and safety file, not a work log. Older material is indexed under `memory/`.

## Baseline

- The authoritative private development baseline is `main` after the
  2026-08-28 consolidation. Verify the actual checkout and HEAD with Git at the
  start of repository work; do not infer them from this file.
- The sanitized open-source sibling `../opensource_staging` was verified clean
  on `main` at `89d30e92b7b05687637f2589f649b8def8d3c8b7`. The private baseline
  carries the current public screening and optimization interfaces while
  retaining private research evidence.
- Active exploration branch: `codex/r013-score-gradient-rl`, based on the
  frozen R04 source commit `95ed6cf`; the active R04 job remains in its own
  worktree. Protocol
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

- Start new exploration from consolidated `main` on a `codex/` branch and
  switch the primary checkout before edits; a secondary worktree is reserved
  for preserved or concurrent work. Keep canonical reports on the owning branch
  and mirror finalized reports/assets into Git-excluded `_shared_reports/`.
- User acceptance promotes a method to `main` with coordinated code, manifests,
  launchers, tests, docs, decisions, and corrections. Preserve frozen manifests
  and never reconstruct settings from prose or CLI defaults.
- Run independent samples and candidates concurrently up to the verified GPU or
  CPU allowance. Serial execution requires a real dependency, resource limit,
  or explicit user request and must record its reason. Full-evaluation
  launchers enforce this through submission-policy JSON files.
- Record qualifying user-discovered or materially costly errors in
  `memory/CORRECTIONS.md` with impact, retained conclusions, evidence,
  containment, and status; routine self-caught fixes stay out of the ledger.
- External documents and figures must pass `memory/WRITING.md`. Use direct,
  affirmative explanations; remove defensive contrast, ambiguous references,
  version leakage, and irrelevant claims after generation.

## Remote Compute Access

- Read root `REMOTE_CODEX_INSTRUCTIONS.md` in full before any remote operation in a fresh or compacted run, and repeat after connection or network changes.
- Reuse only the authenticated WSL `Ubuntu` master for `ustc107` and run every
  documented preflight in order. If the master is absent, stop and give the
  user the documented command; never authenticate, unlock keys, request
  credentials, guess paths, or bypass the master. Heavy work runs via Slurm.

## Active Analytic-Prior Exploration

- The failed Online RWCFM evidence is specific to an independent Gaussian in
  QUASR featurewise-standardized Fourier coordinates. It does not establish an
  RL limitation for structured physical priors. The canonical failure report
  is on `codex/qh-online-validity-rwcfm-zero@9079fa3` and mirrored in
  `_shared_reports/qh_online_rwcfm_failed_20260901.md`.
- Analytic-prior v1 is visually invalid because it resembles circular-axis
  tokamak coils. The accepted balanced-v2 morphology produced 6000 ABI-11
  scores with maximum `16.4393`; its conditional Adam200 follow-up and two full
  evaluations are historical evidence in
  `reports/axis_surface_prior_balanced_v2_adam200_results_20260901.md`.
- Compact-flexible v3 centers winding radius at `0.20 m` and excludes `nc=5`.
  Its 3600-score batch had 569 valid starts (`15.8056%`); 97/120 selected valid
  starts reached 50 under Adam200, for estimated all-prior abundance `12.78%`.
  The maximum was `68.8804`. Full details and two physical evaluations are in
  `reports/axis_surface_prior_compact_flexible_v3_adam200_results_20260902.md`.
- The signed-helicity audit found all 569 valid compact-v3 starts and all 120
  positive-target Adam200 trajectories on the negative-iota branch. The leading
  bias is the fixed-sign construction-axis harmonic; surface and contour signs
  remain unisolated. Negative-target Adam200 reached `80.8523` and `84.9308`.
  Frozen positive-target scores and non-QS physical results remain valid.
- Axis-flip stream Adam200 v1-v3 are invalidated for screened-axis,
  numeric-representation, and score-configuration inconsistencies. Their Adam
  outcomes are quarantined; v1's 24 positive-iota starts remain preliminary
  handedness evidence. V4 at `d8de349` fixes all three; smoke `52758` and six
  formal workers passed their axis/score gates. Final: 110 screened, 56 valid
  positive-iota starts, 50 complete, zero failed, six excluded partials; 37
  reached 70, nine reached 80, maximum `81.8258`. Median volume-QS rose
  `19.4311 -> 59.1199`; coil stayed near 69. Full evaluation accepted
  `s=0.49` surfaces for cases 18/23 with direct QH errors `0.001626/0.002174`; both DESC boundaries stayed nested at the 50-step cap. See
  `reports/axis_surface_prior_axisflip_v4_results_20260902.md`; no default change.
- Fixed-condition teacher protocol `nfp=8,nc=3` synthesized 200,000 positive-hand
  compact-flexible-v4 samples at commit `b03af40` using 16 P107 plus 24 Students
  CPU cores. Jobs `52958/52959/52960` completed and the immutable dataset passed
  manifest verification.
- Corrected experimental commit `34a6148` converged q0 distillation at epoch
  201/step 35376 and passed an independent teacher-versus-Flow ABI-11 audit.
  The user stopped P107 job `52977` after atomically complete round 27; partial
  round 28 is excluded. Across 1792 samples, 1778 were valid. Initial-score
  median improved `63.6751 -> 72.0170` and peaked at `72.7139` in round 26;
  score-at-least-70 changed `4/64 -> 44/64` and peaked at `57/64`. All 28 near-
  duplicate rates were zero; effective rank rose `16.5%` while descriptor
  variance contracted `18.5%`. Rounds 21--27 form a score plateau. See
  `reports/axisflip_prior_online_rl_results_20260903.md`.
- Frozen-policy latent experiment
  `qh-axisflip-rl-round12-flow-screen32-adam200-64d-abi11-v1` uses the
  atomically complete round-12 EMA checkpoint (source round 11, SHA-256
  `1ffbd632...537b72`) at fixed `nfp=8,nc=3`. Only the Flow checkpoint differs
  from the current screen32/Adam200/64D/RK4-128 latent recipe; its source RL
  state was untouched during that experiment. Initial array `53046` stopped before Adam because of a
  QUASR-only checkpoint-step guard. Corrected Students array `53049` completed
  all eight Adam200 trajectories at commit `8f5d57a`; best-score median was
  `78.2001`, maximum was `79.6088`, and 204800/204800 gradient endpoints were
  `ok`. See `reports/axisflip_rl_round12_latent_adam200_results_20260903.md`.
- Independent positive-hand Adam2000 continuations of full-evaluated v4 cases
  18 and 23 completed 2,000 updates each as Students jobs `52971/52972`, reaching
  native screening bests `87.6821/89.5535`. Both traded about 5--6 coil points
  for about 19--22 volume-QS points. Full evaluation of case 23's step-1985 best
  accepted `s=0.81`, volume `0.102164 m3`, and direct `QH_(1,+1)=2.3487e-4`;
  `s=1.00` was the nearest workflow rejection. DESC retained nested boundaries
  but stopped at the 50-step cap with final mean normalized force `0.02379`.
  See the supplement in
  `reports/axis_surface_prior_axisflip_v4_results_20260902.md`.
- Case-23 coil-shrink work uses one fixed magnetic-axis anchor per base coil and
  an exact similarity transform. In 50 prior original-space Adam200 runs, 48
  enlarged their effective coil radius. The shape-preserving `scale=0.8`
  candidate repaired to `88.9862` while regrowing `38.84 mm` and losing
  `5.79 mm` minimum spacing; `scale=0.6` remained trapped near score `7.72`.
  The earlier pointwise-nearest-axis map is superseded diagnostic history. See
  `reports/axisflip_case23_coil_shrink_results_20260903.md`.
- Registered fixed-condition experiment
  `qh-axis-surface-contour-compact-flexible-axisflip-r012-curvature-r04-adam200-64d-abi11-v1`
  centers the prior winding radius at `0.12 m`, changes only the experimental
  curvature-p95 scale to `25 m^-1` (`0.04 m` radius), retains the `35 m^-1`
  maximum-curvature guard, and completed a fixed 384-sample audit plus 12
  six-GPU Adam200 trajectories. Legality was `95/384`; 11/12 trajectories
  reached 70, with maximum `80.2724`. Effective radius still grew in 12/12
  trajectories while median paired coil score changed `-0.4595`; see
  `reports/axisflip_r012_curvature_r04_adam200_results_20260903.md`.
- Active R04 protocol `qh-axisflip-r012-distilled-online-adam20-trajectory-rwcfm-r04-abi11-v1`
  runs in its preserved P107 worktree (`53372`); its 512-rollout FIFO and
  four-day, no-round-cap policy remain unchanged.
- New student protocol `qh-axisflip-r012-score-gradient-transport-rl-r04-abi11-v1`
  runs as job `53957` in a separate worktree with two GPUs and no round cap.
  After three rounds, valid rates were `14.06%, 17.19%, 15.63%, 23.44%`; 64D
  gradients passed for 100%, 100%, 100%, and 86.7% of valid centers. Fixed
  `beta=0.0060219592`, calibrated to a 10% transport-loss fraction.
- Job and round state is volatile and must be checked directly before use.

## Memory Map
- `memory/README.md`: memory architecture; `memory/PROTOCOLS.md`: protocols and gates.
- `memory/CORRECTIONS.md`: error ledger; `memory/DECISIONS.md`: active decisions.
- `memory/WRITING.md`: deliverable rules; `memory/HISTORY.md`: evidence chronology.
- `memory/archive/` and `MEMORY_archive_20260808.md`: historical archives only.

Update only for current truth/routing; route chronology to `memory/HISTORY.md`, errors to `memory/CORRECTIONS.md`, and detail to reports. Never store credentials.
