# QH Protocol Registry

Last verified: 2026-09-01 (Asia/Shanghai).

## Status Vocabulary

- `current-default`: accepted baseline used when no experimental override is
  requested.
- `unregistered-experimental`: an automatic manifest label for any deviation
  from the current shared defaults; it has not been accepted for reuse.
- `registered-experimental`: a named, manifest-backed deviation on an
  exploration branch. It does not change the default.
- `historical-deprecated`: retained for provenance and prohibited for new runs.
- `invalidated`: the run occurred, but an implementation or protocol mismatch
  prevents the stated conclusion.

## Current Default

Protocol ID: `qh-flow-screen32-adam200-64d-abi11-v1`.

| Stage or setting | Current value |
| --- | --- |
| Screening | 32 random Flow candidates; select by current native score |
| Parameter space | Flow latent |
| Optimizer | Adam |
| Updates | 200 |
| Gradient estimate | 64 fresh random-orthogonal directions, centered |
| Score endpoints per update | 128 directional endpoints |
| Perturbation | `0.005` |
| Learning rate | `0.02` |
| Adam beta | `(0.7, 0.999)` |
| Flow decode | FP32 RK4-128 |
| Evaluator | ABI-11 axis-circulation, cubic-iota native library |
| Evaluator SHA-256 | `1c6c78b0dee662233215a56dbdc1e50b8ed29f2d0eee9ae8c4ff7d0403b895ed` |
| Axis handling | strict continuation, mode 2, within an optimization only |

The shared constants and classifier are in `flow_matching/optimization.py`.
Canonical commands are implemented by `scripts/screen_flow_starts.py` and
`scripts/optimize_flow_latent.py`. Compatibility wrappers and Slurm launchers
must inherit the shared values rather than defining another default.

## Run Gate

Every new run must write a machine-readable protocol block containing at least:

- protocol ID and status;
- code commit and dirty-state declaration;
- score-library ABI and SHA-256;
- Flow checkpoint SHA-256 and decoder discretization;
- input identity, `nfp`, base-coil count, and random seeds;
- parameter space and optimizer;
- candidate count, updates, directions, difference rule, perturbation,
  learning rate, beta, and axis mode;
- every deviation from the current default.

A reproduction uses the frozen original manifest. A new run using modified
settings is an experiment, even when it starts from a historical sample.

The current protocol classifier compares the actual score-library hash with
the accepted ABI-11 hash. A different ABI-11 rebuild is experimental until its
numerical contract is validated and promoted. Native ABI 10 is
`historical-deprecated`; current Python wrappers reject its ABI and current
resume checks reject its manifest requirements.

`--resume` is reserved for an interrupted run whose saved machine protocol and
repository commit/dirty state exactly match the requested continuation. Legacy
or unclassified manifests are historical inputs and cannot resume on current
`main`.

Exactly two directions is rejected by current Python entry points. If the user
explicitly requests a future two-direction study, create a new protocol on a
dedicated branch, write a new launcher and manifest, and deliberately review
the guard. Do not edit, copy, or re-enable a historical launcher.

## Deprecated Evaluator Registry

| Protocol or evaluator | Status | Interpretation |
| --- | --- | --- |
| `qh-flow-screen32-adam200-64d-v1` with ABI 10 | `historical-deprecated` | Former default; frozen trajectories reproduce the old all-current `G` objective |
| Native ABI 10, SHA-256 `565c3207...c729` | `historical-deprecated` | Counts every physical coil in vacuum `G`; invalid for unlinked-coil geometries and prohibited for new runs |
| ABI 9 and earlier | `historical-deprecated` | Earlier score definitions retained only with their frozen manifests |

ABI-11 defines vacuum `G` from the full selected-axis circulation,
`sign(flux)*abs(integral_axis(B dl))/(2*pi)`. This is equivalent to the signed
physical-current sum weighted by oriented coil-axis linking numbers.

## Deprecated 2D Registry

All entries below are `historical-deprecated`; an entry may also have an
invalidated conclusion.

| Period | Family | Interpretation |
| --- | --- | --- |
| 2026-08-06 | early score-fast optimizer matrix and beta1 sweeps | Old throughput/optimizer experiments; never a current default |
| 2026-08-10 | cubic-iota Adam-200 and direction-reuse comparison | Historical score-compression evidence under two fresh directions |
| 2026-08-24/25 | 32-case Flow/data coordinate control | Actually 2 directions and 100 steps; coordinate-causality claim invalidated |

Historical shell launchers containing 2D settings must carry the exact marker
`DEPRECATED HISTORICAL 2-DIRECTION PROTOCOL` and exit with status 64 before the
first old setting. Tests scan this invariant repository-wide.

Old files may contain words such as `default`, `production`, or `standard`.
Those words are timestamp-local historical metadata and never override this
registry. Reports requiring such files must label them as historical at the
point of reference.

## Promotion

Exploration remains `registered-experimental` until the user accepts it as the
default. Promotion requires one coordinated mainline change: shared defaults,
canonical and compatibility entry points, manifests, launchers, tests, current
documentation, `DECISIONS.md`, and affected correction entries. The former
default then moves to the historical registry; it is never left as a competing
implicit default.

The 2026-08-30 ABI-11 promotion completed this process. The unchanged 32/200/64D
optimizer settings now belong to `qh-flow-screen32-adam200-64d-abi11-v1`; the
former ABI-10 protocol ID remains historical and cannot resume as the new ID.

## Registered Experiments

Protocol `qh-axis-surface-contour-prior-score-abi11-v1` is `invalidated` by its
visual gate. Jobs `51581`, `51582`, and `51583` were cancelled after 953 partial
samples because the `nfp^-2` axis amplitudes, near-circular tube, and
spacing-scaled contour warp produced a nearly circular-axis, tokamak-like coil
family. The partial scores do not measure the intended stellarator prior and
must not be used for abundance conclusions. Frozen specification and outcome:
`evaluation/axis_surface_contour_prior_v1.json`.

The user accepted only the `balanced_stellarator` v2 morphology on 2026-09-01.
Protocol `qh-axis-surface-contour-balanced-score-abi11-v2` is a
`registered-experimental` completed score-only run. It sampled 6000
configurations over 26 `nc<=4` conditions with six count-bounded shards and no
optimization. All rows completed under ABI 11. The maximum total score was
`16.4393`; 56/6000 reached 10, 0/6000 reached 20, and 0/6000 reached 50. The
coil-component median was `72.4114`, while minimum intercoil distance had a
`27.27 mm` median and `4.04 mm` p10. Per-sample variation was restricted to the
accepted balanced family. Frozen specification:
`evaluation/axis_surface_contour_prior_balanced_abi11_v2.json`.

Canonical result report:
`reports/axis_surface_prior_balanced_v2_results_20260901.md`. The first analysis
job's representative selector failure and its verified repair are recorded as
`CORR-20260901-48`; the 6000 scores and aggregate plots were unaffected.

The `axis_dominant` and `surface_dominant` prototypes remain visual-review
artifacts and cannot enter this protocol. This experiment does not modify the
current default.

Protocol
`qh-axis-surface-contour-balanced-random-ok-adam200-64d-abi11-v1` is the
registered direct-data optimizability follow-up. It draws 84 samples uniformly
without replacement from the 2250 balanced-v2 rows whose initial ABI-11 status
is `ok`, assigns 14 to each of six GPUs, and runs 200 Adam updates with 64 fresh
orthogonal centered directions. Direct-data settings are `h=0.0025`, learning
rate `0.01`, and beta `(0.7,0.999)`. The exact analytic-prior start is preserved
with unclipped standardized coordinates and a per-start fixed current-L1
reference; Flow is never called. The measured 0.375 initial-valid rate is
multiplied by conditional Adam success to estimate unconditional abundance.
Frozen specification:
`evaluation/axis_surface_contour_prior_balanced_adam200_abi11_v1.json`.
Frozen run `axis_surface_prior_balanced_v2_adam200_20260901_11f703f` completed
83 of 84 selected starts; all 83 completed 200 updates with status `ok`, no
runtime failure occurred, and one `nc=1` sample was never started because a
worker reached its reserve gate. Historical best reached 50 in 31/83 complete
trajectories and reached 50 within 50 updates in 13/83. The maximum ABI-11
score was `69.6456`. Canonical evidence and interpretation are in
`reports/axis_surface_prior_balanced_v2_adam200_results_20260901.md`.

Representative full evaluation used fixed workflow commit `89206f4` and a GPU
library built from that exact commit. `axisv2_case_02986` (`nfp=5,nc=4`) accepted
the largest tested standard surface at `s=0.64`, volume `0.074017 m3`, direct QH
error `0.017012`; the nearest outer failure was `s=0.81`.
`axisv2_case_04428` (`nfp=5,nc=2`) accepted `s=0.81`, volume `0.092547 m3`,
direct QH error `0.017202`; the nearest outer failure was `s=1.00`. Both cases
showed nested Poincare sections. DESC retained nested boundaries and reduced
normalized force residuals substantially, while both optimizers stopped at the
50-iteration cap with `optimizer_success=false`. The report and its full
machine-readable artifacts preserve the acceptance checks and failure bounds.

Full-evaluation scheduling treats source-psi candidates, fixed surface
candidates, independent samples, and cross-sample downstream jobs as parallel
work. Each GPU candidate remains single-GPU, while candidate pools may use four
P107 and two Students slots concurrently. `SERIAL_CANDIDATES=1` requires a
nonempty `SERIAL_REASON`; launchers write the mode, reason, candidate count, and
pool assignment to a submission-policy JSON before the first Slurm submission.

Protocol
`qh-axis-surface-balanced-top2-continue-adam200-64d-abi11-v1` is a registered
direct-data continuation experiment. It starts from the saved Adam200 best
states of `axisv2_case_02986` (`nfp=5,nc=4`) and `axisv2_case_04428`
(`nfp=5,nc=2`), preserves the fixed current-L1 parameterization, and runs 200
new Adam updates with 64 fresh orthogonal centered directions, `h=0.0025`,
learning rate `0.01`, and beta `(0.7,0.999)`. Initial submission `52174` did
not enter optimization because result JSON files lacked the required exact-data
start wrapper; `CORR-20260901-62` records the contained failure. Replacement
array `52206` uses prepared and hashed continuation starts from commit `35a04b4`.
Both workers completed 200 updates. `axisv2_case_02986` improved from its
`69.6472` continuation re-evaluation to `70.0929` at step 63 and finished at
`69.8508`;
`axisv2_case_04428` improved from `69.1175` to `69.6224` at step 164 and
finished at `67.3170`. These are ABI-11 screening results; no new full physical
evaluation is implied.

Protocol `qh-axis-surface-contour-compact-flexible-score-abi11-v3` is a
registered score-only prior experiment. It keeps the independent analytic
axis/surface/contour construction, excludes `nc=5`, centers the winding-surface
minor radius at `0.20 m` with a `[0.18,0.22] m` generator range, and broadens
size, elongation, cross-section rotation, triangularity, helical ripple, and
contour harmonics relative to balanced-v2. It samples 3600 cases across the 26
supported `nc<=4` conditions without Adam. Arrays `52244` (four P107 GPUs) and
`52245` (two Students GPUs) started together after continuation array `52206`;
each shard has 600 cases and a hard one-hour limit. Analysis job `52246`
produces score, status, coil-engineering, condition, and representative-HTML
outputs. All six shards passed their initial runtime check with zero scoring
errors in the first logged batches. The experiment does not change the current
QH default.
