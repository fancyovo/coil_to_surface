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

A replacement prior is not registered for scoring. It must first show the
construction reference axis, shaped inner reference surface, winding surface,
and coils together and receive explicit user visual acceptance. No GPU batch
may launch before that gate. This status does not modify the current default.
