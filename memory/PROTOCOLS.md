# QH Protocol Registry

Last verified: 2026-08-28 (Asia/Shanghai).

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

Protocol ID: `qh-flow-screen32-adam200-64d-v1`.

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
| Evaluator | current ABI-10 cubic-iota native library |
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

`--resume` is reserved for an interrupted run whose saved machine protocol and
repository commit/dirty state exactly match the requested continuation. Legacy
or unclassified manifests are historical inputs and cannot resume on current
`main`.

Exactly two directions is rejected by current Python entry points. If the user
explicitly requests a future two-direction study, create a new protocol on a
dedicated branch, write a new launcher and manifest, and deliberately review
the guard. Do not edit, copy, or re-enable a historical launcher.

## Registered Experimental Protocols

### `qh-data-gaussian-global-survey-v1`

Status: `registered-experimental` on
`codex/qh-basin-atlas-random-survey` (2026-08-28).

This initialization survey draws independent standard Gaussians in the fixed
per-coordinate coil-data normalization. It applies the condition-specific
current L1 and dominant-sign projection, then runs one history-independent
current ABI-10 QH global evaluation. Each worker first runs one discarded
evaluation of the frozen reference case; formal calls explicitly pin the
standalone solver modes `psi_solver_mode=2` and `alpha_solver_mode=2`, while all
other fields use library defaults. The sampling prior is exactly balanced over
all 33 QUASR-supported `(nfp,n_base_coils)` groups. Six one-GPU workers use
disjoint machine-recorded random streams and atomic resumable output chunks.

The global survey records score-tail prevalence and retains every candidate at
score 20 or above. It does not classify a start as optimizable. After the
survey, a separate score-stratified follow-up must run 200 data-space Adam
updates with 64 fresh random-orthogonal centered directions, perturbation
`0.0025`, learning rate `0.01`, and beta `(0.7,0.999)`. The final basin-rate
estimate combines condition-balanced score-band prevalence with weighted Adam
success rates. Its cost report includes both global evaluations and conditional
Adam200 work.

This protocol neither uses Flow decoding nor changes
`qh-flow-screen32-adam200-64d-v1`. Exactly two directions remain prohibited.

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
