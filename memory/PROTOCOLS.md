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
| Formal evaluator | current compressed ABI-10 cubic-iota native library |
| Local-gradient oracle | query-batched ABI-10 library, separately identified |
| Axis handling | strict continuation, mode 2, within an optimization only |

The shared constants and classifier are in `flow_matching/optimization.py`.
Canonical commands are implemented by `scripts/screen_flow_starts.py` and
`scripts/optimize_flow_latent.py`. Compatibility wrappers and Slurm launchers
must inherit the shared values rather than defining another default.

The current compressed formal library SHA-256 `565c3207...c729` does not export
the query-batched field API. `CORR-20260829-15` therefore blocks new default
launches until the dual-library execution contract is validated and promoted:
the formal library scores every accepted center, while a separately hashed
batch library supplies only the approximate local-gradient oracle. This
corrects artifact plumbing without changing the 32/200/64D method definition.

## Run Gate

Every new run must write a machine-readable protocol block containing at least:

- protocol ID and status;
- code commit and dirty-state declaration;
- formal score-library ABI, SHA-256, and role;
- local-gradient library ABI, SHA-256, required batch symbols, and role;
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
current ABI-10 QH global evaluation. Each worker first evaluates the frozen
reference case and must match the current runtime reference `94.6254147736`
within `1e-5`; formal calls explicitly pin the standalone solver modes
`psi_solver_mode=2` and `alpha_solver_mode=2`, while all other fields use
library defaults. The sampling prior is exactly balanced over all 33
QUASR-supported `(nfp,n_base_coils)` groups. Six one-GPU workers use disjoint
machine-recorded random streams and atomic resumable output chunks.

The global survey records score-tail prevalence and retains every candidate at
score 20 or above. It does not classify a start as optimizable. After the
survey, a separate score-stratified follow-up must run 200 data-space Adam
updates with 64 fresh random-orthogonal centered directions, perturbation
`0.0025`, learning rate `0.01`, and beta `(0.7,0.999)`. The final basin-rate
estimate combines condition-balanced score-band prevalence with weighted Adam
success rates. Its cost report includes both global evaluations and conditional
Adam200 work.

The follow-up uses an explicit dual-library contract. Formal initial and trial
centers use compressed ABI-10 library SHA-256 `565c3207...c729`, preserving the
global-survey scale and `94.6254147736` reference. The local endpoint oracle
uses current-source query-batch library SHA-256 `b6697f54...48d6`; its separate
ABI-10 reference is `94.6368686721`. The optimizer manifest records both roles
and hashes. Only formal-library scores define improvements, thresholds, and
basin success; the batch library cannot contribute a reported candidate score.

The immutable first selection contains all 10 `score >= 20` samples and four
uniform `[0,20)` controls. The four controls happened to have `no_axis` status,
which makes them ineligible for this optimizer's required valid initial center.
On 2026-08-29 the user authorized a post-acceptance extension for an
approximately three-hour, six-GPU window: retain the original 14 records, take
a census of all 34 `score >= 10` samples, and add 38 uniformly sampled
`score < 10,status=ok` controls stratified by `n_base_coils` with quotas
`[4,8,7,7,12]`. The expanded manifest must label this as a post-acceptance
extension, record every inclusion probability, and preserve the original
selection file unchanged. It contains 76 records, of which 72 run Adam200.

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
