# QH Protocol Registry

Last verified: 2026-09-03 (Asia/Shanghai).

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
completed after all six shards finished. All 3600 rows were scored with zero
runtime errors. Native status counts were 569 `ok`, 1982 `no_axis`, 760
`no_surface`, 267 `drift_rejected`, and 22 `flux_rejected`, giving an initial
valid rate of `0.1580556`. Total-score median and maximum were `0.1247` and
`46.6289`; coil-component median and maximum were `72.0143` and `84.5682`.
Frozen specification:
`evaluation/axis_surface_contour_prior_compact_flexible_abi11_v3.json`.

Protocol
`qh-axis-surface-contour-compact-flexible-random-ok-adam200-64d-abi11-v1`
is the registered direct-data optimizability follow-up. It draws 120 samples
uniformly without replacement from the 569 v3 rows with native `status=ok`
using seed `20260904`. Selection counts by `nc=1..4` are `48,31,23,18`.
Runtime-weighted assignment puts 20 trajectories on each of six GPUs with
nearly equal predicted loads. Every trajectory requests 200 Adam updates with
64 fresh orthogonal centered directions, `h=0.0025`, learning rate `0.01`, and
beta `(0.7,0.999)` in exact-unclipped standardized data coordinates. Worker
arrays `52315/52316` completed concurrently after preparation `52313` and
three-step smoke `52314`; analysis `52317` produced the frozen summary. All
120 trajectories completed 200 updates with status `ok`, zero runtime failures,
and no incomplete or unstarted case. Under the frozen `(M,N)=(1,+nfp)` target,
best-so-far reached 50 by update 50 in 65/120 trajectories and by update 200 in
97/120. The corresponding all-prior threshold estimates were `8.56%` and
`12.78%`; the signed-helicity audit showed that all 120 trajectories remained
in the negative-iota mirror branch. These percentages are frozen score-threshold
statistics and no longer estimate same-handed QH abundance relative to the old
QUASR/Flow branch. The median and maximum Adam200 best scores were `65.3774`
and `68.8804`. Worker wall times were 5.31--5.47 hours. Frozen run commit is
`08a3c3f`. Canonical report:
`reports/axis_surface_prior_compact_flexible_v3_adam200_results_20260902.md`.
Frozen specification:
`evaluation/axis_surface_contour_prior_compact_flexible_adam200_abi11_v1.json`.
Neither compact-flexible experiment changes the current QH default.

Protocol
`qh-axis-surface-compact-v3-top2-negative-hand-continue-adam200-64d-abi11-v1`
is the registered signed-objective continuation audit. It starts from the
saved positive-target Adam200 best states of `axisv3_case_01341` and
`axisv3_case_02832`, immediately rescores each state with native ABI-11 target
`(M,N)=(1,-nfp)`, and then runs 200 direct-data Adam updates with 64 fresh
orthogonal centered directions, `h=0.0025`, learning rate `0.01`, and beta
`(0.7,0.999)`. The two samples run on separate P107 GPUs. Positive- and
negative-target scores are distinct objectives and must retain their signed
labels. Frozen specification:
`evaluation/axis_surface_prior_compact_flexible_negative_hand_continuation_abi11_v1.json`.
The two formal trajectories completed all 200 updates under array job `52566`.
`axisv3_case_01341` improved from negative-target initial score `75.8320` to
best `80.8523` at update 196; `axisv3_case_02832` improved from `79.9665` to
best `84.9308` at update 103. Population job `52580` found 569/569 valid source
samples and all 120/120 saved positive-target Adam200 trajectories in the
negative-iota branch. Mirror and dual-sign job `52581` confirmed that the
negative target raises the saved endpoints primarily through the volume-QS
component. Canonical report:
`reports/axis_surface_prior_handedness_audit_and_negative_optimization_20260902.md`.
This audit does not change the current QH default.

Protocol
`qh-axis-surface-contour-compact-flexible-axisflip-stream-adam200-64d-abi11-v1`
is the invalidated first construction-axis-only handedness experiment. For each seed,
it preserves the compact-flexible-v3 sampled radial axis coefficients, surface
parameters, contour-scalar parameters, current construction, and RNG sequence,
while multiplying every construction reference-axis vertical Fourier
coefficient by `-1`. The resulting surface and coils are regenerated from that
reflected reference axis. Native ABI-11 still scores the positive-hand target
`(M,N)=(1,+nfp)`. Six independent one-GPU streams cover the registered 26
conditions with `nc<=4`; every native `status=ok` sample immediately enters a
200-step direct-data Adam run with 64 fresh orthogonal centered directions,
`h=0.0025`, learning rate `0.01`, and beta `(0.7,0.999)`. Each worker stops
discovering candidates at four hours, finishes its active trajectory, then
exits. The five-hour Slurm limit is tail capacity rather than a five-hour
sampling budget. Frozen specification:
`evaluation/axis_surface_contour_prior_compact_flexible_axisflip_stream_adam200_abi11_v1.json`.
This experiment does not change the current QH default.
Formal smoke job `52677` passed one positive-iota valid start and three Adam
updates. P107 array `52678` and Students array `52679` launched all six workers
at commit `599dd31`, but the runner failed to retain screening `axis_R/axis_Z`.
Optimizer step 0 could select a different global-search branch, and the score
consistency check ran only after 200 updates. The remaining workers were
canceled after the systemic failure was confirmed. All v1 Adam200 outcomes are
quarantined; initial screening handedness remains preliminary evidence.
The remote run root is
`/home/scc/pb24511935/local_surface_evaluator_runs/axis_surface_prior_axisflip_stream_adam200_20260902_599dd31`.

Replacement protocol
`qh-axis-surface-contour-compact-flexible-axisflip-stream-adam200-64d-abi11-v2`
kept the v1 generator, seed streams, target, resources, and optimizer while
adding strict screened-axis continuation and a pre-update score gate. Smoke
`52730` passed case 0, but formal arrays `52731/52732` exposed a second error:
screening used generator `float64` tokens while the optimizer used their
`float32` standardized reconstruction. Arrays were canceled; analysis `52733`
counted 21 valid starts, 15 pre-update failures, 6 incomplete starts, and zero
completed trajectories. V2 is invalidated. Frozen specification:
`evaluation/axis_surface_contour_prior_compact_flexible_axisflip_stream_adam200_abi11_v2.json`.

Replacement protocol
`qh-axis-surface-contour-compact-flexible-axisflip-stream-adam200-64d-abi11-v3`
first maps each generated sample into the exact-unclipped standardized data
coordinates used by the optimizer, reconstructs physical tokens, and screens
those same tokens. It separately records source-to-start quantization, retains
the screened magnetic axis, and applies the `0.1` consistency gate before Adam.
Its case-25 smoke `52747` exposed a remaining score-configuration mismatch;
formal arrays `52748/52749` never started. V3 is invalidated. Frozen specification:
`evaluation/axis_surface_contour_prior_compact_flexible_axisflip_stream_adam200_abi11_v3.json`.

Replacement protocol
`qh-axis-surface-contour-compact-flexible-axisflip-stream-adam200-64d-abi11-v4`
retains v3's represented-token and axis-continuation fixes. Screening now calls
the optimizer's shared formal `score_config` with global axis search; optimizer
step 0 adds only the strict saved-axis hint. The pinned case-25 smoke must pass
before the same six-worker arrays can start. Frozen specification:
`evaluation/axis_surface_contour_prior_compact_flexible_axisflip_stream_adam200_abi11_v4.json`.
Frozen run commit `d8de349`; smoke `52758` passed case 25 with score delta
`0.0133273260`, identical axis coordinates, positive iota, and three Adam
updates. P107 array `52759` and Students array `52760` started all six formal
workers. Every worker passed its pre-update gate, with score deltas in
`[0.0007109622, 0.0133273260]`, identical saved/continued axes, and positive
initial iota. At the user's early-stop request, all six workers finished their
active Adam200 trajectory and stopped after about 2.3 hours of discovery.
Analysis `52761` accounted for 110 screened samples, 56 valid starts, 50
complete trajectories, zero failures, and six successor partials excluded from
formal statistics. All 56 valid starts had positive iota; 37/50 complete
trajectories reached 70, 9/50 reached 80, and the maximum was `81.8258373`.
The component analysis found median volume-QS score `19.4311 -> 59.1199` and
median coil-engineering score `69.3725 -> 69.1066`. Canonical report:
`reports/axis_surface_prior_axisflip_v4_results_20260902.md`. The run root is
`/home/scc/pb24511935/local_surface_evaluator_runs/axis_surface_prior_axisflip_stream_adam200_v4_20260902_d8de349`.

Protocol `qh-axisflip-prior-distilled-online-adam20-rwcfm-abi11-v1` is the
registered first online-RL experiment for the accepted axis-flip analytic
prior. It fixes the condition to `nfp=8,nc=3`. A 200,000-sample immutable
teacher corpus is generated from the compact-flexible v4 construction with
`axis_chirality=-1`, split deterministically 90/5/5, and used to train the
5,761,380-parameter `CoilFlowTransformer`. Distillation stops only after the
validation-loss plateau and generated-distribution stability gates both pass;
the 5000-epoch limit is a failing runaway guard, not an accepted training target.
Model initialization, per-rank training noise, data order, permutation,
validation, and monitor streams have recorded deterministic seed derivations.
An independent ABI-11 teacher-versus-Flow audit gates entry into online rounds.

Each online round samples 64 starts through four concurrent one-GPU workers.
Every ABI-11 `status=ok` start runs 20 exact-data Adam updates using 64 fresh
orthogonal centered directions, `h=0.0025`, learning rate `0.01`, beta
`(0.7,0.999)`, and the strict saved-axis/step-0 score gate. Invalid starts retain
their represented coordinates and receive reward weight `0.05`; valid targets
use their best point from steps 0 through 20 and rank weights in `[1,2]`. The
Flow update mixes 85% unweighted current-policy anchoring, 10% weighted current
targets plus a 512-sample FIFO replay, and 5% original teacher anchoring. It
continues raw training weights and online AdamW state across policy rounds after
a fresh round-0 optimizer; EMA weights define the sampled policy. It records
validity, score and score-component medians, Adam gains, threshold
counts, component correlation, diversity, policy movement, and runtime each
round. This experiment is not a default change. Frozen specification:
`evaluation/axisflip_prior_distilled_online_adam20_rwcfm_abi11_v1.json`.
The corrected formal run at commit `34a6148` converged its q0 distillation at
epoch 201/global step 35376 and passed the independent 128-versus-128 teacher/
Flow ABI-11 audit. The accepted reporting snapshot contains complete online
rounds 0 through 21: 1408 generated starts, 1394 valid. Initial-score median
rose from `63.6751` to `71.9023`, initial score-at-least-70 count rose from
`4/64` to `53/64`. Initial volume-QS and coil medians changed by `+12.7776`
and `+2.7805`. Near-duplicate rate remained zero; effective rank changed by
`-2.0%` while descriptor variance fell by `16.6%`, a monitored contraction
without evidence of catastrophic collapse
in this snapshot. P107 job `52977` remained running after the snapshot.
Canonical report:
`reports/axisflip_prior_online_rl_results_20260903.md`.

Protocol `qh-axisflip-rl-round12-flow-screen32-adam200-64d-abi11-v1` is a
registered frozen-policy latent experiment at fixed `nfp=8,nc=3`. It pins the
online-RL run's atomically complete `round_012.pt` EMA policy after source round
11, SHA-256 `1ffbd6329a23feb060aa2b279e96ef89d71ff5fd45e01fdaa0de768e08537b72`.
Each of eight independent trajectories screens 32 standard-normal Flow latents
and runs the selected valid start through 200 latent Adam updates with 64 fresh
orthogonal centered directions, `h=0.005`, learning rate `0.02`, beta
`(0.7,0.999)`, and pipelined FP32 RK4-128 decoding. The screening axis is
reused as a strict step-0 hint with a `0.1` score-consistency gate. Two
single-GPU Students workers run concurrently and execute four trajectories per
GPU. The optimizer mechanics match the current QH default; the frozen RL Flow
identity makes this a separate experiment and does not change that default.
Frozen specification:
`evaluation/axisflip_rl_round12_latent_adam200_abi11_v1.json`.
Initial Students array `53046` is a preserved failed attempt: both tasks
completed screen32, then a QUASR-only checkpoint-step guard rejected the RL
checkpoint before optimizer step 0. Corrected commit `8f5d57a` retains step
`30000` as the default expectation while allowing this registered runner to
pin step `36616` and the exact checkpoint hash. Replacement array `53049`
completed two concurrent workers and all eight Adam200 trajectories. All
204800 gradient endpoints were `ok`; best-score median was `78.2001`, maximum
was `79.6088`, and 4/8 trajectories reached 78. The screen-to-step-0 score
delta remained below the `0.1` gate for every case. Its run root is
`/home/scc/pb24511935/local_surface_evaluator_runs/axisflip_rl_round12_latent_adam200_20260903_v2_8f5d57a`;
canonical report:
`reports/axisflip_rl_round12_latent_adam200_results_20260903.md`.

Protocol `qh-axisflip-v4-representative-adam2000-64d-abi11-v1` is a separate
long-horizon continuation of the two v4 representatives already subjected to
full physical evaluation: case 18 (`nfp=8,nc=3`) and case 23
(`nfp=6,nc=4`). The two independent Students jobs run concurrently. Each begins
from the frozen Adam200 best, retains positive-hand ABI-11 and the exact
axis/score gate, and requests 2,000 new exact-data Adam updates with 64 fresh
orthogonal directions, `h=0.0025`, learning rate `0.01`, and beta
`(0.7,0.999)`. These screening trajectories do not repeat or replace full
physical evaluation. Frozen specification:
`evaluation/axisflip_v4_representative_adam2000_abi11_v1.json`.
Students jobs `52971/52972` completed concurrently at commit `6350b73`, each
with 2,000 accepted updates and status `ok`. Case 18 improved from long-run
step-0 score `81.8336` to `87.6821` at step 1583; case 23 improved from
`80.3063` to `89.5535` at step 1985. Their best-point volume-QS changes were
`+18.9698/+21.8622`, accompanied by coil-component changes
`-6.2991/-5.1897`. These endpoints have native screening evidence only; the
earlier full evaluations remain attached to the Adam200 representatives.
Canonical results are appended to
`reports/axis_surface_prior_axisflip_v4_results_20260902.md`.

The user-requested full evaluation of the higher case-23 endpoint then used its
step-1985 best. Students-only candidate and downstream jobs accepted the largest
tested standard surface at `s=0.81`, volume `0.1021639 m3`, with direct
`QH_(1,+1)=0.0002348713`; `s=1.00` failed the fixed volume-sampling budget before
standard LS/Newton. Poincare ordering and DESC initial/final nesting passed.
DESC reduced mean normalized force `0.78224 -> 0.02379` and stopped at the
50-iteration cap. This physical follow-up does not change the optimizer or QH
default.

Representative full evaluation used fixed workflow commit `e4590ae` and the
full-evaluation GPU library built at `89206f4`, SHA-256
`23158593e57cd82300aa8d2efb2ee3023662d7f9d1765d1cfa22c84f26434af0`.
`axisv3_case_01341` (`nfp=5,nc=3`) accepted the largest tested standard
surface at `s=0.24`, volume `0.026633 m3`, direct `QH_(1,+1)` error `0.009187`; its
nearest outer failure was `s=0.36`. `axisv3_case_02832` (`nfp=7,nc=4`)
accepted `s=0.36`, volume `0.046051 m3`, direct `QH_(1,+1)` error `0.009728`; its
nearest outer failure was `s=0.49`. Both DESC runs kept nested boundaries and
reduced normalized force residuals substantially but reached the 50-iteration
cap. The `axisv3_case_02832` vacuum Poincare map shows a multi-lobed/island-
chain pattern; its accepted outer boundary and DESC equilibrium do not certify
a clean globally nested vacuum interior. Canonical evidence is appended to the
compact-flexible-v3 Adam200 report.

Protocol
`qh-axisflip-v4-case23-axis-centered-coil-shrink-adam200-64d-abi11-v1`
is the superseded pointwise diagnostic on the fully evaluated case-23
Adam2000 step-1985 best. It first audits coil-scale drift and the exact
curvature/distance engineering-score contributions in all 50 complete v4
original-space Adam200 trajectories. It then maps each case-23 coil point as
`a_nearest + scale * (point - a_nearest)` around the verified magnetic axis,
refits the original order-16 Fourier representation, preserves currents, and
scores eleven scales from `1.0` through `0.2` with positive-hand ABI-11. The
original `s=0.81` surface is a fixed geometric clearance reference for this
scan. Two valid compact candidates that remain outside this reference surface
are repaired concurrently on the two Students GPUs using 200 exact-data Adam
updates, 64 fresh orthogonal directions, `h=0.0025`, learning rate `0.01`, and
beta `(0.7,0.999)`. Its Fourier refit imported scale-dependent shape distortion:
the maximum residual was `0.0814 m` at `scale=0.8` and `0.2646 m` at
`scale=0.35`. Scan jobs `53203/53207` and repair jobs `53210/53211` remain
valid evidence about that pointwise map only. Frozen specification:
`evaluation/axisflip_case23_coil_shrink_adam200_abi11_v1.json`. The method is
preserved as diagnostic history and does not change the current QH default.

Protocol
`qh-axisflip-v4-case23-coil-anchor-shrink-adam200-64d-abi11-v1` is the
shape-preserving continuation of the case-23 geometry intervention. For each
base coil it selects the verified magnetic-axis point nearest the coil
centroid, then applies one uniform similarity transform about that fixed point.
This retains the order-16 Fourier shape to numerical precision and leaves all
currents unchanged. The scan emphasizes `scale=0.4`, which the geometry-only
preflight predicts will give an effective coil radius near `0.193 m` while
remaining outside the fixed source `s=0.81` surface. Up to two valid outside
candidates enter the same exact-data Adam200/64-direction ABI-11 repair used by
the pointwise diagnostic. Frozen specification:
`evaluation/axisflip_case23_coil_anchor_shrink_adam200_abi11_v1.json`. This
experiment does not change the current QH default. Scan job `53217` found that
`scale=0.8` retained native status `ok`, effective radius `0.38634 m`, and coil
score `69.0738`, while volume-QS fell to `39.7922`; `scale=0.6` was the smallest
valid scan point and `scale=0.5` was flux-rejected. Concurrent Students repair
jobs `53223/53224` completed. The `scale=0.8` best reached `88.9862` at step
113 while expanding by `38.84 mm` and losing `5.79 mm` minimum intercoil
spacing; curvature contribution rose `1.7581` and distance contribution fell
`0.5172`. The `scale=0.6` run accepted only three updates and reached `7.7187`.
Canonical report: `reports/axisflip_case23_coil_shrink_results_20260903.md`.
