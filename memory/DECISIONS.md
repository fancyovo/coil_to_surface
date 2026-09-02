# Active Decisions

Last reviewed: 2026-09-02 (Asia/Shanghai).

## DEC-20260828-01 - One current optimization default

Status: active.

Protocol `qh-flow-screen32-adam200-64d-abi11-v1` is the sole implicit QH default.
Two-direction work is historical and hard-blocked. Nondefault work must be a
named experiment with manifest differences. See `PROTOCOLS.md`.

## DEC-20260825-01 - Flow latent is the practical main method

Status: active, with bounded interpretation.

Flow latent optimization is the current practical high-score method. Direct
standardized-data optimization remains a useful faster baseline. The 309-pair
result supports the tested complete recipes; differing learning rates and
perturbations leave pure coordinate causality unresolved.

## DEC-20260810-01 - Current native evaluator settings

Status: superseded by `DEC-20260830-01`.

ABI 10 was the production evaluator before the unlinked-coil current audit.
Its settings and scores are frozen historical evidence.

## DEC-20260830-01 - ABI 11 is the production evaluator

Status: active; explicitly accepted by the user on 2026-08-30.

Use ABI 11 with selected-axis Ampere circulation for vacuum `G`, psi grid 48,
cubic iota, strict axis-hint mode 2 in optimizer continuation, and continuous
surface confidence. ABI 10 is historical and cannot launch or resume through
current entry points. Standalone and corpus scoring remain history-independent.
Complete physical acceptance remains separate from native score.

## DEC-20260828-02 - Exploration and default promotion

Status: active.

Begin exploration from consolidated `main` and use a `codex/` branch when
isolation is useful. A method becomes the default only after explicit user
acceptance and mainline integration of implementation, protocol metadata,
launchers, tests, docs, decisions, and corrections. Earlier methods become
clearly labeled history; they do not remain alternate implicit defaults.

## DEC-20260828-03 - Layered memory with corrections

Status: active.

Root memory contains only current high-risk truth and routing. Protocols,
corrections, decisions, writing rules, chronology, and immutable archives have
separate owners. Root memory is read once per fresh context, not once per prompt.

## DEC-20260828-04 - External writing quality gate

Status: active.

External documents and multimodal deliverables use direct affirmative prose,
explicit versioned references, and only claims relevant to the core argument.
They receive a second semantic audit after generation. Detailed rules are in
`WRITING.md`.

## DEC-20260831-01 - Exploration checkout and shared report delivery

Status: active; explicitly requested by the user on 2026-08-31.

A request to explore establishes a dedicated `codex/` branch and switches the
primary user-visible repository checkout to it before substantive work. A
secondary worktree is reserved for cases where preserved changes or concurrent
work make the primary switch unsafe; the exception must be reported explicitly.

The canonical report and its referenced assets remain tracked on the branch
that owns the experiment. At delivery they are mirrored into `_shared_reports/`
in the primary checkout. That directory is excluded through the repository's
local Git metadata, survives branch switches, and provides a stable local file
surface. The tracked branch copy and commit remain authoritative provenance.

## DEC-20260901-02 - Analytic axis/surface/contour prior study

Status: v1 invalidated; v2/v3 retained as geometry and engineering baselines;
signed QH interpretation superseded.

The first replacement for the failed QUASR-marginal Gaussian prior used a
hierarchical analytic construction, but its axis, tube, and contours were all
too weakly deformed. The resulting geometry looked like a circular-axis
tokamak coil set and failed the user's visual requirement.

The replacement retains a clearly noncircular reference axis, a winding
surface that bends with that axis, and cross-section shaping that varies over a
field period. Prototype views include the axis, an inner plasma-like reference
surface, the winding surface, and coils. The reference surface is a geometric
prior and must not be called a computed magnetic surface.

The user accepted only the `balanced_stellarator` morphology on 2026-09-01.
Protocol `qh-axis-surface-contour-balanced-score-abi11-v2` then measured a
6000-sample, six-GPU batch over 26 `nc<=4` conditions. The
`axis_dominant` and `surface_dominant` prototypes remain unaccepted and cannot
enter this run. This acceptance authorizes the registered experiment and does
not promote the prior into the current default.

The completed batch measured a maximum ABI-11 score of `16.4393`, with no
sample at 20 or above. The prior reliably produces low-high-mode coil geometry,
yet it does not provide evidence of QH-basin abundance. Intercoil clearance
also degrades sharply with base-coil count: median clearance falls from
`68.95 mm` at `nc=1` to `9.88 mm` at `nc=4`. Future revisions should include an
explicit clearance construction and condition-dependent physical shaping
before another large score or RL experiment. The v2 result does not change the
current Flow/Adam default.

The authorized Adam200 follow-up estimates optimizability conditional on an
initial ABI-11 `ok` status. It samples 84 of those 2250 rows uniformly without
replacement, then assigns equal counts and balanced `nc` cost to six GPUs. The
known 0.375 initial-valid probability is retained separately and multiplied by
conditional score-at-least-50 success for the full-prior estimate. Exact
unclipped standardized-data coordinates preserve each analytic-prior start;
the frozen data-space settings are 200 updates, 64 directions, `h=0.0025`, and
learning rate `0.01`. This follow-up measures the v2 prior and does not alter
the current default.

The 2026-09-02 signed-helicity audit established a one-sided final-coil
population: all 569 valid compact-flexible-v3 source samples and all 120 frozen
Adam200 trajectories occupied the negative-iota branch, opposite the project's
positive-hand QH target. A later causal review withdrew the claim that the
surface scalar field alone caused that sign. The leading code-level bias is the
fixed-positive dominant radial/vertical harmonic of the construction axis;
fixed-sign winding-surface and contour terms are coupled candidates. The frozen
scores remain valid for `(M,N)=(1,+nfp)`; their threshold rates no longer measure
same-handed QH abundance. Explicit `(1,-nfp)` Adam200 raised the two representative
endpoints to `80.8523` and `84.9308`.

The user authorized the construction-axis factor as the first isolated
intervention: reflect only the sampled reference-axis `Z(phi)` for a given seed,
then regenerate the downstream surface and coils without changing the sampled
surface/scalar parameters or positive-hand scoring target. The experiment uses
six concurrent streams, sends every initial ABI-11 `status=ok` candidate into
Adam200, and stops new discovery after four hours while completing the active
trajectory. Its v1 execution is invalidated because screening artifacts omitted
the selected magnetic axis and optimizer step 0 could repeat global search on a
different branch. V2 preserves the screened axis, validates its strict-hint
step-0 score before any update, but its formal run revealed that screening still
preceded conversion into the optimizer's `float32` representation. V2 is also
invalidated. V3 constructs the actual optimizer-representable tokens before
both stages, then exposed that screening and optimizer formal centers still
used different surface-selection configurations. V3 is invalidated. V4 shares
the optimizer's formal score configuration across both stages while retaining
the same physical chirality intervention, represented start, and pre-update
axis/score gate. This is an experiment, not a chirality-default promotion.

The v4 stream was drained at the user's early-stop request after every active
trajectory completed. Final analysis contains 110 screened samples, 56 valid
positive-iota starts, 50 complete Adam200 trajectories, zero failures, and six
explicitly excluded successor partials. Of the complete trajectories, 37
reached 70 and 9 reached 80; the maximum was `81.8258373`. Median volume-QS
score increased from `19.4311` to `59.1199`, while median coil-engineering score
changed from `69.3725` to `69.1066`. This supports the axis sign as the dominant
source of the previous one-sided handedness and shows access to the positive-QH
high-score basin. It does not isolate the axis sign's paired causal effect or
promote a new default. Canonical evidence is in
`../reports/axis_surface_prior_axisflip_v4_results_20260902.md`.

Representative full evaluation selected the v4 maximum-score case 18 and the
higher-engineering 80-point case 23. Both accepted the largest tested standard
surface at `s=0.49`; direct `QH_(1,+1)` errors were `0.0016255` and `0.0021736`.
Both DESC boundaries remained nested and their mean normalized force fell by
about 11139-fold and 1280-fold, while both solvers reached the 50-iteration
limit without a successful termination flag. This is two-case feasibility
evidence and does not estimate full-population physical acceptance.

Before promotion, independently test winding-surface and contour-scalar
chirality as remaining factors and retain exact final-coil reflection as the
sign-flipping control. Promotion still requires mirror-pair regression tests
and handedness-stratified population statistics. A future default may sample
both signs uniformly or canonicalize to the positive-hand target only after
those results are reviewed.
