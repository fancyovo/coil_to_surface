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

## DEC-20260903-01 - Fixed-condition analytic-prior online Flow experiment

Status: completed experimental decision; no default impact.

The first structured-prior RL implementation fixes `nfp=8,nc=3` and distills
the accepted positive-hand axis-flip analytic generator before applying reward
updates. The teacher corpus is generated once and persisted so repeated Flow
updates do not pay analytic-construction cost. All 40 authorized CPU cores are
used as two independent deterministic shards: 16 on P107 and 24 on Students.

The Flow architecture matches the earlier small online model. Its set-valued
Transformer has no positional encoding; random whole-coil permutations
symmetrize the generator's arbitrary contour order. The q0 current target is
equal across coils, while the current channel remains trainable so Adam20 can
teach relative current allocation under the fixed total-current L1 projection.

Each policy round uses 64 samples split evenly over four P107 GPUs. Legal starts
receive Adam20 targets and illegal starts remain unchanged with low reward
weight. A bounded update combines current-policy anchoring, reward-weighted
targets with replay, and a small original-prior anchor. The first version uses
only the minimum anti-collapse measures: a 512-sample FIFO replay, 5% q0 anchor,
bounded rank weights, and round-by-round invariant diversity measurements.
Raw training weights and online AdamW moments persist across rounds, while EMA
weights define the sampled policy. The repeated updates therefore remain one
continuous policy optimization rather than independent per-round fits.
Distillation convergence and a teacher-versus-Flow ABI-11 audit are mandatory
before round 0. Promotion requires observed enrichment without material
diversity loss and explicit user acceptance.

The user stopped job `52977` after round 27 completed. Across rounds 0--27,
initial-score median improved from `63.6751` to `72.0170`, peaked at `72.7139`,
and entered a plateau after round 21. All 28 near-duplicate rates were zero;
effective rank finished `16.5%` above round 0 while total descriptor variance
finished `18.5%` lower. This is successful enrichment with moderate monitored
contraction, not a promotion decision. Partial round 28 is excluded.

## DEC-20260903-02 - Freeze an RL policy for standard latent Adam200

Status: active experimental decision; no default impact.

The first test of optimization inside the RL-trained Flow latent space freezes
the latest fully written checkpoint available at implementation time:
`round_012.pt`, whose EMA policy follows completed online round 11. The frozen
SHA-256 is `1ffbd6329a23feb060aa2b279e96ef89d71ff5fd45e01fdaa0de768e08537b72`;
later RL rounds cannot silently change this experiment.

The comparison changes only the Flow checkpoint and fixes the condition to its
trained support, `nfp=8,nc=3`. Screening, latent parameter space, 200 Adam
updates, 64 fresh orthogonal centered directions, step size, Adam coefficients,
ABI-11 target, and pipelined FP32 RK4-128 decoding match the current stable
latent recipe. Two Students GPUs run independent workers in parallel. Four
trajectories share each worker's sole GPU sequentially, which is the explicit
resource reason for that ordering.

## DEC-20260903-03 - Smaller analytic-prior radius with relaxed p95 curvature scale

Status: completed registered experiment; no default impact.

The fixed `nfp=8,nc=3` follow-up centers the compact-flexible axis-flipped
winding radius at `0.12 m`. It tests whether smaller initial coils remain legal
and optimizable when the p95 curvature reward stops favoring radius growth at
the former `0.10 m` characteristic radius. The experimental score therefore
uses `25 m^-1`, or `0.04 m`, for curvature p95 while retaining the existing
`35 m^-1` maximum-curvature guard. All remaining ABI-11 score terms and the
direct-data Adam200/64-direction recipe stay fixed.

The legality estimate uses 384 precommitted samples rather than stopping its
denominator after a fixed number of legal discoveries. Six workers each
complete two Adam200 trajectories and continue score-only sampling until their
64-sample audit is complete. The build flag is experiment-specific; the
default ABI-11 library and current QH protocol remain unchanged.

The experiment found 95/384 legal starts and 11/12 Adam200 trajectories above
70, with maximum `80.2724`. The volume-QS component supplied the improvement:
its paired gain median was `+48.0524`, while the coil component median changed
`-0.4595`. All 12 effective radii nevertheless increased, by a paired median
of `0.0751 m`. Relaxing curvature p95 removes a consistent coil-score reward
for growth but does not keep optimized geometry near the 0.12 m prior scale.
Any follow-up that requires compact final coils must target size explicitly.

## DEC-20260904-01 - R012 trajectory-replay online Flow experiment

Status: active registered experiment; no default impact.

The second structured-prior RL run starts from a fresh distillation of the
fixed `nfp=8,nc=3` R012 generator and uses the experimental R04 curvature score
throughout its complete scoring and Adam20 path. It does not warm-start the
earlier v4 prior policy.

All formal Adam20 centers enter a 512-rollout FIFO. Rollout-length division
prevents the 21 highly correlated centers of a legal optimization from gaining
21 times the objective mass solely through storage, while a score-temperature
weight with `tau=7.5` and epsilon `0.01` retains low-score starts with small
positive mass. The trajectory batch grows with the mean stored rollout length;
microbatch accumulation keeps the number of optimizer updates fixed at 250.
The 85/10/5 policy/reward/original-prior mixture remains unchanged so this run
tests the new prior, scorer, and trajectory target construction without also
changing the outer RL step strength.

Four P107 GPUs run until the four-day wall reserve or an explicit
`STOP_AFTER_ROUND` sentinel, with no round-count cap. In parallel, the two
Students GPUs continue R012 Adam200 best cases 36 and 4 through 3,000 new R04
Adam updates, one independent job per GPU.

The P107 stream was directly verified through complete round 61. A separate
single-GPU `qos_stu_default` job (`54223`, 4 CPU, 16G, 4h) completed the old
direct-point-cloud route for the highest initial point `r048_w02_i11` and the
highest Adam20 point `r057_w03_i04`; those outputs are retained as quarantined
diagnostic evidence. Corrected jobs `54321` and `54323` reran both samples with
the mandatory alpha+nu provenance chain and are now the canonical full-
evaluation results. Their DESC boundaries are nested; the R048 run also
reports a max-iteration warning. The canonical numerical and image evidence
is appended to the R012 report.

## DEC-20260904-04 - First-order score-gradient Flow policy

Status: active registered experiment; no default impact.

The student comparison keeps R04's q0 checkpoint, `nfp=8,nc=3`, RK4-32
generation, optimizer coordinate system, 64-direction centered gradient probe,
`h=0.0025`, ABI-11 R04 score library, and target helicity `[1,8]`. The only
policy change is the online update: valid scored samples contribute the direct
first-order term `g_flow^T grad_x ell_theta(x;z,t)` on the original sample;
invalid samples contribute ordinary Flow matching loss with `alpha=0.05`.
There is no explicit transport target, `rho`, `epsilon` displacement, Adam20
rollout, reward replay, or separate policy target.

Each round samples 64 centers over two Student GPUs, computes one native
128-endpoint gradient query per valid center, and performs exactly one global
DDP AdamW update. Four independent `(z,t)` draws per center are vectorized in
the batch. The transport forward forces math attention because the fused
attention backward has no second derivative; ordinary Flow loss keeps the
normal implementation. The current-coordinate chain uses an explicit local
Jacobian, with a float32-stable directional diagnostic.

Job `53957` runs from code commit `6ff70ab` in a separate worktree while R04
job `53372` continues in its preserved worktree. After three completed rounds,
valid rates were `14.06%, 17.19%, 15.63%, 23.44%`; accepted gradient rates
among valid centers were `100%, 100%, 100%, 86.7%`. The fixed
`beta=0.0060219592` was calibrated once from the q0 batch so the transport
term's absolute mean was 10% of ordinary valid loss, then frozen. This is an
early stability record, not evidence of promotion.

## DEC-20260905-01 - Replay-only score-gradient schedule

Status: active registered experiment; no default impact.

The score-gradient policy comparison now uses a 512-center FIFO replay pool
and 50 optimizer updates per outer round. Each update draws the unchanged
global batch size (64) uniformly with replacement from the synchronized pool.
The loss, score-gradient estimator, ABI-11 R04 evaluator, beta, optimizer,
and EMA rule remain unchanged; P107 reward weighting and source-mixture
sampling are deliberately excluded. The one-update run remains the historical
control. Job `54046` has a current complete snapshot through round 114
without a collapse signal and continues running.

## DEC-20260905-02 - Ten-update, faster-EMA schedule comparison

Status: registered experimental comparison; no default impact.

The Students replay50 score-gradient run is replaced by a schedule-only
comparison after the user requested a lower number of Flow updates per outer
round. The new run keeps the 64 newly collected centers per round, 32 centers
per rank, 512-record FIFO, 32-record-per-rank replay batch, ABI-11 R04 score
library, 64-direction gradient estimator, frozen beta, loss, AdamW settings,
and RK4-32 generation unchanged. It performs 10 replay updates per round
instead of 50 and uses EMA interpolation `0.1` instead of `0.01`. The old
`replay50` manifest and completed rounds remain frozen historical comparison
evidence; the new protocol is
`qh-axisflip-r012-score-gradient-replay10-ema10-rl-r04-abi11-v1`.

## DEC-20260905-03 - P107 radius comparison

Status: registered experimental comparison; no default impact.

The previous four-GPU R012 trajectory-replay job is stopped. Two new P107
jobs compare the same R04 ABI-11 trajectory-replay policy at compact-flexible
teacher radii `0.15 m` and `0.20 m`. Each radius gets an independently
regenerated 200,000-sample teacher corpus and a fresh converged q0 Flow
distillation. The online protocol, Adam20 estimator, score library, reward,
replay, loss mixture, 250 Flow updates, and 64 centers per round stay fixed.
To use all four GPUs concurrently, each job is a two-GPU/two-worker
decomposition with 32 centers per worker; this changes only resource
decomposition, not the global round sample or batch sizes.

The first submissions `54444` and `54446` stopped at q0 because the
distillation entry point still enforced a four-GPU world size. The corrected
submissions `55183` (`0.15 m`) and `55185` (`0.20 m`) use the explicit
two-GPU override, retain the same protocol, and reuse the already verified
teacher datasets. Both passed startup and were training q0 at the last check;
no radius result is available yet.

## DEC-20260905-02 - Mandatory alpha+nu full-evaluation chain

Status: active project-wide evaluation invariant.

Every new full physical evaluation uses the fixed single-job workflow:
source-psi selection, alpha fitting, nu toroidal correction, standard Simsopt
LS/Newton, surface selection, then Poincare, Boozer diagnostics, HTML and DESC.
The standard solver accepts only `kind=alpha_nu`, and downstream evaluation
accepts only `kind=alpha_nu_standard_ls_newton`. Alpha-only and direct GPU
point-cloud fits remain available solely as low-level diagnostics and cannot
be presented as full evaluations.

Strict dense residual limits retain their quality-grade meaning. A converged,
correctly wound, nondegenerate standard surface may continue with an explicit
`accepted_with_quality_warning`; the report must display each exceeded metric
and limit. Independent samples are submitted as separate jobs and run in
parallel whenever the verified resource allowance permits.
