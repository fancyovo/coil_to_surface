# Archived Codex Handoff: cem_qh03 DESC Initial Guess

> **Archived 2026-07-31.** This file describes the project state on 2026-07-10
> and is retained only as historical context. Read `MEMORY.md` for the current
> branch, active jobs, validated score/model hashes, workflows, and next steps.

Last reconciled: 2026-07-10 (Asia/Shanghai)

## 1. Objective

The active objective is to construct a better DESC volume initial guess for
`cem_qh03`. The evaluator can find a local/Boozer-solvable surface, but the
current DESC paths do not produce a trustworthy nested fixed-boundary
equilibrium.

The immediate priority is narrower than "make DESC converge": determine at
which layer the volume geometry first loses nestedness:

1. fitted `psi` and its ray-extracted level surfaces;
2. the `R/Z` Fourier-Zernike volume fit;
3. the boundary parameterization used together with the interior layers;
4. only after a nested `R/Z` volume exists, the `lambda/beta/iota` phase fit;
5. only after all pre-solve geometry gates pass, a DESC force solve.

## 2. Repository State (Source of Truth)

- Branch: `desc-psi-volume-initial-guess`
- `HEAD`: `c5ac4d8e10bff0ff1ccdd4f26a733e48473d808e`
- `main`: the same commit
- Commit subject: `Support older Simsopt Boozer LS signature`
- Tracked changes before this handoff: none
- Untracked files before this handoff: 105
- This file is an additional untracked file.

All work for this branch currently lives in untracked files. In particular:

- `stellarator_eval/desc_joint_ls.py`
- `scripts/desc_external_rzl_data_ls_experiment.py`
- `scripts/desc_joint_rzl_initial_guess_experiment.py`
- `scripts/desc_psi_volume_initial_guess_experiment.py`
- `scripts/diagnose_desc_rz_nesting.py`
- `scripts/plot_poincare_validation.py`
- `scripts/plot_psi_level_poincare_probe.py`
- `tests/test_desc_joint_ls.py`
- `reports/` (88 untracked files at reconciliation time)
- five untracked case files under `examples/`

There are also unrelated-looking untracked top-level utilities
(`metric_c.py`, `metric_s.py`, `tmp.py`, and `viz_stell.py`). Do not remove or
rewrite them without separate inspection.

`runs/` is ignored by `.gitignore`. At reconciliation time it contained 488
files (about 390 MB), with the newest local run artifact dated 2026-07-09.
No run was started during recovery.

## 3. Completed Work

### Joint linear LS implementation

`stellarator_eval/desc_joint_ls.py` implements external data-driven fits for
`R/Z/L` and phase constraints with per-field-line `beta` plus an `iota(rho)`
profile. The phase equation is consistent with DESC's convention

```text
theta_PEST = theta_DESC + lambda
lambda(q_i) - beta_s - zeta_i * iota(rho_i) = -theta_i.
```

This phase target can fit straight-field-line coordinates, but it does not by
itself enforce nested `R/Z` geometry or force balance.

Three confirmed implementation fixes are present in the working tree:

1. `A_beta` uses paired row/column indexing, so every phase sample connects to
   exactly one field-line `beta_s`.
2. Phase constraints now receive the CLI `lambda_weight`.
3. Ridge coefficients enter the augmented least-squares matrix with
   `sqrt(alpha)` scaling, rather than applying the intended regularization
   strength twice.

Focused tests in `tests/test_desc_joint_ls.py` cover the one-beta-entry matrix
structure and exact synthetic recovery of `L`, `beta`, and `iota`.

### Audit report

`reports/desc_joint_linear_ls_audit.md` documents the calculation flow and the
DESC interface audit. Its current high-level conclusions are:

- the old phase RMS near `1.570 rad` was dominated by the `A_beta` indexing
  bug; an isolated corrected run reduced it to about `0.377 rad`;
- after that correction, the tested volume was still non-nested;
- about 18.5% of sampled `sqrt(g)` had the minority sign, so `R/Z` geometry was
  already folded before `lambda` could repair anything;
- tiny median force coexisted with singular-point values near `3e6`, so a force
  mean alone was not a valid physics acceptance metric;
- local DESC source and the remotely installed DESC 0.16.0 are not identical;
  interface claims were rechecked against the installed version.

### R/Z nesting diagnostic

`scripts/diagnose_desc_rz_nesting.py` has been added, but has not been run. It
is intended to compare:

- fitted-`psi` root residual and `dpsi/dr` monotonicity;
- adjacent ray radius gaps between `psi` levels;
- Poincare return drift relative to fitted `psi` contours;
- `R/Z` fit residuals and DESC `sqrt(g)` sign statistics;
- edge scales `1.0, 0.75, 0.5, 0.35`;
- spectral resolutions `4, 6, 8`;
- boundary choices `psi_ray`, `boozer_native`, and `boozer_geometric`.

The main current hypothesis is a parameterization mismatch: interior `psi`
layers use a geometric ray angle about the magnetic axis, while the native
Boozer boundary uses its own surface spectral parameter. Treating both as the
same DESC computational `theta` can fold the fitted volume even when each
surface is individually well behaved. This is not yet experimentally proven.

## 4. Recovered Experiment Context

The following results come from the old thread's remote experiments. The
corresponding remote `runs/desc_*` directories are not present in the local
`runs/` tree, so treat the numbers as recovered context rather than locally
reproduced results.

### Earlier point/trace fits

- Dense field-line `trace_RZ_lambda0` reached initial normalized mean force
  about `4.35e-3`, but remained non-nested and had a `rho=1` body mismatch of
  about `8.67e-2 m`.
- Directly appending boundary points improved boundary mismatch to about
  `2.33e-2 m` but made force much worse (about `4.76e5`).
- A direct DESC solve from the dense trace fit was non-nested and had a large
  bad-solution cost (about `3.18e10`). `ensure_nested=True` and a lambda refine
  were worse.
- Shrinking the trace fit to `rho_max=0.55` did not fix the representation.

### External `R/Z/L` and phase fits

- In the earlier external target fit, `trace_joint_lambda_sfl_minus` was the
  best tested variant at initial normalized mean force about `1.895`, but was
  non-nested.
- The first joint `L/beta/iota` phase variant gave about `2.382` and remained
  non-nested. A simple theta sign flip was worse (about `12.81`), and the
  initial psi/phase hybrid was much worse (about `52.29`).
- Those phase results predated the confirmed `A_beta` fix. They must not be
  used as final comparisons without rerunning the corrected implementation.

### Important corrected interpretation

An older report initially described a very small DESC force objective from a
`psi_inner` path. The same report later corrected that conclusion after
finding boundary handling/overwrite errors. With corrected boundary handling,
the result remained non-nested and had high force residual. Do not cite the
old small objective as a valid equilibrium result.

## 5. Local Artifacts and Missing Inputs

The latest report is `reports/desc_joint_linear_ls_audit.md` (2026-07-10).
`reports/desc_psi_volume_initial_guess/` contains copied summaries and prior
geometry plots, including a diagnostic that reported the first final-solution
Jacobian sign flip near `rho=0.293`; that diagnostic belongs to the older DESC
path and does not replace the new pre-fit layer isolation.

The local `runs/cem_qh03_full_eval/` contains only `cem_qh03_raw.json`. It does
not contain the inputs required by `diagnose_desc_rz_nesting.py`, namely a run
`summary.json`, `axis_data.npz`, `psi_model.npz`, and the selected
`boozer_surface.npz`. Existing reports refer to remote run paths that were not
synced locally.

Therefore the new diagnostic cannot be reproduced locally from current files
alone. Either run it against the existing remote evaluator outputs or sync the
minimal required inputs without overwriting local artifacts.

## 6. Verification Performed During Recovery

- Rollout parsed line by line: 1,068 JSONL records, zero JSON parse failures.
  Only sanitized user messages, assistant conclusions, plan events, recent
  tool calls, and failure events were inspected. Credentials were omitted.
- `python -m pytest tests/test_desc_joint_ls.py -q`: `2 passed`.
- `python -m py_compile` on the joint-LS module, all related experiment and
  plotting scripts, the new diagnostic, and its test: passed.
- `python scripts/diagnose_desc_rz_nesting.py --help`: failed locally at import
  because `simsopt` is not installed in the local Python environment. This is
  an environment limitation, not a completed runtime validation.
- No expensive evaluator, field-line, Boozer, or DESC calculation was run.
- No existing result was overwritten.

Relevant old-thread failures/interruption context:

- five turns were interrupted;
- one PowerShell source search failed due quoting and was rerun correctly;
- the first synthetic phase recovery test used degenerate field lines whose
  constant lambda could be absorbed into free beta; the fixture was corrected
  and then both tests passed;
- the old thread ended immediately after adding the nesting diagnostic, before
  a remote run or final nesting report.

## 7. Remote Execution Constraints

Do not record or echo credentials. For any remote work:

- work only under `~/local_surface_evaluator`;
- use at most one GPU;
- use at most 16 CPU cores;
- keep outputs in new run-specific directories;
- do not overwrite earlier runs or reports;
- on completion or failure, terminate launched processes and check that no
  background or zombie processes remain.

The latest code fixes and the new diagnostic have not been confirmed synced to
the remote workspace.

## 8. Remaining Steps

1. Review the new diagnostic's runtime imports and data assumptions against the
   remote DESC 0.16.0 environment, then sync only the required changed files.
2. Run one constrained `cem_qh03` diagnostic in a new output directory. Do not
   run DESC solve yet.
3. Decide from plots and JSON metrics whether failure starts in fitted `psi`,
   Poincare returns, boundary theta parameterization, or the finite `R/Z`
   basis. Compare all three boundary variants and the reduced edge scales.
4. If `psi` surfaces are good and a parameterization/resolution fix yields
   single-sign `sqrt(g)`, implement the smallest clear correction and repeat
   only the pre-solve geometry gates.
5. Once `R/Z` is nested, hold it fixed and refit `L/beta/iota`; require
   single-sign `sqrt(g)_PEST` and `min(1 + lambda_t) > 0` before DESC solve.
6. Run a limited DESC solve only for a candidate passing boundary, Jacobian,
   and phase gates. Report force quantiles and boundary mismatch, not only the
   optimizer objective.
7. Regress the accepted workflow on `cem_1` and `cem_3`.
8. Write the final report with the psi nesting, Poincare, resolution sweep, and
   Jacobian maps, clearly separating verified results from hypotheses.

