# Repository Agent Instructions

## Context Loading

1. Read `MEMORY.md` once at the start of a fresh agent run, before substantive
   planning or changes. Do not reread it for every user message.
2. Read `MEMORY.md` again after context compaction or handoff, and after a
   checkout, merge, rebase, or external update that may have changed the
   repository baseline.
3. Load only the routed file needed for the task:
   - experiments or defaults: `memory/PROTOCOLS.md`;
   - interpreting disputed or superseded claims: `memory/CORRECTIONS.md`;
   - design and promotion decisions: `memory/DECISIONS.md`;
   - external documents, reports, or multimodal deliverables:
     `memory/WRITING.md`;
   - older chronology: `memory/HISTORY.md`.
   Archive files are never part of routine context.
4. Before the first remote read, edit, synchronization, or Slurm operation in a
   fresh run or after compaction, read `REMOTE_CODEX_INSTRUCTIONS.md` in full
   and execute its mandatory preflight in order. Read it again and restart the
   preflight after any SSH error, network change, or computer sleep. Never
   substitute direct SSH authentication, a guessed host/path, or a locally
   unlocked key for the documented WSL master connection.

## Source Of Truth

5. `MEMORY.md` contains current truth only. Keep it short, dated, and below
   250 lines. Verify volatile branch, job, and filesystem state directly rather
   than preserving a stale transcript.
6. Record a newly discovered model or user-reported error in
   `memory/CORRECTIONS.md` during the same turn. Preserve the erroneous artifact,
   mark it superseded at its point of use when practical, and state exactly
   which conclusions remain valid.
7. Update current decisions in `memory/DECISIONS.md` and collapse completed work
   in `MEMORY.md` to an outcome plus evidence pointer. Never recreate a full
   chronological log in the hot memory.

## Protocol And Branch Rules

8. The current QH default is protocol `qh-flow-screen32-adam200-64d-v1`: screen
   32 Flow starts, then run 200 latent Adam steps with 64 fresh orthogonal
   centered directions, `h=0.005`, `lr=0.02`, beta `(0.7, 0.999)`, and FP32
   RK4-128. Code and manifests outrank prose summaries.
9. Two-direction optimization is deprecated historical evidence. It must not
   run in a future experiment. A future explicit user request involving two
   directions requires a new named protocol, a dedicated exploration branch,
   a new launcher and manifest, and deliberate review of the hard guard. Never
   reactivate or copy a historical 2D launcher.
10. Put material exploration on a `codex/` branch and switch the primary,
   user-visible checkout to that branch before substantive work. Use a secondary
   worktree only when preserved local changes or concurrent work make the switch
   unsafe, and state that exception explicitly. When the user accepts a method
   as the default, integrate it into `main` and update code defaults, manifests,
   tests, current documentation, decisions, and corrections together. Earlier
   methods remain labeled historical.
11. Reproductions must load the original machine-readable manifest and pin its
    code, score library, checkpoint, parameter space, and optimizer settings.
    A current CLI default is not a substitute for a frozen historical protocol.
12. Every full physical evaluation must use the fixed single-job workflow in
    `evaluation/full_physical/`: source psi, alpha, nu, standard Simsopt
    LS/Newton, then downstream diagnostics and DESC. Alpha-only and direct GPU
    point-cloud surfaces are not valid full-evaluation initializers or
    downstream artifacts. The NPZ provenance guards must remain enabled.

## Execution And Scheduling

13. Submit independent jobs concurrently up to the verified resource allowance.
    This applies across samples, candidate values, seeds, and downstream CPU
    evaluations. A per-job one-GPU limit never implies serializing the batch
    when the user has authorized multiple GPUs.
14. Serial execution requires a true data dependency, a verified scheduler or
    resource restriction, or an explicit user request. Record the reason in the
    machine-readable run metadata. Never finish one independent sample before
    starting another merely for operational convenience.

## Communication And Hygiene

15. For external-facing material, follow `memory/WRITING.md` during drafting
    and run its post-generation audit before delivery. Internal status reports
    to the user may be direct and diagnostic.
16. Keep the canonical report and its assets tracked on the owning branch. At
    delivery, mirror them into the primary checkout's Git-excluded
    `_shared_reports/` directory and verify that the mirrored document resolves
    every local asset. This local mirror persists across branch switches; it is
    a delivery surface, not the provenance source.
17. Preserve unrelated and untracked artifacts. Never store or print passwords,
    tokens, private keys, one-time codes, or credential-bearing URLs.
