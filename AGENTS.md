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

## Source Of Truth

4. `MEMORY.md` contains current truth only. Keep it short, dated, and below
   250 lines. Verify volatile branch, job, and filesystem state directly rather
   than preserving a stale transcript.
5. Record a newly discovered model or user-reported error in
   `memory/CORRECTIONS.md` during the same turn. Preserve the erroneous artifact,
   mark it superseded at its point of use when practical, and state exactly
   which conclusions remain valid.
6. Update current decisions in `memory/DECISIONS.md` and collapse completed work
   in `MEMORY.md` to an outcome plus evidence pointer. Never recreate a full
   chronological log in the hot memory.

## Protocol And Branch Rules

7. The current QH default is protocol `qh-flow-screen32-adam200-64d-v1`: screen
   32 Flow starts, then run 200 latent Adam steps with 64 fresh orthogonal
   centered directions, `h=0.005`, `lr=0.02`, beta `(0.7, 0.999)`, and FP32
   RK4-128. Code and manifests outrank prose summaries.
8. Two-direction optimization is deprecated historical evidence. It must not
   run in a future experiment. A future explicit user request involving two
   directions requires a new named protocol, a dedicated exploration branch,
   a new launcher and manifest, and deliberate review of the hard guard. Never
   reactivate or copy a historical 2D launcher.
9. Put material exploration on a `codex/` branch when isolation is useful.
   When the user accepts a method as the default, integrate it into `main` and
   update code defaults, manifests, tests, current documentation, decisions,
   and corrections together. Earlier methods remain labeled historical.
10. Reproductions must load the original machine-readable manifest and pin its
    code, score library, checkpoint, parameter space, and optimizer settings.
    A current CLI default is not a substitute for a frozen historical protocol.

## Communication And Hygiene

11. For external-facing material, follow `memory/WRITING.md` during drafting
    and run its post-generation audit before delivery. Internal status reports
    to the user may be direct and diagnostic.
12. Preserve unrelated and untracked artifacts. Never store or print passwords,
    tokens, private keys, one-time codes, or credential-bearing URLs.
