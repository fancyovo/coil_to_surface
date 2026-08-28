# Repository Agent Instructions

1. At the start of every conversation, and immediately after any context
   compaction or handoff, read `MEMORY.md` before planning or changing code.
2. Treat `MEMORY.md` as the living source of truth for the active branch,
   remote jobs, validated numerical conclusions, required workflows, and known
   invalid results. `CODEX_HANDOFF.md` is an archived historical handoff.
3. Update `MEMORY.md` in the same turn whenever an important fact changes. This
   includes branch/commit state, submitted or completed jobs, validated hashes,
   interfaces, numerical conclusions, user requirements, discovered bugs, and
   invalidated results.
4. Date updates, distinguish recorded metadata from current re-evaluation, and
   mark superseded or invalid results explicitly. Never store passwords,
   authentication tokens, private keys, or one-time codes in repository files.
5. Before the final response, verify that any important work completed in the
   turn is reflected in `MEMORY.md`.
6. `MEMORY_archive_20260808.md` is the immutable full history from before the
   2026-08-08 compaction. Do not update it or read it by default; consult it only
   when `MEMORY.md` or a current report points to historical detail.
7. Keep `MEMORY.md` below a soft limit of 500 lines. Active work may be detailed,
   but completed jobs must be collapsed to outcomes and report pointers. Do not
   recreate a full dated change log in the living memory.
