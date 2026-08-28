# Memory Architecture

The memory system separates current truth from history so context stays small
without discarding provenance.

## Loading Route

| Need | Read |
| --- | --- |
| Fresh run or post-compaction orientation | root `MEMORY.md` |
| Starting, reproducing, or promoting an experiment | `PROTOCOLS.md` |
| A report or remembered claim may be wrong | `CORRECTIONS.md` |
| Why the current method or branch policy was chosen | `DECISIONS.md` |
| External report, document, figure, or video text | `WRITING.md` |
| Older milestone or report pointer | `HISTORY.md` |
| Forensic recovery explicitly requiring original wording | `archive/` |

Do not reload this tree on every user message. `AGENTS.md` defines the few
events that require another root-memory read. Load one routed file at a time.

## Ownership

- `MEMORY.md` owns current, high-risk operational truth and routes deeper reads.
- `PROTOCOLS.md` owns executable defaults, experiment identity, and deprecation.
- `CORRECTIONS.md` owns errors. It is append-only at the entry level: update an
  entry's status, but do not erase the original error or its impact.
- `DECISIONS.md` owns accepted methods and promotion semantics.
- `WRITING.md` owns external-facing language and post-generation review.
- `HISTORY.md` owns compact chronology and evidence pointers.
- Reports and manifests own detailed tables and frozen run metadata.
- `archive/` owns immutable snapshots and has no current authority.

## Update Rules

1. Verify facts from Git, manifests, artifacts, tests, or scheduler output.
2. Update the owning file only; add a short root-memory anchor only when
   forgetting the fact could cause an incorrect run or conclusion.
3. Date observations and distinguish recorded metadata from a new evaluation.
4. When a current decision changes, mark the prior decision superseded and add
   a history pointer. Never silently rewrite history.
5. When an error is found, create or update a correction before relying on the
   affected evidence again.
6. Keep secrets and credential-bearing paths out of every memory file.
