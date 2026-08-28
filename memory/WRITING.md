# External Writing And Multimodal Review

Last reviewed: 2026-08-28 (Asia/Shanghai).

These rules apply to documents, reports, README text intended for outsiders,
submissions, captions, diagrams, slides, video overlays, and generated images.
They do not prevent direct diagnostic language in private status updates to the
user.

## Drafting Rules

1. State the subject, version, method, and conclusion directly. Prefer an
   affirmative explanation of what the project does.
2. Avoid defensive contrast that denies an unasked claim. Rewrite stock forms
   such as "不是...而是...", "并非...而是...", and "not X but Y" unless the
   audience explicitly raised X and resolving that dispute is necessary.
3. Name the referent. Replace vague phrases such as "this method", "the prior
   event", "the above result", or "now" when multiple versions or experiments
   could fit. Use a protocol ID, report date, sample ID, or section name.
4. Keep every logical step connected to the document's core claim. Remove
   speculative transitions, ornamental abstractions, and statements whose
   meaning or evidentiary role cannot be explained in one sentence.
5. Separate current methods, registered experiments, historical evidence, and
   invalidated conclusions. Historical labels inside old artifacts never define
   the current default.
6. Attach numbers to their score definition, protocol, sample set, and evidence.
   Do not compare incompatible objectives or imply causality from a complete-
   recipe comparison.
7. Make captions and visual labels understandable on their own. Identify the
   protocol, quantity, units, cohort, and status; do not rely on an ambiguous
   pronoun or on history absent from the figure.

## Required Post-Generation Audit

Run this review on the rendered deliverable, not only its source:

- Search for defensive contrast forms and rewrite them unless each denial
  answers an explicit audience question.
- Resolve every ambiguous pronoun, temporal reference, and versionless use of
  `default`, `current`, `previous`, `production`, or `standard`.
- Check every method value against `PROTOCOLS.md` or a frozen manifest and every
  corrected claim against `CORRECTIONS.md`.
- Remove claims unrelated to the central argument and repair any unexplained
  jump between evidence and conclusion.
- Inspect all figures, captions, legends, callouts, and video overlays for the
  same semantic problems, plus clipping, occlusion, and unreadable text.
- Confirm that historical or invalid material is visibly labeled at its point
  of use and cannot be mistaken for the active method.

Delivery is incomplete until this second pass is done.
