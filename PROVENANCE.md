# Code provenance

The LaTeX normalization, tokenization and ANLS implementation is adapted from
LeNcyer/SciFigBench, commit
edfde8175890bc65f884a87c1cf17b5cbdd13d2e.
The upstream file was eval/_latex_norm.py. Task scoring, response parsing and
rendering conventions are adapted from that revision's task evaluators,
eval/_eval_lib.py and tools/render_eval_input.py.

Copyright and MIT attribution remain with SciFigBench authors. The upstream
repository can be referenced at https://github.com/LeNcyer/SciFigBench.

This project started an independent Git history as SciFigBench-toolkit. Version
0.2.0 renames it to SciFigBench-jittor and retains that history. No dataset blobs, image
annotations, private paths or experiment output directories are imported.
The synthetic assets are generated specifically for this toolkit.

Changes from the upstream evaluator:

- Missing predictions contribute zero to all task and subgroup denominators.
- Failed parsing is always zero, including when the reference is an empty list.
- IDs and row structure are validated rather than silently overwritten/skipped.
- Optimal assignment always uses SciPy, without brute-force/greedy fallbacks.
- Unmatched list slots receive zero explicitly, including empty-string predictions.
- Lists can be native JSON arrays. Elements must be strings.
- Rendering resolves relative paths against an explicit data root.
- Reports have task-specific destinations, overwrite protection and input hashes.
- Metadata and counts come from inputs rather than private experiment conventions.

Metric formula changes beyond these are not part of this preview. In particular,
dense single-box scoring still uses only the first parsed string.

## Jittor migration (0.2.0)

Normalization and tokenization retain the upstream implementation. A new Jittor
CPU C++ operator implements the same two-row token edit-distance recurrence.
Jittor float64 operations replace Python/NumPy scoring arithmetic and reductions.
SciPy remains the exact assignment solver; its indices are gathered and reduced
by Jittor. Rectangular assignment replaces zero-padded square assignment while
retaining the maximum-cardinality denominator and zero credit for unmatched slots.

The migration includes independent score comparisons and actual CPU operator
tests. It adds runtime provenance to reports and restores parse-failure isolation
for oversized integer strings and excessively nested JSON responses. No model
implementation, weights, PyTorch compatibility layer or inference service is added.

The toolkit code remains MIT licensed. Jittor is a separately installed dependency
under its own Apache-2.0 license; its source is not vendored into this repository.
