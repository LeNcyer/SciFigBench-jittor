# Code provenance

The LaTeX normalization, tokenization and ANLS implementation is adapted from
LeNcyer/SciFigBench, commit
edfde8175890bc65f884a87c1cf17b5cbdd13d2e.
The upstream file was eval/_latex_norm.py. Task scoring, response parsing and
rendering conventions are adapted from that revision's task evaluators,
eval/_eval_lib.py and tools/render_eval_input.py.

Copyright and MIT attribution remain with SciFigBench authors. The upstream
repository can be referenced at https://github.com/LeNcyer/SciFigBench.

This project starts an independent Git history. No dataset blobs, image
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
