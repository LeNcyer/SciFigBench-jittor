# Changelog

## 0.2.0 — Jittor CPU preview

- Rename the repository/distribution to SciFigBench-jittor / scifigbench-jittor.
  Keep the scifigbench import, CLI and existing Git history.
- Execute token edit distance, ANLS, accuracy and report reductions through Jittor
  CPU operators, with explicit float64 scoring and exact large-integer comparisons.
- Keep SciPy for optimal assignment indices and Pillow for rendering.
- Require Linux/WSL2, Python 3.10–3.12 and a C++ compiler; lock Jittor and NumPy 1.x.
- Add execution metadata to report schema 1.1, retaining legacy 1.0 compatibility.
- Keep validate/render/version independent of Jittor initialization and direct
  compilation logs to stderr; fail clearly when the required backend cannot run.
- Treat oversized integer and deeply nested list responses as scored parse failures.
- Reject case-colliding render filenames before writing, including on WSL mounts.
- Add Linux CPU CI, independent numerical checks and isolated wheel operator tests
  in environments without PyTorch, Accelerate or xformers.

## 0.1.0 — Preview

- Package five scientific-figure task scorers, LaTeX-aware metrics and a renderer.
- Add validate, render and evaluate commands, packaged JSON Schemas and synthetic examples.
- Count missing predictions as zero using the full selected-task denominator.
- Distinguish missing responses, parse failures and valid empty lists.
- Require SciPy optimal assignment at every list length.
- Preserve duplicates in set-ANLS; unmatched slots explicitly score zero.
- Reject duplicate/unknown IDs, inconsistent response fields and invalid QA.
- Add task-specific reports, input hashes and explicit overwrite handling.
- Include tests, dependency locking and wheel installation checks.

This preview has no official dataset or model benchmark results. Complete,
parseable predictions retain the source metrics except for the documented
unmatched-empty-slot correction. Old reports using partial denominators or a
greedy assignment fallback must not be treated as equivalent.
