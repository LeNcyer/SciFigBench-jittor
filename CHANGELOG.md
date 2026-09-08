# Changelog

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
