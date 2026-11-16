# Code provenance

SciFigBench-Jittor 0.3.0 tracks LeNcyer/SciFigBench commit
`13088544a1884a1d25286932ce9eab65943ba677` (October 21, 2026):
https://github.com/LeNcyer/SciFigBench/tree/13088544a1884a1d25286932ce9eab65943ba677

The public API, dataset contracts, download verification, canonical rendering,
runner, durable resume, parsers, normalization and task scoring conventions are
adapted from that MIT-licensed revision. Copyright and attribution remain with
the SciFigBench authors. This repository retains its independent Git history
and its Jittor CPU backend from 0.2.0.

The implementation differences are numerical execution and provenance:

- Jittor CPU C++ operators compute exact token edit distances.
- Jittor float64 computes similarity, accuracy, coverage and score reductions.
- SciPy supplies exact Hungarian assignment indices; Jittor gathers and reduces
  the selected values. Square empty-string padding matches upstream, including
  empty-string and normalization-to-empty behavior.
- Execution metadata is added to evaluator and runner reports, and backend
  versions participate in strict resume identity. Initialization remains lazy;
  user Callables execute outside the CPU scoring context.
- Missing Jittor/compiler support is an explicit error, without backend fallback.

The pinned upstream numerical/data/evaluation sources and license are preserved
under tests/reference_upstream for offline differential tests. They are test-only
and are not installed in the runtime wheel. The suite checks per-question and
aggregate behavior to absolute tolerance 1e-12, including official empty-answer
and parse-failure semantics. The local package is not a dependency of upstream.

The original synthetic diagram is retained and re-encoded to the public image
layout. Synthetic-v2 QA removes preview construction metadata and uses public
field whitelists. Preview expected scores change with the approved upstream
parser and matched-question denominator; see CHANGELOG.md.

No official data, model weights, private mappings or actual experiment responses
are imported. Jittor is an external Apache-2.0 dependency; it is not vendored.
