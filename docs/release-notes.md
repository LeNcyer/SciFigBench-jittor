# SciFigBench-Jittor — Jittor CPU Evaluation Preview

Release tag: v0.2.0-preview. Package version: 0.2.0.
Release type: Pre-release.

This preview provides Jittor CPU evaluation and rendering for five scientific
figure understanding tasks. The benchmark dataset is undergoing second-pass
validation and is not included. Synthetic demonstration scores are not benchmark
results.

Included:

- Jittor CPU edit distance, accuracy, token-level ANLS, position-aligned ANLS and
  score reductions in float64; SciPy exact assignment indices for set-ANLS*.
- Input validation, prediction parsing, rendering and task-specific reports.
- Full-QA denominators, coverage, missing/parse-failure accounting, input hashes
  and runtime provenance in report schema 1.1, with legacy 1.0 compatibility.
- Original synthetic examples, expected outputs, tests and CI configuration.
- Source and wheel distributions under the MIT license.

Use Linux or WSL2, Python 3.10–3.12, and a C++ compiler. Install the source checkout
with python -m pip install -e ., or install the attached wheel. Follow the README
quick start. The first scoring run compiles Jittor; subsequent runs reuse its
cache. No GPU, model service, PyTorch, Accelerate, xformers or distributed launcher
is needed. Native Windows and GPU execution are outside this preview's support.

Create a fresh virtual environment when migrating from scifigbench-toolkit: both
packages use the scifigbench import/CLI and must not be installed together. The
repository rename preserves history. Synthetic-v1 data and metric definitions
remain unchanged; this release adds no official benchmark data or model inference.

See CHANGELOG.md and PROVENANCE.md for differences from the source evaluator.
Dataset and toolkit versions will be tracked separately.
