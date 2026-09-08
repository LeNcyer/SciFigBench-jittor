# SciFigBench Toolkit — Evaluation Code Preview

Release tag: v0.1.0-preview. Package version: 0.1.0.
Release type: Pre-release.

This preview opens the evaluation and rendering toolkit for five scientific
figure understanding tasks. The benchmark dataset is undergoing second-pass
validation and is not included. Synthetic demonstration scores are not benchmark
results.

Included:

- Accuracy, token-level ANLS, position-aligned ANLS and optimal set-ANLS*.
- Input validation, prediction parsing, rendering and task-specific reports.
- Full-QA denominators, coverage, missing/parse-failure accounting and input hashes.
- Original synthetic examples, expected outputs, tests and CI configuration.
- Source and wheel distributions under the MIT license.

Install the source checkout with python -m pip install -e ., or install the
attached wheel. Follow the README quick start. No GPU or model service is needed.

See CHANGELOG.md and PROVENANCE.md for differences from the source evaluator.
Dataset and toolkit versions will be tracked separately.
