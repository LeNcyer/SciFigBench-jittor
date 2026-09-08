# SciFigBench Toolkit

Evaluation and image-rendering tools for scientific figure understanding.

**Preview release:** This repository releases the evaluation toolkit and original
synthetic examples only. The SciFigBench dataset is undergoing second-pass
validation. This toolkit does not distribute the benchmark dataset or final
benchmark results. Synthetic example scores are software checks, not model scores.

## Quick start

Python 3.10 or newer is required. No GPU, SSH connection, model API or external
benchmark data is needed. Run these commands from this repository's root:

~~~bash
python -m pip install -e .
scifigbench --version
scifigbench validate --qa examples/synthetic/qa/edge_level_verification.jsonl
scifigbench render --qa examples/synthetic/qa/edge_level_verification.jsonl --data-root examples/synthetic --out-dir rendered/edge
scifigbench evaluate --task edge_level_verification --qa examples/synthetic/qa/edge_level_verification.jsonl --pred examples/synthetic/predictions/correct/edge_level_verification.jsonl --report-dir reports/correct
~~~

The last command produces accuracy 1.0, coverage 1.0, and zero missing/invalid
responses. Results appear under reports/correct/edge_level_verification/.
To rerun into the same destination, add --overwrite.

To render and evaluate **all five tasks**, including intentionally incorrect,
missing and malformed predictions:

~~~bash
python examples/synthetic/run_demo.py --out-dir reports/synthetic
~~~

The demo checks scores against hand-specified expected values. CI runs this same
script against an installed wheel, from outside the source checkout.

## Tasks

| Task | Output | Primary metric |
|---|---|---|
| Dense Content Perception | JSON list of strings | Token-level ANLS; positions matter for multiple boxes |
| Edge-Level Verification | true / false | Accuracy |
| Node-Level Degree Counting | Integer | Accuracy |
| Path-Level Reachability | true / false | Accuracy |
| Information-Flow Tracing | JSON list of strings | Optimal-assignment set-ANLS* |

For custom evaluations, provide your own QA JSONL and predictions. Prediction
rows normally contain qid and response. Each evaluation selects one task; all
prediction IDs must belong to that selected task.

## Documentation

- [Task definitions](docs/tasks.md)
- [QA, predictions and reports](docs/data-format.md)
- [Metrics, parsing and evaluation policy](docs/evaluation.md)
- [Data availability](DATA_AVAILABILITY.md)
- [Code provenance](PROVENANCE.md)
- [Changes](CHANGELOG.md)
- [Preview release notes](docs/release-notes.md)
- [Synthetic examples](examples/synthetic/README.md)

Missing and unparseable predictions score zero. Every QA contributes to the
task denominator. Duplicate and unknown IDs are errors. Reports include input
SHA-256 hashes, software/schema/data versions, coverage and subgroup scores.

## Development and reproducibility

~~~bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m build
~~~

For locked development dependencies with uv:

~~~bash
uv sync --frozen --extra dev
uv run --frozen --extra dev python -m pytest
~~~

The source distribution includes docs, tests and synthetic examples. The wheel
contains the importable package, packaged schemas and license. Install a downloaded
wheel with python -m pip install <wheel-file>; the CLI is also available as
python -m scifigbench. Source and wheel packages are distributed through
[GitHub Releases](https://github.com/LeNcyer/SciFigBench-toolkit/releases);
there is no PyPI release.

## License and citation

MIT, copyright 2026 SciFigBench authors. Original synthetic figures and questions
in this repository are covered by the same license. No third-party paper figures
are included. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).

When citing the preview, identify toolkit version 0.1.0 and tag
v0.1.0-preview. Dataset availability and future benchmark versions are tracked
separately.
