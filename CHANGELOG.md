# Changelog

## 0.3.0 — Public-release synchronization

Tracks official SciFigBench commit 13088544a1884a1d25286932ce9eab65943ba677.

- Adds verified Hub downloads, public QA loading, model-safe visual payloads,
  all-task evaluation and the one-call SciFigBench API with durable resume.
- Preserves real Jittor CPU/float64 numerical execution and SciPy exact matching.
- Synchronizes public schemas, JSON-first parsing, LaTeX normalization (including
  proportional-to), canonical rendering and official scoring edge cases.
- Adds backend metadata to reports and runtime identity checks to resume.
- Refreshes README, data availability, paper citation, examples and release checks.

### Migration from 0.2.0

Use a fresh Linux/WSL2 virtual environment with Python 3.10–3.12. The distribution
remains scifigbench-jittor; import and CLI remain scifigbench. Do not install the
official scifigbench or old scifigbench-toolkit alongside it.

The preview API, validate/render CLI commands, --qa/--pred/--report-dir options
and report schemas 1.0/1.1 are replaced. Use:

```bash
scifigbench evaluate --predictions predictions.jsonl --task all --data-root ./data --output report.json
```

Python evaluate(predictions, data_root=..., task=...) now returns a report dict.
Use load_qa, render and prepare_model_payload for validation and image inputs.
QA uses QA/qa_<task>.jsonl, anonymous images/Fxxxxxx.jpg paths and public meta
fields. DCP K comes from its boxes; construction subtask/bucket tags are removed.
Predictions must contain qid and raw string response; historical field names
require an explicit prediction_field. Native list/bool/int responses are errors.

Full selected-QA coverage is required by default. --allow-partial scores matched
questions only; unassessed tasks return null, not 0. Missing questions no longer
enter the score denominator. JSON lists are parsed before Markdown, and approved
list item coercion is retained. Square matching pads with empty strings, which
can earn credit against empty or normalized-empty predictions. IFT parsing
failure becomes an empty prediction and scores 1 when its reference is empty,
while n_parse_fail still counts the failure. These official behaviors replace
the preview's stricter policies. Do not compare old/new scores without noting
the data revision, coverage and policy changes.

Synthetic fixtures are now synthetic-v2. Correct predictions still score 1 on
all tasks. Mixed Edge and IFT now score 1 over their matched questions with
coverage 1/2 and 2/3 respectively; the latter includes an empty-answer parse
failure. See examples/synthetic/expected.json.

## 0.2.0

Renamed the preview to SciFigBench-Jittor and migrated scoring to Jittor CPU.

## 0.1.0

Initial evaluation toolkit preview and original synthetic examples.
