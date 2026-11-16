# Public data and prediction formats

The public contract is the official SciFigBench release at commit 13088544.
Data roots contain QA/qa_<task>.jsonl and QA/manifest.json; images live at
images/Fxxxxxx.jpg. Downloaded subsets also contain SHA-256 manifests.

Every QA has qid, task, sample_id, image_path, image_size, prompt and answer.
The loader validates exact task-specific field/meta whitelists, finite bounded
coordinates, unique QIDs and consistent sample/image names. DCP uses 1–6
boxes_xyxy and the corresponding prefix of red/orange/yellow/green/blue/violet;
K is derived from the number of boxes. Public construction subtask/bucket tags
are not accepted. IFT answers are JSON-encoded string lists; DCP answers are
native string lists; boolean/degree answers are strings.

Predictions use unique string qid and raw string response, for example:

```json
{"qid":"cf32f75bc3d9","response":"true"}
```

Prediction files are UTF-8 JSONL. The API also accepts a QID-to-text mapping or
an iterable of objects. Explicit prediction_field selects a historical field;
answer/gt fields are never guessed. Empty inputs, duplicate, unknown/unselected
QIDs and non-string responses are errors.

Evaluator reports contain per-task metrics, counts, coverage, QA hashes, data
version, package version and execution_backend. include_details adds per_question.
Unassessed metrics are null; no overall score is defined. Runner reports add
sampling completion, pinned dataset identity and durable output locations.
Preview report schemas 1.0/1.1 no longer describe these reports.
