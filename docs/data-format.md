# Data format

JSONL is UTF-8 (an initial BOM is accepted), with one object per nonblank line.
Schemas are bundled under the installed scifigbench/schemas directory and can
be read using scifigbench.io.schema("qa"), schema("prediction"), or schema("report").

## QA

Required common fields: qid, task, sample_id, image_path, image_size, prompt,
answer. Optional common fields: meta (object), dataset_version (string).
Additional metadata fields are accepted for compatibility.

IDs must start with an ASCII letter/digit and use only letters, digits,
underscores, dots or hyphens. Image paths use forward slashes, remain relative to
--data-root, and cannot contain parent traversal. Box coordinates must define a
positive-area rectangle within image_size. Rendering also checks actual dimensions.

Task-specific fields:

- Dense: subtask (single or flow_path), boxes_xyxy, colors, and an array answer.
  Box, color and answer counts must agree, and colors follow the fixed palette order.
- Edge/reachability: red_box_xyxy, blue_box_xyxy; answer is true/false or the
  lowercase string "true"/"false".
- Degree: red_box_xyxy; answer is a nonnegative integer or decimal integer string.
- Information flow: answer is a list of strings or a JSON-encoded list string,
  including [] / "[]". No box is required.

Ground-truth labels cannot be blank. Metadata for subgroup reports follows the
source task format (label, neg_subtype, variant, degree_bucket, subtask, K,
area_bucket, len_bucket, target_type and path_len_bucket). Subtask and K are
available directly/from the visual fields if needed. Empty subgroup metadata is
omitted; negative/path subtype nulls are reported as n/a.

All rows in a QA file must use the same dataset_version. If omitted everywhere,
the report uses unspecified. Synthetic examples explicitly use synthetic-v1.
Counts are always calculated from input rows, never assumed from a benchmark release.

## Predictions

~~~json
{"qid": "SYN-edge-positive", "response": "true"}
~~~

Choose exactly one response field per row and the same field throughout the
file: response, pred, prediction, answer, model_output, or output.
Raw values can be strings, string arrays, booleans, finite numbers or null.
Strings that cannot be parsed as the selected answer type score zero. Invalid
row structure is a file error, not a scored parse failure.

Native lists and encoded lists are both accepted:

~~~json
{"qid": "SYN-dense-single", "response": ["Encoder"]}
~~~

An empty prediction file is valid and scores zero on every question. Duplicated
prediction IDs or predictions outside the selected task are errors. A mixed-task
QA file can be selected with --task, but the prediction file must contain only
IDs from that task.

## Reports

Each task writes report.json and per_question__<task>.jsonl into
<report-dir>/<task>/. An existing task destination is refused unless --overwrite
is specified. Overwrite replaces only those two output files.

Reports contain toolkit_version, schema_version, dataset_version, task,
n_questions, n_submitted, n_missing, n_parse_fail, coverage, primary_metric,
subgroups, input_sha256, prediction_field and latex_normalize. Selected tasks
also expose additional_metrics. Input hashes cover the original file bytes,
including newline/BOM choices. Versions and SHA-256 hashes identify an evaluation;
no absolute source paths or environment secrets are recorded.

Per-question records include qid, task, gt, pred_raw, pred_parsed, score, parse_ok,
status and stratum. Status is ok, missing or parse_failure. Missing questions
have parse_ok=false but are counted separately from n_parse_fail.
