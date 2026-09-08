"""Evaluate one task using all its QA rows as the denominator."""
from __future__ import annotations

import json
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .io import TASKS, load_predictions, load_qa, sha256_file, validate_schema, write_json
from .tasks import aggregate, score_question


def evaluate(
    qa_path: str | Path, pred_path: str | Path, *, task: str,
    report_dir: str | Path | None = None, normalize: bool = True,
    overwrite: bool = False,
) -> tuple[dict, list[dict]]:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    all_qa = load_qa(qa_path)
    qa = [q for q in all_qa if q["task"] == task]
    if not qa:
        raise ValueError(f"{qa_path}: no QA rows for task {task}")
    predictions, field = load_predictions(pred_path)
    unknown = predictions.keys() - {q["qid"] for q in qa}
    if unknown:
        raise ValueError(f"{pred_path}: unknown qid for selected task: {sorted(unknown)[0]!r}")
    destination = Path(report_dir) / task if report_dir is not None else None
    if destination is not None and destination.exists() and not overwrite:
        raise FileExistsError(f"{destination}: results already exist; use --overwrite")
    rows = [
        score_question(
            q, predictions.get(q["qid"]), missing=q["qid"] not in predictions,
            normalize=normalize,
        )
        for q in qa
    ]
    report = aggregate(task, rows)
    report.update({
        "toolkit_version": __version__, "schema_version": SCHEMA_VERSION,
        "dataset_version": qa[0].get("dataset_version", "unspecified"),
        "task": task, "n_questions": len(qa), "n_submitted": len(predictions),
        "n_missing": sum(r["status"] == "missing" for r in rows),
        "n_parse_fail": sum(r["status"] == "parse_failure" for r in rows),
        "coverage": len(predictions) / len(qa), "prediction_field": field,
        "latex_normalize": normalize,
        "input_sha256": {"qa": sha256_file(qa_path), "predictions": sha256_file(pred_path)},
    })
    validate_schema(report, "report", f"report for {task}")
    if destination is not None:
        outputs = [destination / "report.json", destination / f"per_question__{task}.jsonl"]
        inputs = {Path(qa_path).resolve(), Path(pred_path).resolve()}
        if any(path.resolve() in inputs for path in outputs):
            raise ValueError("report output would overwrite an input file")
        destination.mkdir(parents=True, exist_ok=True)
        write_json(outputs[0], report)
        outputs[1].write_text(
            "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    return report, rows
