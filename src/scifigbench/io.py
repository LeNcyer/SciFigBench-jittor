"""Validated UTF-8 JSONL input and packaged JSON Schemas."""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from importlib.resources import files
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator

TASKS = (
    "dense_content_perception",
    "edge_level_verification",
    "node_level_degree_counting",
    "path_level_reachability",
    "information_flow_tracing",
)
PRED_FIELDS = ("response", "pred", "prediction", "answer", "model_output", "output")
COLORS = ("red", "orange", "yellow", "green", "blue", "violet")


@lru_cache(maxsize=3)
def schema(name: str) -> dict:
    return json.loads(
        files("scifigbench").joinpath("schemas", f"{name}.schema.json").read_text(encoding="utf-8")
    )


@lru_cache(maxsize=3)
def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(schema(name))


def validate_schema(value: dict, name: str, context: str) -> None:
    error = next(_validator(name).iter_errors(value), None)
    if error:
        location = ".".join(str(x) for x in error.absolute_path) or "<row>"
        raise ValueError(f"{context}: {location}: {error.message}")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("JSON number exceeds the finite floating-point range")
    return value


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_jsonl(path: str | Path, kind: str) -> list[dict]:
    path = Path(path)
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            context = f"{path}:{lineno}"
            try:
                row = json.loads(
                    line, parse_constant=_reject_constant,
                    parse_float=_finite_float, object_pairs_hook=_unique_keys,
                )
            except ValueError as exc:
                raise ValueError(f"{context}: {exc}") from exc
            validate_schema(row, kind, context)
            qid = row["qid"]
            if qid in seen:
                raise ValueError(f"{context}: duplicate qid {qid!r}")
            seen.add(qid)
            if kind == "qa":
                validate_qa_row(row, f"{context} qid={qid}")
            rows.append(row)
    return rows


def answer_list(q: dict) -> list[str]:
    value = q["answer"]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError("answer must encode a JSON list of strings") from exc
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError("answer must be a list of strings")
    return value


def validate_qa_row(q: dict, context: str) -> None:
    try:
        rel = PurePosixPath(q["image_path"])
        if rel.is_absolute() or ".." in rel.parts or "\\" in str(rel) or ":" in str(rel):
            raise ValueError("image_path must be relative, use '/', and contain no '..'")
        width, height = q["image_size"]
        boxes = q.get("boxes_xyxy", []) + [
            q[k] for k in ("red_box_xyxy", "blue_box_xyxy") if k in q
        ]
        for x1, y1, x2, y2 in boxes:
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("box must have positive area and lie within image_size")
        task = q["task"]
        if task in ("dense_content_perception", "information_flow_tracing"):
            gt = answer_list(q)
            if any(not text.strip() for text in gt):
                raise ValueError("ground-truth labels cannot be empty/whitespace strings")
        if task == "dense_content_perception":
            if len(q["boxes_xyxy"]) != len(q["colors"]) or len(gt) != len(q["colors"]):
                raise ValueError("boxes, colors, and answer lengths must agree")
            if q["colors"] != sorted(q["colors"], key=COLORS.index):
                raise ValueError("colors must follow red/orange/yellow/green/blue/violet order")
            if q["subtask"] == "single" and (len(gt) != 1 or q["colors"] != ["red"]):
                raise ValueError("single requires exactly one red box and one answer")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context}: {exc}") from exc


def load_qa(path: str | Path) -> list[dict]:
    rows = read_jsonl(path, "qa")
    if not rows:
        raise ValueError(f"{path}: QA file is empty")
    versions = {q.get("dataset_version", "unspecified") for q in rows}
    if len(versions) != 1:
        raise ValueError(f"{path}: dataset_version must be consistent across all QA rows")
    return rows


def load_predictions(path: str | Path) -> tuple[dict, str | None]:
    rows = read_jsonl(path, "prediction")
    selected = None
    result = {}
    for row in rows:
        present = [field for field in PRED_FIELDS if field in row]
        if len(present) != 1:
            raise ValueError(f"{path}: qid={row['qid']}: use exactly one prediction field")
        field = present[0]
        if selected is not None and field != selected:
            raise ValueError(f"{path}: qid={row['qid']}: inconsistent prediction field")
        selected = field
        result[row["qid"]] = row[field]
    return result, selected


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
