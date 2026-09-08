"""Five task scorers and task-specific diagnostics."""
from __future__ import annotations

from collections import defaultdict

from .io import answer_list
from .metrics import anls, position_anls, set_anls
from .parsers import parse_bool, parse_int, parse_json_list


def _neighbor_bucket(n: int) -> str:
    return str(n) if n <= 1 else ("2-3" if n <= 3 else "4+")


def score_question(q: dict, raw, *, missing: bool, normalize: bool) -> dict:
    task, meta = q["task"], q.get("meta", {})
    set_score = None
    if task in ("edge_level_verification", "path_level_reachability"):
        gt, parsed = str(q["answer"]).lower(), parse_bool(raw)
        strata = {k: meta.get(k) for k in ("label", "neg_subtype")}
        if task == "path_level_reachability":
            strata["path_len_bucket"] = meta.get("path_len_bucket")
    elif task == "node_level_degree_counting":
        gt, parsed = int(q["answer"]), parse_int(raw)
        strata = {k: meta.get(k) for k in ("variant", "degree_bucket")}
    else:
        gt, parsed = answer_list(q), parse_json_list(raw)
        if task == "dense_content_perception":
            strata = {k: meta.get(k) for k in ("K", "area_bucket", "len_bucket")}
            strata["subtask"] = q["subtask"]
            strata["K"] = meta.get("K", len(q["colors"]))
        else:
            strata = {k: meta.get(k) for k in ("variant", "target_type")}
            strata["n_gt_neighbors"] = _neighbor_bucket(len(gt))

    parse_ok = parsed is not None and not missing
    if not parse_ok:
        score = 0.0
        if task == "dense_content_perception":
            set_score = 0.0
    elif task == "dense_content_perception":
        if q["subtask"] == "single":
            score = anls(gt[0], parsed[0] if parsed else "", normalize=normalize)
            set_score = score
        else:
            score = position_anls(gt, parsed, normalize=normalize)
            set_score = set_anls(gt, parsed, normalize=normalize)
    elif task == "information_flow_tracing":
        score = set_anls(gt, parsed, normalize=normalize)
    else:
        score = float(parsed == gt)
    result = {
        "qid": q["qid"], "task": task, "gt": gt, "pred_raw": raw,
        "pred_parsed": parsed, "score": score, "parse_ok": parse_ok,
        "status": "missing" if missing else ("ok" if parse_ok else "parse_failure"),
        "stratum": strata,
    }
    if set_score is not None:
        result["set_score"] = set_score
    return result


def _mean(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def aggregate(task: str, rows: list[dict]) -> dict:
    metric = {
        "dense_content_perception": "ANLS",
        "information_flow_tracing": "set-ANLS*",
    }.get(task, "accuracy")
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row["stratum"].items():
            if key in ("area_bucket", "len_bucket") and row["stratum"].get("subtask") != "single":
                continue
            if value is None:
                if key in ("neg_subtype", "path_len_bucket"):
                    value = "n/a"
                else:
                    continue
            buckets[f"by_{key}"][str(value)].append(row["score"])
    report = {
        "primary_metric": {"name": metric, "value": _mean([r["score"] for r in rows])},
        "subgroups": {
            name: {key: {"n": len(values), "mean": _mean(values)}
                   for key, values in sorted(groups.items())}
            for name, groups in sorted(buckets.items())
        },
    }
    extras = {}
    if task == "dense_content_perception":
        extras["set-ANLS*"] = {"value": _mean([r["set_score"] for r in rows])}
    if task == "edge_level_verification":
        selectors = {
            "hard_neg_accuracy": lambda s: s.get("label") == "negative"
            and s.get("neg_subtype") not in (None, "easy"),
            "reverse_neg_accuracy": lambda s: s.get("neg_subtype") == "reverse",
        }
    elif task == "path_level_reachability":
        selectors = {
            "reverse_path_neg_accuracy": lambda s: s.get("neg_subtype") == "reverse_path",
        }
    else:
        selectors = {}
    for name, select in selectors.items():
        scores = [r["score"] for r in rows if select(r["stratum"])]
        extras[name] = {"n": len(scores), "value": _mean(scores)}
    if extras:
        report["additional_metrics"] = extras
    return report
