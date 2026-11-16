"""Edge level verification scoring."""
from __future__ import annotations
from ... import backend
from ..parsing import parse_bool

def score(q: dict, raw, args) -> dict:
    gt = str(q["answer"]).lower()
    pred = parse_bool(raw)
    parse_ok = pred is not None
    s = backend.exact_match(gt, pred) if parse_ok else 0.0
    m = q.get("meta", {})
    return {
        "score": s,
        "parse_ok": parse_ok,
        "pred_parsed": pred,
        "gt": gt,
        "stratum": {
            "label": m.get("label"),
            "neg_subtype": m.get("neg_subtype"),
        },
    }
