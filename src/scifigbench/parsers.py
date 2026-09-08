"""Extract typed predictions from raw model responses."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(chr(96) * 3 + r"(?:json)?\s*(.*?)\s*" + chr(96) * 3,
                    re.DOTALL | re.IGNORECASE)
_BOOL = re.compile(r"\b(true|false)\b", re.IGNORECASE)
_INT = re.compile(r"[-+]?\d+")


def parse_bool(raw) -> str | None:
    if raw is None or isinstance(raw, (list, dict)):
        return None
    match = _BOOL.search(str(raw))
    return match.group(1).lower() if match else None


def parse_int(raw) -> int | None:
    if raw is None or isinstance(raw, (bool, list, dict)):
        return None
    match = _INT.search(str(raw))
    return int(match.group()) if match else None


def parse_json_list(raw) -> list[str] | None:
    if isinstance(raw, list):
        return raw.copy() if all(isinstance(x, str) for x in raw) else None
    if not isinstance(raw, str):
        return None
    fence = _FENCE.search(raw)
    text = (fence.group(1) if fence else raw).strip()
    try:
        value = json.loads(text)
    except ValueError:
        start = text.find("[")
        if start < 0:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except ValueError:
            return None
    return value if isinstance(value, list) and all(isinstance(x, str) for x in value) else None
