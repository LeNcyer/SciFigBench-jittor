"""Render and score the synthetic fixture with the installed public API."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def run_demo(data_root: Path, out_dir: Path, overwrite: bool = False) -> None:
    from scifigbench import load_qa, render

    expected = json.loads((data_root / "expected.json").read_text(encoding="utf-8"))
    rows = load_qa(data_root)
    outputs = [out_dir / "rendered" / q["task"] / f"{q['task']}__{q['qid']}.png" for q in rows]
    reports = [out_dir / variant / "report.json" for variant in ("correct", "mixed")]
    if not overwrite and any(p.exists() for p in outputs + reports):
        raise FileExistsError("Demo outputs already exist; use --overwrite")
    for q, path in zip(rows, outputs):
        path.parent.mkdir(parents=True, exist_ok=True)
        with render(q, data_root=data_root) as image:
            image.save(path)
    for variant, path in zip(("correct", "mixed"), reports):
        command = [sys.executable, "-m", "scifigbench", "evaluate", "--task", "all",
                   "--data-root", str(data_root), "--predictions",
                   str(data_root / "predictions" / f"{variant}.jsonl"),
                   "--include-details", "--output", str(path)]
        if variant == "mixed":
            command.append("--allow-partial")
        subprocess.run(command, check=True, capture_output=True)
        report = json.loads(path.read_text(encoding="utf-8"))
        for task, wanted in expected.items():
            result = report["tasks"][task]
            target = ({"score": 1.0, "coverage": 1.0, "n_missing": 0, "n_parse_fail": 0}
                      if variant == "correct" else wanted)
            actual = {"score": result["primary_metric"]["value"],
                      "coverage": result["coverage"], "n_missing": result["n_missing_pred"],
                      "n_parse_fail": result["n_parse_fail"]}
            for key, value in target.items():
                if not math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-12):
                    raise AssertionError(f"{task}/{variant} {key}: {actual[key]} != {value}")
            print(f"{task}/{variant}: score={actual['score']:.6f}, coverage={actual['coverage']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/synthetic"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        run_demo(args.data_root.resolve(), args.out_dir.resolve(), args.overwrite)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        raise SystemExit(exc.returncode)
