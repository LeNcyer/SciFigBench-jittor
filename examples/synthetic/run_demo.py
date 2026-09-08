"""Run the installed CLI on every synthetic task and check expected outcomes."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def run_demo(data_root: Path, out_dir: Path, overwrite: bool = False) -> None:
    expected = json.loads((data_root / "expected.json").read_text(encoding="utf-8"))
    cli = [sys.executable, "-m", "scifigbench"]
    for task, wanted in expected.items():
        qa = data_root / "qa" / f"{task}.jsonl"
        subprocess.run(cli + ["validate", "--qa", str(qa)], check=True, capture_output=True)
        subprocess.run(cli + [
            "render", "--qa", str(qa), "--data-root", str(data_root),
            "--out-dir", str(out_dir / "rendered" / task),
        ] + (["--overwrite"] if overwrite else []), check=True, capture_output=True)
        for variant in ("correct", "mixed"):
            subprocess.run(cli + [
                "evaluate", "--task", task, "--qa", str(qa),
                "--pred", str(data_root / "predictions" / variant / f"{task}.jsonl"),
                "--report-dir", str(out_dir / variant),
            ] + (["--overwrite"] if overwrite else []), check=True, capture_output=True)
            report = json.loads((out_dir / variant / task / "report.json").read_text(encoding="utf-8"))
            target = {"score": 1.0, "coverage": 1.0, "n_missing": 0, "n_parse_fail": 0} if variant == "correct" else wanted
            actual = {k: report[k] for k in ("coverage", "n_missing", "n_parse_fail")}
            actual["score"] = report["primary_metric"]["value"]
            for key, value in target.items():
                if not math.isclose(actual[key], value, abs_tol=1e-12):
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
