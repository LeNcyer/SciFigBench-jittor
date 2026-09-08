"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .evaluate import evaluate
from .io import TASKS, load_qa
from .render import image_path, render_qa


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="scifigbench", description=__doc__)
    result.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = result.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="validate a QA JSONL file without loading images")
    check.add_argument("--qa", required=True)
    render = sub.add_parser("render", help="render model-facing images")
    render.add_argument("--qa", required=True)
    render.add_argument("--data-root", required=True)
    render.add_argument("--out-dir", required=True)
    render.add_argument("--overwrite", action="store_true")
    score = sub.add_parser("evaluate", help="evaluate predictions for one selected task")
    score.add_argument("--task", choices=TASKS, required=True)
    score.add_argument("--qa", required=True)
    score.add_argument("--pred", required=True)
    score.add_argument("--report-dir", required=True)
    score.add_argument("--no-latex-normalize", action="store_true")
    score.add_argument("--overwrite", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            rows = load_qa(args.qa)
            print(json.dumps({
                "n_questions": len(rows), "tasks": sorted({q["task"] for q in rows}),
                "dataset_version": rows[0].get("dataset_version", "unspecified"),
            }))
        elif args.command == "render":
            rows = load_qa(args.qa)
            destination = Path(args.out_dir)
            outputs = [destination / f"{q['task']}__{q['qid']}.png" for q in rows]
            sources = {image_path(q, args.data_root) for q in rows}
            for output in outputs:
                if output.resolve() in sources:
                    raise ValueError(f"{output}: output would overwrite a source image")
                if output.exists() and not args.overwrite:
                    raise FileExistsError(f"{output}: already exists; use --overwrite")
            destination.mkdir(parents=True, exist_ok=True)
            for q, output in zip(rows, outputs):
                try:
                    with render_qa(q, args.data_root) as rendered:
                        rendered.save(output)
                except (ValueError, OSError) as exc:
                    raise ValueError(f"qid={q['qid']}: {exc}") from exc
            print(f"Rendered {len(rows)} images to {destination}")
        else:
            report, _ = evaluate(
                args.qa, args.pred, task=args.task, report_dir=args.report_dir,
                normalize=not args.no_latex_normalize, overwrite=args.overwrite,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
