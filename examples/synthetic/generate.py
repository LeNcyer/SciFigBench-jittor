"""Generate original synthetic figures and questions; no benchmark data needed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TASKS = (
    "dense_content_perception", "edge_level_verification", "node_level_degree_counting",
    "path_level_reachability", "information_flow_tracing",
)
BOXES = {
    "Input": [60, 220, 200, 280], "Encoder": [300, 220, 460, 280],
    "Output": [570, 220, 730, 280], "SkipTop": [330, 60, 450, 120],
    "SkipBottom": [330, 400, 450, 460], "Isolated": [740, 400, 910, 460],
}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(root: Path) -> None:
    image_dir = root / "images" / "flow"
    image_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (960, 520), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    for name, box in BOXES.items():
        draw.rounded_rectangle(box, radius=8, fill=(239, 245, 250), outline=(30, 40, 55), width=2)
        label = "Skip" if name.startswith("Skip") else name
        x1, y1, x2, y2 = box
        draw.text(((x1 + x2) / 2, (y1 + y2) / 2), label, font=font, fill=(20, 30, 40), anchor="mm")
    for start, end in [((200, 250), (300, 250)), ((460, 250), (570, 250)),
                       ((390, 120), (390, 220)), ((390, 400), (390, 280))]:
        draw.line([start, end], fill=(30, 40, 55), width=3)
        x, y = end
        if start[0] != x:
            arrow = [(x, y), (x - 12, y - 7), (x - 12, y + 7)]
        elif start[1] < y:
            arrow = [(x, y), (x - 7, y - 12), (x + 7, y - 12)]
        else:
            arrow = [(x, y), (x - 7, y + 12), (x + 7, y + 12)]
        draw.polygon(arrow, fill=(30, 40, 55))
    image.save(image_dir / "SYN000001.png")

    def row(task, suffix, prompt, answer, **extra):
        return {
            "qid": f"SYN-{suffix}", "task": task, "sample_id": "SYN000001",
            "dataset_version": "synthetic-v1", "image_path": "images/flow/SYN000001.png",
            "image_size": [960, 520], "prompt": prompt, "answer": answer, **extra,
        }

    dense, edge, degree, path, flow = TASKS
    qa = {
        dense: [
            row(dense, "dense-single", "Read the text in the red box. Return only a JSON list of strings.",
                ["Encoder"], subtask="single", boxes_xyxy=[BOXES["Encoder"]], colors=["red"],
                meta={"K": 1, "area_bucket": "synthetic", "len_bucket": "synthetic"}),
            row(dense, "dense-path", "Read the boxes in red, orange, yellow order. Return only a JSON list of strings.",
                ["Input", "Encoder", "Output"], subtask="flow_path",
                boxes_xyxy=[BOXES[k] for k in ("Input", "Encoder", "Output")],
                colors=["red", "orange", "yellow"], meta={"K": 3}),
        ],
        edge: [
            row(edge, "edge-positive", "Is there a direct edge from the red node to the blue node? Follow arrow direction. Return true or false.",
                "true", red_box_xyxy=BOXES["Input"], blue_box_xyxy=BOXES["Encoder"],
                meta={"label": "positive"}),
            row(edge, "edge-reverse", "Is there a direct edge from the red node to the blue node? Follow arrow direction. Return true or false.",
                "false", red_box_xyxy=BOXES["Encoder"], blue_box_xyxy=BOXES["Input"],
                meta={"label": "negative", "neg_subtype": "reverse"}),
        ],
        degree: [
            row(degree, "degree-in", "How many direct edges enter the red node? Count edges, including parallel edges separately. Return only an integer.",
                "3", red_box_xyxy=BOXES["Encoder"], meta={"variant": "in", "degree_bucket": "3"}),
            row(degree, "degree-out", "How many direct edges leave the red node? Return only an integer.",
                "1", red_box_xyxy=BOXES["Encoder"], meta={"variant": "out", "degree_bucket": "1"}),
        ],
        path: [
            row(path, "path-positive", "Can the red node reach the blue node by following one or more directed edges? Return true or false.",
                "true", red_box_xyxy=BOXES["Input"], blue_box_xyxy=BOXES["Output"],
                meta={"label": "positive", "path_len_bucket": "2"}),
            row(path, "path-reverse", "Can the red node reach the blue node by following one or more directed edges? Return true or false.",
                "false", red_box_xyxy=BOXES["Output"], blue_box_xyxy=BOXES["Input"],
                meta={"label": "negative", "neg_subtype": "reverse_path"}),
        ],
        flow: [
            row(flow, "flow-in", "List all direct incoming neighbor texts of the unique Encoder node. Include each distinct neighbor node, retaining duplicate texts. Return a JSON list; order does not matter.",
                ["Input", "Skip", "Skip"], meta={"variant": "in", "target_type": "both"}),
            row(flow, "flow-out", "List direct outgoing neighbor texts of the unique Encoder node. Return a JSON list, or [] if none.",
                json.dumps(["Output"]), meta={"variant": "out", "target_type": "both"}),
            row(flow, "flow-empty", "List direct incoming neighbor texts of the unique Isolated node. Return a JSON list, or [] if none.",
                "[]", meta={"variant": "in", "target_type": "in_only"}),
        ],
    }
    # The partial fixture intentionally includes missing and invalid predictions.
    mixed = {
        dense: [
            {"qid": "SYN-dense-single", "response": ["Encoder"]},
            {"qid": "SYN-dense-path", "response": ["Output", "Encoder", "Input"]},
        ],
        edge: [{"qid": "SYN-edge-positive", "response": "true"}],
        degree: [
            {"qid": "SYN-degree-in", "response": "3"},
            {"qid": "SYN-degree-out", "response": "0"},
        ],
        path: [
            {"qid": "SYN-path-positive", "response": True},
            {"qid": "SYN-path-reverse", "response": "uncertain"},
        ],
        flow: [
            {"qid": "SYN-flow-in", "response": ["Skip", "Input", "Skip"]},
            {"qid": "SYN-flow-empty", "response": "uncertain"},
        ],
    }
    expected = {
        # Input/Output each have ANLS 0.5, so the swapped triple scores 2/3.
        dense: {"score": 5 / 6, "coverage": 1.0, "n_missing": 0, "n_parse_fail": 0},
        edge: {"score": 0.5, "coverage": 0.5, "n_missing": 1, "n_parse_fail": 0},
        degree: {"score": 0.5, "coverage": 1.0, "n_missing": 0, "n_parse_fail": 0},
        path: {"score": 0.5, "coverage": 1.0, "n_missing": 0, "n_parse_fail": 1},
        flow: {"score": 1 / 3, "coverage": 2 / 3, "n_missing": 1, "n_parse_fail": 1},
    }
    for task, rows in qa.items():
        write_rows(root / "qa" / f"{task}.jsonl", rows)
        write_rows(root / "predictions" / "correct" / f"{task}.jsonl", [
            {"qid": q["qid"], "response": q["answer"]} for q in rows
        ])
        write_rows(root / "predictions" / "mixed" / f"{task}.jsonl", mixed[task])
    with (root / "expected.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(expected, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    generate(parser.parse_args().out_dir)
