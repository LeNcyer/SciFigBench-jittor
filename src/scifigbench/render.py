"""Render only the visual references specified by a question."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .io import validate_qa_row, validate_schema

RGB = {
    "red": (255, 0, 0), "orange": (255, 140, 0), "yellow": (250, 200, 0),
    "green": (0, 170, 0), "blue": (0, 90, 255), "violet": (160, 32, 240),
}


def image_path(q: dict, data_root: str | Path) -> Path:
    root = Path(data_root).resolve()
    path = (root / q["image_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"qid={q['qid']}: image_path escapes data-root")
    return path


def box_thickness(width: int, height: int) -> int:
    return max(5, int(round(min(width, height) / 130)))


def render_qa(q: dict, data_root: str | Path) -> Image.Image:
    validate_schema(q, "qa", "render QA")
    validate_qa_row(q, f"qid={q['qid']}")
    path = image_path(q, data_root)
    with Image.open(path) as source:
        if list(source.size) != q["image_size"]:
            raise ValueError(f"qid={q['qid']}: actual image size differs from image_size")
        canvas = source.convert("RGB")
    task = q["task"]
    if task == "information_flow_tracing":
        return canvas
    if task == "dense_content_perception":
        boxes = zip(q["boxes_xyxy"], q["colors"])
    else:
        boxes = [(q["red_box_xyxy"], "red")]
        if task in ("edge_level_verification", "path_level_reachability"):
            boxes.append((q["blue_box_xyxy"], "blue"))
    width, height = canvas.size
    thickness = box_thickness(width, height)
    draw = ImageDraw.Draw(canvas)
    for box, color in boxes:
        x1, y1, x2, y2 = box
        outward = [
            max(0, x1 - thickness), max(0, y1 - thickness),
            min(width, x2 + thickness), min(height, y2 + thickness),
        ]
        draw.rectangle(outward, outline=RGB[color], width=thickness)
    return canvas
