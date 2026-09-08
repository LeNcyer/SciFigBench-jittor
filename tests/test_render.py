from copy import deepcopy

import pytest
from PIL import Image, ImageChops

from scifigbench.io import load_qa
from scifigbench.render import RGB, box_thickness, render_qa


def test_outward_color_and_content_preservation(examples, monkeypatch, tmp_path):
    q = load_qa(examples / "qa" / "edge_level_verification.jsonl")[0]
    monkeypatch.chdir(tmp_path)
    output = render_qa(q, examples)
    with Image.open(examples / q["image_path"]) as source:
        for key, color in (("red_box_xyxy", "red"), ("blue_box_xyxy", "blue")):
            x1, y1, x2, y2 = q[key]
            thickness = box_thickness(*source.size)
            assert output.getpixel((x1 - thickness, (y1 + y2) // 2)) == RGB[color]
            assert ImageChops.difference(source.crop((x1 + 1, y1 + 1, x2 - 1, y2 - 1)),
                                         output.crop((x1 + 1, y1 + 1, x2 - 1, y2 - 1))).getbbox() is None


def test_flow_is_unmodified_and_multibox_colors(examples):
    q = load_qa(examples / "qa" / "information_flow_tracing.jsonl")[0]
    with Image.open(examples / q["image_path"]) as source:
        assert ImageChops.difference(source.convert("RGB"), render_qa(q, examples)).getbbox() is None
    q = load_qa(examples / "qa" / "dense_content_perception.jsonl")[1]
    result = render_qa(q, examples)
    for box, color in zip(q["boxes_xyxy"], q["colors"]):
        assert result.getpixel((box[0] - 5, box[1] + 20)) == RGB[color]


def test_image_size_mismatch_and_clipping(examples):
    q = load_qa(examples / "qa" / "node_level_degree_counting.jsonl")[0]
    bad = deepcopy(q)
    bad["image_size"] = [1000, 600]
    with pytest.raises(ValueError, match="image size"):
        render_qa(bad, examples)
    q["red_box_xyxy"] = [0, 0, 40, 40]
    assert render_qa(q, examples).getpixel((0, 0)) == RGB["red"]
