import pytest
from jsonschema import Draft202012Validator

from scifigbench.io import PRED_FIELDS, load_predictions, load_qa, schema
from scifigbench.parsers import parse_bool, parse_int, parse_json_list


@pytest.mark.parametrize("name", ["qa", "prediction", "report"])
def test_schemas_are_valid(name):
    Draft202012Validator.check_schema(schema(name))


@pytest.mark.parametrize("field", PRED_FIELDS)
def test_prediction_aliases(write_rows, field):
    result, selected = load_predictions(write_rows("p.jsonl", [{"qid": "SYN-1", field: ["X"]}]))
    assert result == {"SYN-1": ["X"]}
    assert selected == field


@pytest.mark.parametrize("rows", [
    [{"qid": "SYN-1", "response": "true"}, {"qid": "SYN-2", "pred": "false"}],
    [{"qid": "SYN-1", "response": "true", "answer": "false"}],
    [{"response": "true"}], [{"qid": "SYN-1"}],
])
def test_prediction_format_errors(rows, write_rows):
    with pytest.raises(ValueError):
        load_predictions(write_rows("pred.jsonl", rows))


@pytest.mark.parametrize("raw,expected", [
    (["x"], ["x"]), ('["x"]', ["x"]), ('prefix ["a]b", "x"] suffix', ["a]b", "x"]),
    ('[1]', None), ("invalid", None), ([], []), ('["unfinished"', None),
])
def test_list_parser(raw, expected):
    assert parse_json_list(raw) == expected


def test_fences_and_legacy_scalar_extraction():
    fence = chr(96) * 3
    assert parse_json_list(fence + 'json\n["x"]\n' + fence) == ["x"]
    assert parse_bool("Answer: TRUE.") == "true"
    assert parse_bool("untrue") is None
    assert parse_int("There are 12 edges.") == 12
    assert parse_int(True) is None


@pytest.mark.parametrize("problem", ["missing_key", "bad_box", "bad_colors", "bad_gt", "path", "boolean_count"])
def test_invalid_qa(examples, write_rows, problem):
    task = "dense_content_perception" if problem != "boolean_count" else "node_level_degree_counting"
    q = load_qa(examples / "qa" / f"{task}.jsonl")[0]
    if problem == "missing_key":
        q.pop("prompt")
    elif problem == "bad_box":
        q["boxes_xyxy"][0] = [20, 20, 10, 30]
    elif problem == "bad_colors":
        q["colors"] = ["blue"]
    elif problem == "bad_gt":
        q["answer"] = [1]
    elif problem == "path":
        q["image_path"] = "../outside.png"
    else:
        q["answer"] = True
    with pytest.raises(ValueError, match="qa.jsonl:1"):
        load_qa(write_rows("qa.jsonl", [q]))


@pytest.mark.parametrize("line", [
    '{"qid": "SYN-1", "response": NaN}',
    '{"qid": "SYN-1", "response": 1e999}',
    '{"qid": "SYN-1", "response": "true", "response": "false"}',
])
def test_line_numbers_and_nonfinite(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + line + "\n")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_predictions(path)
