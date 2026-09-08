import json

import pytest

from scifigbench.evaluate import evaluate
from scifigbench.io import TASKS, load_qa, schema
from jsonschema import Draft202012Validator


@pytest.mark.parametrize("task", TASKS)
def test_correct_and_mixed_fixtures(examples, task, tmp_path):
    expected = json.loads((examples / "expected.json").read_text())[task]
    for variant in ("correct", "mixed"):
        report, rows = evaluate(
            examples / "qa" / f"{task}.jsonl",
            examples / "predictions" / variant / f"{task}.jsonl",
            task=task, report_dir=tmp_path / variant,
        )
        assert report["primary_metric"]["value"] == pytest.approx(
            1 if variant == "correct" else expected["score"]
        )
        assert report["coverage"] == pytest.approx(1 if variant == "correct" else expected["coverage"])
        assert report["dataset_version"] == "synthetic-v1"
        assert len(rows) == report["n_questions"]
        assert report["n_missing"] == (0 if variant == "correct" else expected["n_missing"])
        assert report["n_parse_fail"] == (0 if variant == "correct" else expected["n_parse_fail"])
        Draft202012Validator(schema("report")).validate(report)
        assert (tmp_path / variant / task / "report.json").exists()


@pytest.mark.parametrize("task", TASKS)
def test_empty_predictions_score_zero(examples, task, write_rows):
    report, rows = evaluate(
        examples / "qa" / f"{task}.jsonl", write_rows("empty.jsonl", []), task=task
    )
    assert report["primary_metric"]["value"] == report["coverage"] == 0
    assert report["n_missing"] == len(rows)
    assert report["n_parse_fail"] == 0
    assert all(r["score"] == 0 and r["status"] == "missing" for r in rows)
    assert all(g["mean"] == 0 for bucket in report["subgroups"].values() for g in bucket.values())


@pytest.mark.parametrize("raw,score,fail", [("[]", 1, 0), ([], 1, 0), ("bad", 0, 1), (None, 0, 1)])
def test_empty_answer_only_rewards_valid_empty_list(examples, write_rows, raw, score, fail):
    task = "information_flow_tracing"
    q = load_qa(examples / "qa" / f"{task}.jsonl")[-1]
    report, _ = evaluate(
        write_rows("qa.jsonl", [q]),
        write_rows("pred.jsonl", [{"qid": q["qid"], "response": raw}]), task=task,
    )
    assert report["primary_metric"]["value"] == score
    assert report["n_parse_fail"] == fail


def test_missing_negative_stays_in_subgroup(examples):
    task = "edge_level_verification"
    report, _ = evaluate(
        examples / "qa" / f"{task}.jsonl",
        examples / "predictions" / "mixed" / f"{task}.jsonl", task=task,
    )
    assert report["subgroups"]["by_label"]["negative"] == {"n": 1, "mean": 0}


@pytest.mark.parametrize("problem", ["duplicate_qa", "duplicate_pred", "unknown", "empty_qa", "wrong_task"])
def test_bad_ids_and_empty_qa(examples, write_rows, problem):
    task = "edge_level_verification"
    qa = load_qa(examples / "qa" / f"{task}.jsonl")
    pred = [{"qid": qa[0]["qid"], "response": "true"}]
    if problem == "duplicate_qa":
        qa += [qa[0]]
    elif problem == "duplicate_pred":
        pred += [pred[0]]
    elif problem == "unknown":
        pred[0]["qid"] = "SYN-unknown"
    elif problem == "empty_qa":
        qa = []
    else:
        task = "information_flow_tracing"
    with pytest.raises(ValueError):
        evaluate(write_rows("q.jsonl", qa), write_rows("p.jsonl", pred), task=task)


def test_report_overwrite_and_hashes(examples, tmp_path, write_rows):
    task = "edge_level_verification"
    qa = examples / "qa" / f"{task}.jsonl"
    pred = examples / "predictions" / "correct" / f"{task}.jsonl"
    original, _ = evaluate(qa, pred, task=task, report_dir=tmp_path)
    with pytest.raises(FileExistsError):
        evaluate(qa, pred, task=task, report_dir=tmp_path)
    changed, _ = evaluate(
        qa, write_rows("empty.jsonl", []), task=task, report_dir=tmp_path, overwrite=True,
    )
    assert original["input_sha256"]["qa"] == changed["input_sha256"]["qa"]
    assert original["input_sha256"]["predictions"] != changed["input_sha256"]["predictions"]
    assert changed["primary_metric"]["value"] == 0


def test_unspecified_version_and_dense_single_compatibility(examples, write_rows):
    task = "dense_content_perception"
    q = load_qa(examples / "qa" / f"{task}.jsonl")[0]
    q.pop("dataset_version")
    report, _ = evaluate(
        write_rows("qa.jsonl", [q]),
        write_rows("pred.jsonl", [{"qid": q["qid"], "response": ["Encoder", "extra"]}]),
        task=task,
    )
    # Deliberately preserved legacy single-box semantics: first item only.
    assert report["primary_metric"]["value"] == 1
    assert report["dataset_version"] == "unspecified"
