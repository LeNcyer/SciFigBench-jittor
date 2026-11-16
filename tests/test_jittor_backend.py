"""Real CPU operators and differential checks against the frozen official release."""
import copy
import json
import random
import sys

import pytest

from scifigbench import backend
from scifigbench.evaluation import evaluate, score_question
from scifigbench.scoring import _latex_norm as actual
from reference_upstream.scoring import _latex_norm as reference
from reference_upstream.evaluation import evaluate as reference_evaluate
from reference_upstream.evaluation import score_question as reference_score

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Real Jittor scoring requires Linux/WSL2")


def compare(left, right):
    if isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in right:
            compare(left[key], right[key])
    elif isinstance(right, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare(a, b)
    elif isinstance(right, float):
        assert left == pytest.approx(right, abs=1e-12, rel=0)
    else:
        assert left == right


def test_real_operator_precision_flags_and_large_integer():
    with backend.cpu_runtime() as jt:
        before = (jt.flags.use_cuda, jt.flags.auto_convert_64_to_32, jt.flags.amp_reg)
        distances = backend.edit_distances(jt, [(["a", "b"], ["a"]), ([], ["x"]), ([], [])])
        assert isinstance(distances, jt.Var)
        assert distances.numpy().tolist() == [1, 1, 0]
        scores = backend.similarities(jt, [(["a", "b", "c"], ["a", "b"])], threshold=0.5)
        assert scores.dtype == "float64"
        assert scores.item() == pytest.approx(2 / 3, abs=1e-12, rel=0)
        assert jt.flags.use_cuda == 0 and jt.flags.no_grad == 1
        with backend.cpu_runtime():
            assert jt.flags.auto_convert_64_to_32 == 0
        assert before == (jt.flags.use_cuda, jt.flags.auto_convert_64_to_32, jt.flags.amp_reg)
    assert backend.exact_match(10**100 + 1, 10**100) == 0
    assert backend.exact_match(10**100 + 1, 10**100 + 1) == 1
    assert backend.mean([1, 0, 0]) == pytest.approx(1 / 3, abs=1e-12, rel=0)
    assert backend.ratio(1, 3) == pytest.approx(1 / 3, abs=1e-12, rel=0)
    assert backend.metadata()["assignment_solver"]["name"] == "scipy"


@pytest.mark.parametrize("normalize", [True, False])
def test_random_and_empty_padded_metrics_match_official(normalize):
    rng = random.Random(20261114)
    labels = ["A", "B", "abc", "abd", "xbc", "", r"\quad", "x^2", "x^{2}", "∝", "```literal```"]
    cases = [([], []), ([], [""]), (["A"], ["A", ""]), ([r"\quad"], []),
             (["A", "A"], ["A"]), (["A", "B"], ["B", "A"])]
    cases += [([rng.choice(labels) for _ in range(rng.randrange(5))],
               [rng.choice(labels) for _ in range(rng.randrange(5))]) for _ in range(30)]
    for a, b in cases:
        for name in ("position_anls", "set_anls"):
            assert getattr(actual, name)(a, b, normalize=normalize) == pytest.approx(
                getattr(reference, name)(a, b, normalize=normalize), abs=1e-12, rel=0)
    for a in labels:
        for b in labels:
            assert actual.anls(a, b, normalize=normalize) == pytest.approx(
                reference.anls(a, b, normalize=normalize), abs=1e-12, rel=0)


@pytest.mark.parametrize("normalize", [True, False])
def test_all_task_questions_match_official(dataset, normalize):
    responses = ["invalid", "[]", '[""]', '["A","B",""]', '["Same","Same"]',
                 '["```literal```",2,null]', '```json\n["Model"]\n```', "true", "false",
                 "3.9", "9" * 5000, "[" * 1200 + "]" * 1200]
    for qa in dataset[1]:
        for response in responses:
            compare(score_question(qa, response, no_latex_normalize=not normalize),
                    reference_score(qa, response, no_latex_normalize=not normalize))


@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_reports_and_details_match_official(dataset, predictions, partial, normalize):
    if partial:
        predictions = dict(list(predictions.items())[::2])
    options = dict(data_root=dataset[0], allow_partial=partial,
                   include_details=True, no_latex_normalize=not normalize)
    report = evaluate(predictions, **options)
    expected = reference_evaluate(predictions, **options)
    assert report.pop("execution_backend")["dtype"] == "float64"
    report.pop("package_version")
    expected.pop("package_version")
    compare(report, expected)


def test_synthetic_reports_match_reference_and_committed_outputs():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples/synthetic"
    for variant in ("correct", "mixed"):
        options = dict(data_root=root, allow_partial=variant == "mixed", include_details=True)
        report = evaluate(root / "predictions" / f"{variant}.jsonl", **options)
        expected = reference_evaluate(root / "predictions" / f"{variant}.jsonl", **options)
        report_without_metadata = copy.deepcopy(report)
        report_without_metadata.pop("execution_backend")
        report_without_metadata.pop("package_version")
        expected.pop("package_version")
        compare(report_without_metadata, expected)
        committed = json.loads((root / "expected-output" / variant / "report.json").read_text(encoding="utf-8"))
        # Committed numerical reports are runtime independent; installed versions may differ.
        committed.pop("execution_backend", None)
        report.pop("execution_backend")
        compare(report, committed)
