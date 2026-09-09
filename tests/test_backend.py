"""Execution, precision and isolation checks for the required Jittor runtime."""
import importlib.util
import itertools
import json
import os
import random
import subprocess
import sys

import pytest

from scifigbench import backend
from scifigbench.metrics import anls, position_anls, set_anls
from scifigbench.metrics.latex import normalize_latex, tokenize_latex


def _reference_anls(a, b, normalize=True):
    a, b = [tokenize_latex(normalize_latex(x) if normalize else x) for x in (a, b)]
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (token != other)))
        previous = current
    similarity = 1.0 - previous[-1] / max(len(a), len(b))
    return similarity if similarity >= 0.5 else 0.0


def test_real_cpu_operator_and_float64():
    with backend.cpu_runtime() as jt:
        distances = backend.edit_distances(jt, [(["a", "b"], ["a"]), ([], ["x"]), ([], [])])
        assert isinstance(distances, jt.Var)
        assert distances.numpy().tolist() == [1, 1, 0]
        scores = backend.similarities(jt, [(["a", "b", "c"], ["a", "b"])], threshold=0.5)
        assert scores.dtype == "float64"
        assert scores.item() == pytest.approx(2 / 3, abs=1e-12, rel=0)
        assert jt.flags.use_cuda == 0 and jt.flags.no_grad == 1
        assert jt.compiler.nvcc_path == ""
    assert backend.metadata()["assignment_solver"]["name"] == "scipy"


def test_exact_large_integer_and_reductions():
    value = 10**100 + 1
    assert backend.exact_match(value, value) == 1
    assert backend.exact_match(value, value - 1) == 0
    assert backend.exact_match(-value, value) == 0
    assert backend.mean([1.0, 0.0, 0.0]) == pytest.approx(1 / 3, abs=1e-12, rel=0)
    assert backend.mean([]) == 0
    assert backend.ratio(1, 3) == pytest.approx(1 / 3, abs=1e-12, rel=0)


@pytest.mark.parametrize("normalize", [True, False])
def test_metrics_against_independent_reference(normalize):
    rng = random.Random(20260909)
    labels = ["A", "B", "ab", "abc", "abd", "xbc", "", r"\quad", "x^2", "x^{2}"]
    for _ in range(40):
        a = [rng.choice(labels) for _ in range(rng.randrange(5))]
        b = [rng.choice(labels) for _ in range(rng.randrange(5))]
        n = max(len(a), len(b))
        matrix = [[_reference_anls(a[i], b[j], normalize) if i < len(a) and j < len(b)
                   else 0.0 for j in range(n)] for i in range(n)]
        optimum = max((sum(matrix[i][p[i]] for i in range(n)) / n
                       for p in itertools.permutations(range(n))), default=1.0) if n else 1.0
        aligned = sum(matrix[i][i] for i in range(n)) / n if n else 1.0
        assert set_anls(a, b, normalize=normalize) == pytest.approx(optimum, abs=1e-12, rel=0)
        assert position_anls(a, b, normalize=normalize) == pytest.approx(aligned, abs=1e-12, rel=0)
    assert anls("ab", "ac", normalize=normalize) == 0.5


def test_no_optional_training_stack():
    assert all(importlib.util.find_spec(name) is None for name in ("torch", "accelerate", "xformers"))
    assert not any(name in sys.modules for name in ("torch", "accelerate", "xformers"))


def test_validation_and_rendering_do_not_import_jittor(examples, tmp_path):
    program = """
import importlib.abc, sys
class BlockJittor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jittor' or fullname.startswith('jittor.'):
            raise ImportError('Jittor is deliberately unavailable')
sys.meta_path.insert(0, BlockJittor())
from scifigbench.cli import main
assert main(['validate', '--qa', sys.argv[1]]) == 0
assert main(['render', '--qa', sys.argv[1], '--data-root', sys.argv[2], '--out-dir', sys.argv[3]]) == 0
assert 'jittor' not in sys.modules
assert main(['evaluate', '--task', 'edge_level_verification', '--qa', sys.argv[1],
             '--pred', sys.argv[4], '--report-dir', sys.argv[5]]) == 2
"""
    result = subprocess.run([
        sys.executable, "-c", program,
        str(examples / "qa/edge_level_verification.jsonl"), str(examples),
        str(tmp_path / "images"),
        str(examples / "predictions/correct/edge_level_verification.jsonl"),
        str(tmp_path / "report"),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "No fallback was used" in result.stderr


def test_missing_compiler_is_actionable(tmp_path):
    # A fresh cache and nonexistent compiler force initialization to fail before scoring.
    env = os.environ.copy()
    env.update({"cc_path": str(tmp_path / "missing-cxx"),
                "JITTOR_HOME": str(tmp_path / "jittor"), "use_mkl": "0"})
    program = """
from scifigbench.backend import BackendError
from scifigbench.metrics import anls
try:
    anls('abc', 'abc')
except BackendError as error:
    assert 'compiler' in str(error)
    print('expected failure')
else:
    raise AssertionError('missing compiler did not stop evaluation')
"""
    result = subprocess.run([sys.executable, "-c", program], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "expected failure"


def test_report_metadata_and_legacy_schema(examples, write_rows):
    from scifigbench.evaluate import evaluate
    from scifigbench.io import validate_schema

    report, _ = evaluate(examples / "qa/edge_level_verification.jsonl",
                         write_rows("empty.jsonl", []), task="edge_level_verification")
    assert report["toolkit_version"] == "0.2.0"
    assert report["schema_version"] == "1.1"
    assert report["execution_backend"]["name"] == "jittor"
    legacy = dict(report, toolkit_version="0.1.0", schema_version="1.0")
    legacy.pop("execution_backend")
    validate_schema(legacy, "report", "legacy report")
    legacy["schema_version"] = "1.1"
    with pytest.raises(ValueError, match="execution_backend"):
        validate_schema(legacy, "report", "new report")


def test_committed_reports_match_current_scores(examples):
    from scifigbench.evaluate import evaluate
    from scifigbench.io import TASKS

    for task in TASKS:
        for variant in ("correct", "mixed"):
            committed = json.loads((examples / "expected-output" / variant / task / "report.json").read_text())
            report, rows = evaluate(examples / "qa" / f"{task}.jsonl",
                                    examples / "predictions" / variant / f"{task}.jsonl", task=task)
            assert report["primary_metric"]["value"] == pytest.approx(
                committed["primary_metric"]["value"], abs=1e-12, rel=0)
            expected_rows = [json.loads(line) for line in (
                examples / "expected-output" / variant / task / f"per_question__{task}.jsonl"
            ).read_text().splitlines()]
            for actual, expected in zip(rows, expected_rows):
                assert actual["qid"] == expected["qid"]
                assert actual["score"] == pytest.approx(expected["score"], abs=1e-12, rel=0)
                assert actual["status"] == expected["status"]


@pytest.mark.parametrize("task,raw", [
    ("node_level_degree_counting", "9" * 5000),
    ("information_flow_tracing", "[" * 1200 + "]" * 1200),
    ("information_flow_tracing", "prefix " + "[" * 1200 + "]" * 1200),
], ids=["long-integer", "nested-array", "nested-array-fallback"])
def test_bad_response_does_not_abort_batch(task, raw, examples, write_rows):
    from scifigbench.evaluate import evaluate
    from scifigbench.io import load_qa

    qa = load_qa(examples / "qa" / f"{task}.jsonl")
    predictions = [{"qid": q["qid"], "response": q["answer"]} for q in qa]
    predictions[-1]["response"] = raw
    report, rows = evaluate(write_rows("qa.jsonl", qa), write_rows("pred.jsonl", predictions), task=task)
    assert report["n_parse_fail"] == 1
    assert report["coverage"] == 1
    assert rows[-1]["score"] == 0 and rows[-1]["status"] == "parse_failure"
    assert all(row["score"] == 1 for row in rows[:-1])
