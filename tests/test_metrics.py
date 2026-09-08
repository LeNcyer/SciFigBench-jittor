import pytest

from scifigbench.metrics import anls, normalize_latex, position_anls, set_anls
from scifigbench.metrics.latex import levenshtein_tokens, tokenize_latex


@pytest.mark.parametrize("left,right", [
    (r"\frac{1}{2}", r"\dfrac{1}{2}"), ("x^2", r"\(x^{2}\)"),
    ("α", r"\alpha"), (r"a \le b", r"a \leq b"),
    ("x''", r"x^{\prime\prime}"), (r"\text{hello}", "hello"),
    (r"\left(x\right)", "(x)"),
])
def test_equivalent_latex(left, right):
    assert anls(left, right) == 1
    for value in (left, right):
        once = normalize_latex(value)
        assert normalize_latex(once) == once


def test_tokens_threshold_and_case():
    assert tokenize_latex(r"\alpha_{12}") == [r"\alpha", "_", "{", "1", "2", "}"]
    assert levenshtein_tokens(["x", "y"], ["x"]) == 1
    assert anls("ab", "ac") == 0.5
    assert anls("abc", "axy") == 0
    assert anls("", "") == 1
    assert anls("", "x") == 0
    assert anls("A", "a") == 0
    assert anls("x^2", "x^{2}", normalize=False) < 1


def test_order_and_cardinality():
    assert position_anls(["A", "B"], ["B", "A"]) == 0
    assert set_anls(["A", "B"], ["B", "A"]) == 1
    assert position_anls(["A", "B"], ["A"]) == 0.5
    assert set_anls(["A"], ["A", "B"]) == 0.5
    assert set_anls(["A", "A"], ["A"]) == 0.5
    assert set_anls([], []) == 1
    assert set_anls([], [""]) == 0
    assert position_anls([], [""]) == 0
    assert set_anls(["A"], ["A", ""]) == 0.5


def test_more_than_eight_requires_optimal_matching():
    # Row-greedy takes abc->abc (1), leaving abd->xbc (0).
    # Optimal takes abc->xbc (2/3), abd->abc (2/3).
    tails = [chr(0x400 + i) for i in range(7)]
    assert set_anls(["abc", "abd"] + tails, ["abc", "xbc"] + tails) == pytest.approx(
        (7 + 4 / 3) / 9
    )
