r"""LaTeX normalization + tokenization + token-level ANLS for dense-content/VLM eval.

Goal: collapse presentation-level differences between two LaTeX strings without
hiding real text transcription errors. Conservative: prefer false-negative
("they didn't match because the normalizer was too cautious") over false-
positive ("they matched because the normalizer washed out an error").

Public API
----------
    normalize_latex(s)         -> str
    tokenize_latex(s)          -> list[str]
    levenshtein_tokens(a, b)   -> int
    anls(gt, pred, threshold=0.5)        -> float
    set_anls(gt_list, pred_list, ...)    -> float   (Hungarian)
    position_anls(gt_list, pred_list, ...) -> float  (pad-with-empty, max denom)

All functions are pure / deterministic / reproducible.

Adapted from SciFigBench; see PROVENANCE.md for the source revision.
"""
from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Unicode → LaTeX (trailing space inserted to prevent \alphax-style merges)
# ---------------------------------------------------------------------------

_UNICODE_TO_LATEX: dict[str, str] = {
    # Relations
    "≥": r"\geq", "≤": r"\leq", "≠": r"\neq",
    "≈": r"\approx", "≡": r"\equiv", "∝": r"\propto",
    "≪": r"\ll", "≫": r"\gg",
    # Arrows
    "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "↦": r"\mapsto", "⟶": r"\longrightarrow",
    # Operators
    "×": r"\times", "÷": r"\div", "·": r"\cdot", "∗": r"\ast",
    "±": r"\pm", "∓": r"\mp",
    "∘": r"\circ", "⊕": r"\oplus", "⊗": r"\otimes",
    # Set theory
    "∈": r"\in", "∉": r"\notin", "∋": r"\ni",
    "⊂": r"\subset", "⊃": r"\supset",
    "⊆": r"\subseteq", "⊇": r"\supseteq",
    "∪": r"\cup", "∩": r"\cap", "∅": r"\emptyset",
    # Big operators
    "∑": r"\sum", "∏": r"\prod", "∐": r"\coprod",
    "∫": r"\int", "∬": r"\iint", "∭": r"\iiint", "∮": r"\oint",
    "⋂": r"\bigcap", "⋃": r"\bigcup",
    # Calculus / analysis
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "√": r"\sqrt",
    "⟨": r"\langle", "⟩": r"\rangle",
    # Logic
    "∀": r"\forall", "∃": r"\exists", "¬": r"\neg",
    "∧": r"\wedge", "∨": r"\vee",
    # Greek lower
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "ϕ": r"\phi", "ϵ": r"\epsilon", "ϑ": r"\theta", "ϱ": r"\rho",
    # Greek upper
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
    "Ψ": r"\Psi", "Ω": r"\Omega", "ϒ": r"\Upsilon",
    # Blackboard convenience (rare; preserved as-is if no LaTeX equivalent)
    "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}",
}


# ---------------------------------------------------------------------------
# Environments to strip headers of (content preserved; structural envs absent)
# ---------------------------------------------------------------------------

_STRIP_ENV_NAMES = (
    "equation", "align", "aligned", "alignat", "gather", "gathered",
    "multline", "split", "eqnarray", "math", "displaymath",
)
_STRIP_ENV_GROUP = "|".join(_STRIP_ENV_NAMES)
_ENV_BEGIN_RE = re.compile(
    rf"\\begin\{{(?:{_STRIP_ENV_GROUP})\*?\}}(?:\s*\{{[^{{}}]*\}})?"
)
_ENV_END_RE = re.compile(rf"\\end\{{(?:{_STRIP_ENV_GROUP})\*?\}}")

# Structural envs that MUST be preserved (matrix family, cases, array)
# - we don't strip them, so their \\ and & survive unchanged.


# ---------------------------------------------------------------------------
# Outer math wrappers (only strip when they wrap the WHOLE remaining string)
# ---------------------------------------------------------------------------

_WRAPPER_PAIRS = [
    (re.compile(r"^\s*\\\(\s*(.*?)\s*\\\)\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\\\[\s*(.*?)\s*\\\]\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\$\$\s*(.*?)\s*\$\$\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*\$\s*(.*?)\s*\$\s*$", re.DOTALL), r"\1"),
    (re.compile(r"^\s*`\s*(.*?)\s*`\s*$", re.DOTALL), r"\1"),
]


# ---------------------------------------------------------------------------
# Sizing / spacing / annotation commands to remove
# ---------------------------------------------------------------------------

# Delimiter sizing — drop the command, keep the delimiter that follows
_SIZING_RE = re.compile(
    r"\\(?:left|right|middle"
    r"|big|Big|bigg|Bigg"
    r"|bigl|bigr|bigm|Bigl|Bigr|Bigm"
    r"|biggl|biggr|biggm|Biggl|Biggr|Biggm)(?![A-Za-z])"
)

# Spacing commands → single space (later collapsed)
_SPACE_CMD_RE = re.compile(
    r"\\(?:,|;|:| |!"
    r"|quad|qquad|thinspace|medspace|thickspace|enspace|negthinspace"
    r"|negmedspace|negthickspace)(?![A-Za-z])"
)

# Spacing commands with one braced arg → drop entirely (incl. arg)
_SPACE_ARG_CMD_RE = re.compile(
    r"\\(?:phantom|hphantom|vphantom|smash|hspace|vspace|kern|mkern|mskip)"
    r"\*?\s*\{[^{}]*\}"
)

# Numbering / labeling / display hints — drop entirely
_LABEL_RE = re.compile(
    r"\\(?:label|tag|nonumber|notag|limits|nolimits|displaystyle"
    r"|textstyle|scriptstyle|scriptscriptstyle)(?:\s*\{[^{}]*\})?"
)


# ---------------------------------------------------------------------------
# Fractions / relations / operators — alias to canonical
# ---------------------------------------------------------------------------

# Order matters: longer first so e.g. \longrightarrow isn't shadowed by \to.
_ALIAS_RULES = [
    # Fractions
    (r"\\dfrac(?![A-Za-z])",      r"\\frac"),
    (r"\\tfrac(?![A-Za-z])",      r"\\frac"),
    (r"\\cfrac(?![A-Za-z])",      r"\\frac"),
    (r"\\nicefrac(?![A-Za-z])",   r"\\frac"),
    (r"\\sfrac(?![A-Za-z])",      r"\\frac"),
    # Arrows — longer first
    (r"\\longrightarrow(?![A-Za-z])",     r"\\rightarrow"),
    (r"\\longleftarrow(?![A-Za-z])",      r"\\leftarrow"),
    (r"\\longleftrightarrow(?![A-Za-z])", r"\\leftrightarrow"),
    (r"\\Longrightarrow(?![A-Za-z])",     r"\\Rightarrow"),
    (r"\\Longleftarrow(?![A-Za-z])",      r"\\Leftarrow"),
    (r"\\Longleftrightarrow(?![A-Za-z])", r"\\Leftrightarrow"),
    (r"\\implies(?![A-Za-z])",            r"\\Rightarrow"),
    (r"\\impliedby(?![A-Za-z])",          r"\\Leftarrow"),
    (r"\\iff(?![A-Za-z])",                r"\\Leftrightarrow"),
    (r"\\to(?![A-Za-z])",                 r"\\rightarrow"),
    (r"\\gets(?![A-Za-z])",               r"\\leftarrow"),
    # Relations — canonical = long form (matches content annotation convention)
    (r"\\le(?![A-Za-z])",   r"\\leq"),
    (r"\\ge(?![A-Za-z])",   r"\\geq"),
    (r"\\ne(?![A-Za-z])",   r"\\neq"),
    # Operators
    (r"\\cdotp(?![A-Za-z])",     r"\\cdot"),
    (r"\\centerdot(?![A-Za-z])", r"\\cdot"),
    # Accents (single-char wide variants → narrow)
    (r"\\widehat(?![A-Za-z])",   r"\\hat"),
    (r"\\widetilde(?![A-Za-z])", r"\\tilde"),
]
_ALIAS_COMPILED = [(re.compile(p), r) for p, r in _ALIAS_RULES]


# ---------------------------------------------------------------------------
# Font handling
#   - \text* family → strip command, keep arg
#   - \mathbb / \mathbf / \mathcal / \mathfrak / \mathds / \mathscr  → PRESERVE
#   - \operatorname{X}  → \mathrm{X}   (both write-styles converge on \mathrm)
#   - \mathrm / \mathit / \mathsf / \mathtt → PRESERVE as-is
# ---------------------------------------------------------------------------

_TEXT_FONT_RE = re.compile(
    r"\\(?:text|textbf|textit|textrm|textsf|texttt|textnormal|emph"
    r"|mbox|hbox)\s*\{([^{}]*)\}"
)
_OPERATORNAME_RE = re.compile(r"\\operatorname\*?\s*\{([^{}]*)\}")


# ---------------------------------------------------------------------------
# Prime normalization: x' / x^\prime / x^{\prime} → x^{\prime}
# Multi-prime: x'' → x^{\prime\prime}
# ---------------------------------------------------------------------------

# Match a run of one or more ASCII apostrophes that is NOT preceded by ^_ {
# (already-attached primes). Replace with ^{\prime...}.
_PRIME_RE = re.compile(r"(?<![\\\^_{])'+")
# Existing x^\prime / x^{\prime\prime} stay valid; we also collapse stray
# spacing inside the brace: x^{ \prime } → x^{\prime}
_PRIME_BRACE_TIDY_RE = re.compile(r"\^\{\s*((?:\\prime\s*)+)\}")


def _collapse_prime_runs(s: str) -> str:
    def repl(m: re.Match) -> str:
        n = len(m.group(0))
        return "^{" + r"\prime" * n + "}"
    s = _PRIME_RE.sub(repl, s)
    # Tidy any pre-existing braced primes
    s = _PRIME_BRACE_TIDY_RE.sub(
        lambda m: "^{" + "".join(m.group(1).split()) + "}", s)
    return s


# ---------------------------------------------------------------------------
# Sub/sup single-token brace wrapping
#
# After `^` or `_`, find the extent of the next token and wrap with `{...}`.
# A "token" is:
#   - already braced group `{...}` (leave alone)
#   - `\command` + optional `*` + any tightly-following `{...}` arg groups
#   - `\<escaped non-letter>` (e.g. `\%`)
#   - single character (letter / digit / other)
# ---------------------------------------------------------------------------

def _next_token_extent(s: str, pos: int) -> int:
    """Return exclusive end index of the token starting at s[pos]. Caller
    guarantees pos < len(s)."""
    c = s[pos]
    if c == "{":
        depth = 1
        i = pos + 1
        while i < len(s) and depth > 0:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        return i
    if c == "\\":
        i = pos + 1
        if i < len(s) and not s[i].isalpha():
            return i + 1  # escaped non-letter e.g. \%
        while i < len(s) and s[i].isalpha():
            i += 1
        if i < len(s) and s[i] == "*":
            i += 1
        # Gobble immediate brace-args (no intervening space) so that
        # x^\frac{a}{b} is treated as one extent.
        while i < len(s) and s[i] == "{":
            depth = 1
            j = i + 1
            while j < len(s) and depth > 0:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            i = j
        return i
    return pos + 1


def _wrap_subsup(s: str) -> str:
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        # Skip escaped underscore/caret: `\_` / `\^` are literal chars, not sub/sup.
        if c == "\\" and i + 1 < n and s[i + 1] in "_^":
            out.append(s[i:i + 2])
            i += 2
            continue
        out.append(c)
        if c == "_" or c == "^":
            i += 1
            while i < n and s[i] == " ":
                i += 1
            if i >= n:
                continue
            end = _next_token_extent(s, i)
            tok = s[i:end]
            if tok.startswith("{"):
                out.append(tok)
            else:
                out.append("{")
                out.append(tok)
                out.append("}")
            i = end
        else:
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Top-level: fixpoint-safe normalize
# ---------------------------------------------------------------------------

def _strip_wrappers_fixpoint(s: str) -> str:
    for _ in range(8):
        new = s
        for pat, repl in _WRAPPER_PAIRS:
            new = pat.sub(repl, new).strip()
        if new == s:
            break
        s = new
    return s


def _strip_envs(s: str) -> str:
    s = _ENV_BEGIN_RE.sub("", s)
    s = _ENV_END_RE.sub("", s)
    return s


def _strip_text_fonts(s: str) -> str:
    for _ in range(4):
        new = _TEXT_FONT_RE.sub(r"\1", s)
        if new == s:
            break
        s = new
    return s


def _operatorname_to_mathrm(s: str) -> str:
    return _OPERATORNAME_RE.sub(lambda m: r"\mathrm{" + m.group(1) + "}", s)


def _apply_aliases(s: str) -> str:
    for pat, repl in _ALIAS_COMPILED:
        s = pat.sub(repl, s)
    return s


def _strip_redundant_outer_braces(s: str) -> str:
    """If the whole string is wrapped in braces and the braces are balanced
    around the entire content, drop one pair. Repeat to fixpoint."""
    for _ in range(4):
        s = s.strip()
        if not (s.startswith("{") and s.endswith("}")):
            return s
        depth = 0
        balanced_whole = True
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    balanced_whole = False
                    break
        if not balanced_whole:
            return s
        s = s[1:-1]
    return s


def normalize_latex(s: str | None) -> str:
    """Conservative Tier 1 + Tier 2 LaTeX normalization.

    Idempotent (multiple applications yield the same result).
    """
    if s is None:
        return ""
    s = str(s)
    # 1. Unicode → LaTeX (pad both sides with space; whitespace gets collapsed later)
    for u, latex in _UNICODE_TO_LATEX.items():
        if u in s:
            s = s.replace(u, " " + latex + " ")
    # 2. Strip equation-family environment headers (matrix family preserved)
    s = _strip_envs(s)
    # 3. Strip outer math/code wrappers, fixpoint
    s = _strip_wrappers_fixpoint(s)
    # 4. Drop \left \right \big ... \middle ...
    s = _SIZING_RE.sub("", s)
    # 5. Drop spacing-arg commands first (so their {...} arg goes away atomically)
    s = _SPACE_ARG_CMD_RE.sub(" ", s)
    # 6. Drop annotation commands (\label \tag \limits \nolimits ...)
    s = _LABEL_RE.sub("", s)
    # 7. Drop spacing commands (\, \; \quad ...)
    s = _SPACE_CMD_RE.sub(" ", s)
    # 8. Strip text-font wrappers (\text \textbf \mbox ...)
    s = _strip_text_fonts(s)
    # 9. \operatorname{X} → \mathrm{X}  (canonical for text fidelity)
    s = _operatorname_to_mathrm(s)
    # 10. Command aliases (frac family, arrow long forms, relation aliases, ...)
    s = _apply_aliases(s)
    # 11. Prime normalization
    s = _collapse_prime_runs(s)
    # 12. Sub/sup single-token brace wrapping
    s = _wrap_subsup(s)
    # 13. Strip redundant whole-string outer braces
    s = _strip_redundant_outer_braces(s)
    # 14. Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+\*?"          # \frac, \alpha, \begin, \frac* etc.
    r"|\\[^a-zA-Z\s]"          # escaped non-letter: \%, \{, \\, \&
    r"|[0-9]"                  # single digit
    r"|[a-zA-Z]"               # single letter
    r"|[^\s]"                  # operators, braces, ^, _, +, =, (, ) ...
)


def tokenize_latex(s: str | None) -> list[str]:
    if s is None:
        return []
    return _TOKEN_RE.findall(str(s))


# ---------------------------------------------------------------------------
# Token-level Levenshtein + ANLS
# ---------------------------------------------------------------------------

def levenshtein_tokens(a: Sequence[str], b: Sequence[str]) -> int:
    """Exact token edit distance, executed by a Jittor CPU custom operator."""
    from .. import backend

    with backend.cpu_runtime() as jt:
        return int(backend.edit_distances(jt, [(a, b)]).item())


def _tokens(value: str, normalize: bool) -> list[str]:
    return tokenize_latex(normalize_latex(value) if normalize else (value or ""))


def anls(gt: str, pred: str, *, normalize: bool = True,
         threshold: float = 0.5) -> float:
    """Token-level ANLS on optionally-normalized LaTeX strings.

    Both empty → 1.0; exactly one empty → 0.0; otherwise
    NLS = 1 - editdist / max(len_tokens_gt, len_tokens_pred), zeroed below
    `threshold` (DocVQA / ANLS convention).
    """
    from .. import backend

    with backend.cpu_runtime() as jt:
        pairs = [(_tokens(gt, normalize), _tokens(pred, normalize))]
        return float(backend.similarities(jt, pairs, threshold=threshold).item())


def position_anls(gt_list: list[str], pred_list: list[str], *,
                  normalize: bool = True, threshold: float = 0.5) -> float:
    """Position-aligned ANLS with explicit zero credit for unmatched slots."""
    from .. import backend

    n = max(len(gt_list), len(pred_list))
    with backend.cpu_runtime() as jt:
        if not gt_list or not pred_list:
            return float(jt.array([1.0 if n == 0 else 0.0], dtype="float64").item())
        pairs = [(_tokens(g, normalize), _tokens(p, normalize))
                 for g, p in zip(gt_list, pred_list)]
        scores = backend.similarities(jt, pairs, threshold=threshold)
        return float((scores.sum() / n).item())


def set_anls(gt_list: list[str], pred_list: list[str], *,
             normalize: bool = True, threshold: float = 0.5) -> float:
    """Jittor similarities and reduction, with SciPy exact assignment indices."""
    from scipy.optimize import linear_sum_assignment
    from .. import backend

    n = max(len(gt_list), len(pred_list))
    with backend.cpu_runtime() as jt:
        if not gt_list or not pred_list:
            return float(jt.array([1.0 if n == 0 else 0.0], dtype="float64").item())
        references = [_tokens(value, normalize) for value in gt_list]
        predictions = [_tokens(value, normalize) for value in pred_list]
        pairs = [(g, p) for g in references for p in predictions]
        # Bound custom-op input batches; no per-token Python/Jittor dispatch.
        chunks = [backend.similarities(jt, pairs[i:i + 256], threshold=threshold)
                  for i in range(0, len(pairs), 256)]
        similarity = jt.concat(chunks).reshape((len(references), len(predictions)))
        # The rectangular assignment is equivalent to zero-padded square matching.
        rows, cols = linear_sum_assignment(-similarity.numpy())
        indices = jt.array(rows, dtype="int32") * len(predictions) + jt.array(cols, dtype="int32")
        matched = similarity.reshape((-1,))[indices]
        return float((matched.sum() / n).item())
