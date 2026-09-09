# Evaluation policy

## Jittor CPU execution

Scoring requires Jittor 1.3.11.0 on Linux or WSL2 with Python 3.10–3.12 and a C++
compiler. It runs on the CPU without gradients or distributed launchers. A missing
or failed Jittor runtime is an error; there is no alternative scoring backend.

LaTeX normalization, tokenization, JSON parsing and metadata grouping are Python
operations. Token sequences are encoded as integer IDs, and a Jittor CPU custom
operator computes exact Levenshtein distance with two dynamic-programming rows.
Jittor float64 operations calculate similarities, thresholds, positional scores,
accuracy, coverage, task means and subgroup means. Integer answers are compared
using canonical decimal character encodings to preserve values beyond int64.

For set-ANLS*, Jittor builds the similarity matrix and SciPy returns the exact
assignment indices. Jittor gathers those matches and computes the final score.
NumPy is used at the Jittor/SciPy data boundary. It is not a fallback scorer.
Neither PyTorch, Accelerate nor xformers is required or imported by this toolkit.

Jittor compiles on first use and caches its generated code. Keep a compiler
available for new operators. CLI compiler output is directed to stderr so stdout
remains valid JSON. Validation and Pillow rendering do not initialize Jittor.

Report schema 1.1 records the Jittor version, CPU device, float64 precision and
SciPy solver version under `execution_backend`. Scores are checked against the
previous implementation with an absolute tolerance of 1e-12.

The denominator is the number of QA rows for the selected task. Missing responses
and parse failures contribute zero, including within subgroups. Coverage counts
submitted IDs, including malformed answer text; it is not parse success rate.
Empty QA, invalid QA, duplicate IDs and unknown prediction IDs terminate evaluation.

## Scalar answers

Edge and reachability use exact match after extracting the first standalone
true/false token, case-insensitively. Degree uses the first signed integer token.
These permissive extraction rules preserve the source evaluator: explanatory
text can parse, and text "1.5" yields integer 1. Submit only the requested answer
to avoid ambiguity. Native booleans are not accepted as degree predictions.

## String-list answers

Native string arrays, JSON array text and fenced JSON are supported. If whole
text is not valid JSON, the parser attempts the first bracketed JSON array.
Non-string list elements are not coerced. A failed parse always scores zero.
A valid empty list scores one only against an empty information-flow answer.

## Token-level ANLS

Optional LaTeX normalization preserves the source implementation, including
presentation wrappers, Unicode/LaTeX aliases and subscript/superscript braces.
It is a syntactic normalizer, not a symbolic equivalence checker.

Tokens are LaTeX commands, individual letters/digits and other non-whitespace
symbols. After token-level Levenshtein distance d, similarity is
1 - d / max(reference_length, prediction_length). Similarities **at least 0.5**
are retained; smaller values become zero. Letter case matters. Two empty token
sequences score one; exactly one empty sequence scores zero.

--no-latex-normalize disables normalization but retains tokenization and the
threshold. Reports always record the selected setting.

Dense single uses the first predicted string only, preserving upstream behavior:
extra strings do not reduce this subtask's score. Dense flow_path averages
position-aligned ANLS over the longer list length; missing/extra positions score
zero. Dense also reports the source additional set-ANLS* variant (single unchanged,
flow_path order-independent).

## set-ANLS*

Build pairwise ANLS similarities and find the maximum-total one-to-one
assignment indices using SciPy. Gather the matches with Jittor and divide by
max(reference_count, prediction_count).
Unmatched items score zero, with no approximate fallback at large sizes.
Repeated labels remain separate occurrences; order does not matter.
For example, ["A", "A"] versus ["A"] scores 0.5.

This is the same intended optimum as the upstream SciPy branch. Results from
the old no-SciPy greedy fallback may differ. Unmatched padding is explicitly
zero rather than a matchable empty string.

## Diagnostics and interpretation

Task-specific subgroups and hard/reverse-negative diagnostics are preserved.
No cross-task combined benchmark score is defined by this preview. Subgroup
entries with n=0 have score zero; their counts must be considered when interpreting
them.

Correct synthetic predictions score one by construction. They validate software
behavior, not the quality of a model. The synthetic mixed fixtures independently
specify their expected scores and demonstrate missing responses and parse failures.
