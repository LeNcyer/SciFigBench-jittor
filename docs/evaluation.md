# Evaluation policy

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
assignment using SciPy. Divide by max(reference_count, prediction_count).
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
