# Evaluation policy

This port follows official SciFigBench commit 13088544a1884a1d25286932ce9eab65943ba677.
The reference code is preserved under tests/reference_upstream for offline tests.

## Alignment and denominators

Full prediction coverage of selected QA is required by default. allow_partial
explicitly scores matched questions only. Missing counts and coverage always
refer to all selected QA, not the sampled plan. A task with no matched responses
has null metrics; a genuinely scored zero is 0.0. No combined score exists.

## Parsing and metrics

Lists are parsed as JSON before interpreting Markdown, preserving literal
backticks inside strings. Complete outer fences and the approved balanced-list
fallback are retained. Upstream coercion of list items to strings is preserved.
Boolean/integer tasks use the approved first-match parsers and accuracy.

Token ANLS uses conservative LaTeX normalization and the inclusive 0.5 threshold.
DCP single binds to prediction item 0, including for its auxiliary metric.
Multi-box DCP uses position-wise ANLS, plus auxiliary set-ANLS*. IFT uses exact
Hungarian set-ANLS*, retaining repeated text from distinct neighbors.

The official implementation pads shorter lists with empty strings. Empty or
normalized-empty text can match padding and receive credit. For example,
position_anls(["A"], ["A", ""]) and set_anls([], [""]) both return 1.0.
IFT parsing failure uses an empty parsed list, so an empty reference scores 1.0
while parse_ok remains false and the report increments n_parse_fail. These
behaviors deliberately match the released evaluator rather than 0.2.0.

## Jittor execution

Jittor 1.3.11.0 executes token edit distance, similarities, accuracy and numerical
reductions on CPU with float64, without gradients. SciPy solves the official
square assignment costs; indices are gathered and reduced through Jittor.
There is no Python/NumPy fallback backend. Python handles strings/JSON, Pillow
handles pixels, and the user's Callable supplies inference.

Importing, loading, downloading and rendering do not initialize Jittor. CPU flags,
initialization environment overrides and stdout redirection are restored after
the scoring context. The runner probes the backend before invoking the model;
user calls occur outside that context. Reports record runtime versions, CPU
and precision; resume rejects backend identity changes. Use Linux/WSL2, supported
Python, matching development headers and a C++ compiler.
