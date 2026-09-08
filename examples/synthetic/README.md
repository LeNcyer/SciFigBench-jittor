# Synthetic examples

One original flow diagram has Input -> Encoder -> Output, two distinct Skip
nodes pointing into Encoder, and an isolated node. All four edges are directed.
Encoder therefore has in-degree 3, out-degree 1, and incoming neighbor texts
["Input", "Skip", "Skip"].

The diagram and 11 questions are software fixtures, not benchmark data.
Five per-task QA files retain the source formats, including JSON-encoded
information-flow reference arrays. Images intentionally live in a nested folder
to exercise data-root path resolution.

Regenerate from this directory's script:

~~~bash
python examples/synthetic/generate.py
~~~

This overwrites the generated synthetic image, QA, predictions and expected.json.
Use --out-dir to generate a separate copy. Pixel reproduction requires matching
Pillow versions; the committed PNG is the canonical preview example.

Run all examples with:

~~~bash
python examples/synthetic/run_demo.py --out-dir reports/synthetic
~~~

The correct prediction set scores 1.0 on every task. Mixed predictions score:

| Task | Score | Coverage |
|---|---:|---:|
| Dense | 5/6 | 1 |
| Edge | 1/2 | 1/2 |
| Degree | 1/2 | 1 |
| Reachability | 1/2 | 1 |
| Information flow | 1/3 | 2/3 |

Input and Output have token ANLS 0.5, so swapping them in the dense three-item
answer receives partial credit. Information flow deliberately combines a
correct duplicate-preserving answer, a missing response, and an invalid answer
to the empty-neighbor question.

Example rendered inputs and reports are provided in expected-output/.
No evaluation metadata is derived from a private benchmark.
