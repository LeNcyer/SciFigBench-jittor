# Original synthetic example

This directory contains one original synthetic diagram and 11 QA pairs across
five tasks, using the released public field contract. Manifest version is
synthetic-v2. SYN question IDs and the public-style images/F000001.jpg name are
local fixtures, not official benchmark data. The toolkit's MIT license applies.

```bash
python examples/synthetic/run_demo.py --out-dir reports/synthetic
```

The demo validates via load_qa, renders all inputs and calls the installed CLI
with correct and mixed raw-text predictions. The mixed run explicitly enables
partial evaluation. Use --overwrite to replace outputs. Expected values are
hand specified in expected.json; committed reports are under expected-output.

Correct predictions score 1.0 on all tasks. Mixed DCP scores 5/6, Degree and
Reachability score 1/2. Mixed Edge scores 1 with coverage 1/2; mixed IFT scores
1 with coverage 2/3 and one parse failure against an empty reference. These
reflect the official matched-question denominator and empty-answer behavior.
Synthetic scores are software checks, not model or benchmark performance.

To regenerate fixture data run generate.py, then rerun the demo with
--out-dir examples/synthetic/expected-output --overwrite on Linux/WSL2 with Jittor.
The generated checksums cover QA and images, not predictions or reports.
