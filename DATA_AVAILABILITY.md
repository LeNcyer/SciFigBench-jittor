# Data availability

SciFigBench was officially released on October 20, 2026. The public dataset is
available at https://huggingface.co/datasets/LeNcyer/SciFigBench: 4,352 figures
and 40,670 QA pairs across five tasks. The second human verification revised
approximately 5.8% of reference answers. The paper used an earlier QA version;
old predictions must not be presented as new inference on revised inputs.

This code repository does not bundle benchmark figures, QA or annotations.
The dataset's terms are separate from the toolkit's MIT license.

```bash
scifigbench download --repo-id LeNcyer/SciFigBench --revision main --data-root ./data
scifigbench download --repo-id LeNcyer/SciFigBench --revision main --data-root ./data --qa-only
```

`main` selects the latest public data and can change. For reproducible evaluation,
use commit `a884c09770b99e6337c4cb27cee4696773e27d6e`, also the one-call API default.
Downloads verify QA counts, manifests and SHA-256 checksums. Scoring needs QA
only; rendering needs images; annotations are optional. Local/offline operations
never implicitly download. The one-call API fetches verified QA and selected
images, pins the actual Hub commit and records provenance.

The original diagram and QA under examples/synthetic are synthetic-v2 software
fixtures with SYN QIDs. Their F000001 image name exercises the public file
contract and does not identify an official figure. Synthetic scores are not
benchmark or model performance results.
