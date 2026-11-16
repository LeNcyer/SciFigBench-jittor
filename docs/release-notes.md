# SciFigBench-Jittor 0.3.0

This release synchronizes with the official public toolkit at commit
13088544a1884a1d25286932ce9eab65943ba677 and retains Jittor CPU/float64 evaluation.

It adds public data downloading/verification, one-call evaluation and strict
resume. It replaces preview data/CLI/report contracts and synchronizes parsing,
normalization, metrics and partial-run policy. See CHANGELOG.md for migration.
README, AGENTS.md, dataset availability and paper citation now reflect the
released benchmark. Dataset files remain separately distributed on Hugging Face.

Use Linux/WSL2, Python 3.10–3.12, a C++ compiler and Python development headers.
The distribution is scifigbench-jittor; import and CLI are scifigbench. Use an
isolated environment without the official package. Import/help/data/render paths
remain lazy; actual scoring requires Jittor and SciPy. No online publication of
this version is implied by these local release notes.
