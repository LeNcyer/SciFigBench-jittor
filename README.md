<h1 align="center">🔬 SciFigBench-Jittor</h1>

<p align="center">
  <strong>English</strong> · <a href="https://github.com/LeNcyer/SciFigBench-jittor/blob/main/docs/README_CN.md">简体中文</a> · <a href="https://zhuanlan.zhihu.com/p/2104932244280366850">中文解读</a>
</p>

<p align="center">
  <strong>Evaluating VLMs' Understanding of Scientific Method Figures.</strong>
</p>

<p align="center">
  <a href="https://github.com/LeNcyer/SciFigBench/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-blue"
       alt="MIT License">
</a>
  <img src="https://img.shields.io/badge/Python-3.10–3.12-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10–3.12">
  <img src="https://img.shields.io/badge/Backend-Jittor%20CPU-e4572e" alt="Jittor CPU float64 evaluation">
  <a href="https://arxiv.org/abs/2610.27087">
    <img src="https://img.shields.io/badge/arXiv-2610.27087-b31b1b" alt="Paper on arXiv">
  </a>
  <a href="https://github.com/LeNcyer/SciFigBench/blob/main/assets/SciFigBench%20Poster.pdf">
  <img src="https://img.shields.io/badge/Poster-PDF-D14836"
       alt="SciFigBench Conference Poster">
</a>
  <a href="https://huggingface.co/datasets/LeNcyer/SciFigBench">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-FFD21E" alt="Hugging Face Dataset">
  </a>
</p>

<p align="center">
  <a href="#news">News</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#license-and-citation">License & Citation</a> ·
  <a href="https://github.com/LeNcyer/SciFigBench">Official SciFigBench</a>
</p>

---

<a id="news"></a>

## 🔥 News

- **[Oct. 20, 2026]** 🎉 **SciFigBench is officially released!** Following a second round of human verification, we further strengthened the benchmark's reliability and revised approximately 5.8% of its reference answers.
- **[Aug. 21, 2026]** 🎉 Our paper, **SciFigBench: A Benchmark for Scientific Method Figure Understanding**, was accepted to **EMNLP 2026 Main Conference**!
- **[May 26, 2026]** 📝 We submitted SciFigBench to the **May 2026 ARR cycle**, targeting EMNLP 2026.

<a id="overview"></a>

## 👋 Overview

**SciFigBench** is a diagnostic benchmark for evaluating Vision-Language Models (VLMs) on scientific method figure understanding. It consists of **4,352 real-world scientific method figures** and **40,670 QA pairs**, organized into five fine-grained tasks across three core capabilities: **Dense Content Perception**, **Visual Relation Comprehension**, and **Information-Flow Tracing**.

By separately evaluating text recognition, arrow-induced structural relations, and information-flow reasoning, SciFigBench helps identify where VLMs struggle to understand scientific method figures.

<p align="center">
  <img src="https://raw.githubusercontent.com/LeNcyer/SciFigBench/main/assets/benchmark_overview.png" alt="SciFigBench Benchmark Overview" width="500">
</p>

<p align="center">
  <em>Five diagnostic tasks across three complementary capabilities.</em>
</p>

<div align="center">

| Task | Evaluation Focus | QA Pairs | Metric |
|---|---|---:|---|
| **Dense Content Perception** | Reading text across highlighted regions | 8,367 | Position-wise ANLS |
| **Edge-Level Verification** | Verifying directed edges between nodes | 8,665 | Accuracy |
| **Node-Level Degree Counting** | Counting incoming or outgoing edges | 7,848 | Accuracy |
| **Path-Level Reachability** | Reasoning about directed paths | 8,144 | Accuracy |
| **Information-Flow Tracing** | Tracing direct neighbors through text and connections | 7,646 | Set-based ANLS* |
| **Total** | | **40,670** | |

</div>

SciFigBench also provides human-verified content and structural annotations for further research. The full dataset is available on [Hugging Face](https://huggingface.co/datasets/LeNcyer/SciFigBench).

**Note on reported results.** The paper reports results on an earlier QA version. Retrospective evaluations show modest score changes, with relative model performance trends and main conclusions remaining largely consistent.

**If you like our project, please kindly give us a star ⭐ on GitHub.**

<a id="quick-start"></a>

## 🚀 Quick Start

For coding agents, advanced configuration and resume, see [AGENTS.md](https://github.com/LeNcyer/SciFigBench-jittor/blob/main/AGENTS.md).

### 1. Installation

Use **Linux/WSL2**, **Python 3.10–3.12**, matching Python development headers and a C++ compiler. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv libomp-dev

git clone https://github.com/LeNcyer/SciFigBench-jittor.git
cd SciFigBench-jittor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "numpy>=1.24,<2" "jittor==1.3.11.0"
python -m pip install .

# Verify Jittor CPU / float64; the first run compiles the runtime.
python -c "from scifigbench.backend import metadata; print(metadata())"
```

Package: **`scifigbench-jittor`**; Python import and CLI: **`scifigbench`**. Use a fresh environment to avoid conflicts with the official package.

### 2. One-call Evaluation

Provide your callable VLM (`model(prompt=..., image=...) -> str`). The toolkit handles data preparation, model calls and Jittor scoring.

```python
from scifigbench import SciFigBench

model = MyVLM(...)  # Your callable VLM, returning raw text

# Quick Test: 5 QA per task, up to 25 calls.
# Partial scores only, NOT official benchmark results.
result = SciFigBench(model=model)
print(result.scores)

# Full Evaluation: all 40,670 QA pairs; substantial API costs may apply.
full = SciFigBench(
    model=model,
    max_samples_per_task=None,
    confirm_full_run=True,
)
print(full.scores)
```

### 3. Download the Dataset

The dataset is hosted on [Hugging Face](https://huggingface.co/datasets/LeNcyer/SciFigBench).

```bash
scifigbench download --repo-id LeNcyer/SciFigBench --revision main --data-root ./data
```

Add `--qa-only` for scoring saved predictions. `main` selects the latest data; pin a commit for reproducibility.

### 4. Run Your VLM

Prepare the official prompt and rendered image for each question:

```python
import json
from scifigbench import load_qa, prepare_model_payload

qa = load_qa("./data", task="edge_level_verification")[0]
payload = prepare_model_payload(qa, data_root="./data")
try:
    response = model(**payload)
finally:
    payload["image"].close()

with open("predictions.jsonl", "w", encoding="utf-8") as output:
    output.write(json.dumps({"qid": qa["qid"], "response": response}) + "\n")
```

Save one `qid` and raw string `response` per JSONL line. Responses are `true`/`false`, integer text, or JSON lists of strings, according to the task prompt.

### 5. Evaluate Predictions

Score the one-question example as a partial Edge evaluation:

```bash
scifigbench evaluate --predictions predictions.jsonl --task edge_level_verification --data-root ./data --allow-partial
```

For complete predictions covering all five tasks:

```bash
scifigbench evaluate --predictions all_predictions.jsonl --task all --data-root ./data
```

For an offline synthetic check of all five Jittor scorers:

```bash
python examples/synthetic/run_demo.py --out-dir reports/synthetic
```

<a id="license-and-citation"></a>

## ✍️ License & Citation

The SciFigBench Evaluation Toolkit is released under the **MIT License**. See [LICENSE](https://github.com/LeNcyer/SciFigBench/blob/main/LICENSE).

If you find our work helpful, please use the following citation.

```bibtex
@inproceedings{wu2026scifigbench,
  title     = {SciFigBench: A Benchmark for Scientific Method Figure Understanding},
  author    = {Wu, Lijia and Li, Zheng and Liu, Yuxuan and Zhang, Jingwen and Zhao, Penghai and Li, Xiang},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  eprint    = {2610.27087},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2610.27087}
}
```
