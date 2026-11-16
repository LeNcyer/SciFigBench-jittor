<h1 align="center">🔬 SciFigBench-Jittor</h1>

<p align="center">
  <a href="https://github.com/LeNcyer/SciFigBench-jittor/blob/main/README.md">English</a> · <strong>简体中文</strong> · <a href="https://zhuanlan.zhihu.com/p/2104932244280366850">中文解读</a>
</p>

<p align="center">
  <strong>评估视觉语言模型对论文主图的理解能力。</strong>
</p>

<p align="center">
  <a href="https://github.com/LeNcyer/SciFigBench/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-blue"
       alt="MIT 许可证">
</a>
  <img src="https://img.shields.io/badge/Python-3.10–3.12-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10–3.12">
  <img src="https://img.shields.io/badge/Backend-Jittor%20CPU-e4572e" alt="Jittor CPU float64 评测">
  <a href="https://arxiv.org/abs/2610.27087">
    <img src="https://img.shields.io/badge/arXiv-2610.27087-b31b1b" alt="arXiv 论文">
  </a>
  <a href="https://github.com/LeNcyer/SciFigBench/blob/main/assets/SciFigBench%20Poster.pdf">
  <img src="https://img.shields.io/badge/Poster-PDF-D14836"
       alt="SciFigBench 会议海报">
</a>
  <a href="https://huggingface.co/datasets/LeNcyer/SciFigBench">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-FFD21E" alt="Hugging Face 数据集">
  </a>
</p>

<p align="center">
  <a href="#news">最新动态</a> ·
  <a href="#overview">项目概览</a> ·
  <a href="#quick-start">快速开始</a> ·
  <a href="#license-and-citation">许可证与引用</a> ·
  <a href="https://github.com/LeNcyer/SciFigBench">SciFigBench 官方仓库</a>
</p>

---

<a id="news"></a>

## 🔥 最新动态

- **[2026 年 10 月 20 日]** 🎉 **SciFigBench 正式发布！** 经过第二轮人工复核，我们进一步提升了基准的可靠性，并修订了约 5.8% 的参考答案。
- **[2026 年 8 月 21 日]** 🎉 论文 **SciFigBench: A Benchmark for Scientific Method Figure Understanding** 被 **EMNLP 2026 主会**录用！
- **[2026 年 5 月 26 日]** 📝 我们将 SciFigBench 论文投稿至 **2026 年 5 月 ARR**，目标会议为 EMNLP 2026。

<a id="overview"></a>

## 👋 项目概览

**SciFigBench** 是用于评估视觉语言模型（VLM）理解论文主图并进行结构推理的诊断性基准，包含 **4,352 张真实论文主图**和 **40,670 对QA**。基准围绕**密集内容感知**、**视觉关系理解**和**信息流追踪**三项核心能力，设置了五个细粒度任务。

SciFigBench 分别评估文本识别、箭头所表达的结构关系和信息流推理，帮助定位 VLM 理解论文主图时的薄弱环节。

<p align="center">
  <img src="https://raw.githubusercontent.com/LeNcyer/SciFigBench/main/assets/benchmark_overview.png" alt="SciFigBench 基准概览" width="500">
</p>

<p align="center">
  <em>覆盖三项互补能力的五个诊断任务。</em>
</p>

<div align="center">

| 任务 | 评估重点 | 问答对数量 | 指标 |
|---|---|---:|---|
| **密集内容感知** | 识别高亮区域内的文本 | 8,367 | 按位置对齐的 ANLS |
| **边级验证** | 验证节点间的有向边 | 8,665 | 准确率 |
| **节点度数统计** | 统计节点的入边或出边 | 7,848 | 准确率 |
| **路径可达性判断** | 推理有向路径的可达性 | 8,144 | 准确率 |
| **信息流追踪** | 结合文本和连接关系追踪直接邻居 | 7,646 | 基于集合的 ANLS* |
| **合计** | | **40,670** | |

</div>

SciFigBench 还提供经过人工核验的内容与结构标注，供进一步研究使用。完整数据集已发布至 [Hugging Face](https://huggingface.co/datasets/LeNcyer/SciFigBench)。

**答案修订对论文的影响：** 论文使用的是较早版本的问答数据。基于新版答案的评估显示，模型分数变化较小（浮动2%左右），且模型间相对分数的变化保持一致，不影响论文核心结论。

**如果你喜欢我们的项目，欢迎在 GitHub 上点个 Star ⭐！**

<a id="quick-start"></a>

## 🚀 快速开始

Coding Agent使用说明见 [AGENTS.md](https://github.com/LeNcyer/SciFigBench-jittor/blob/main/AGENTS.md)。

### 1. 安装

请使用 **Linux/WSL2**、**Python 3.10–3.12**、对应版本的 Python 开发头文件及 C++ 编译器。Ubuntu 安装示例：

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv libomp-dev

git clone https://github.com/LeNcyer/SciFigBench-jittor.git
cd SciFigBench-jittor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "numpy>=1.24,<2" "jittor==1.3.11.0"
python -m pip install .

# 验证 Jittor CPU / float64；首次运行会编译运行时。
python -c "from scifigbench.backend import metadata; print(metadata())"
```

安装包名为 **`scifigbench-jittor`**，Python 导入名和命令行名称均为 **`scifigbench`**。请使用独立虚拟环境，避免与官方包冲突。

### 2. 一键评测

提供你自己的可调用 VLM（`model(prompt=..., image=...) -> str`），工具包会负责数据准备、模型调用和 Jittor 评分。

```python
from scifigbench import SciFigBench

model = MyVLM(...)  # 你自己的可调用 VLM，返回原始文本字符串

# 快速测试：每项任务抽取 5 个问答，最多调用模型 25 次。
# 仅为部分样本分数，不是官方完整基准结果。
result = SciFigBench(model=model)
print(result.scores)

# 完整评测：全部 40,670 个问答，可能产生较高的模型 API 费用。
full = SciFigBench(
    model=model,
    max_samples_per_task=None,
    confirm_full_run=True,
)
print(full.scores)
```

### 3. 下载数据集

数据集托管于 [Hugging Face](https://huggingface.co/datasets/LeNcyer/SciFigBench)。

```bash
scifigbench download --repo-id LeNcyer/SciFigBench --revision main --data-root ./data
```

为已保存的预测结果评分时，可添加 `--qa-only` 仅下载问答数据。`main` 指向最新数据；如需复现，请固定具体提交版本。

### 4. 运行你的 VLM

为每个问题准备官方提示词和渲染后的图像：

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

将每个问题的 `qid` 和原始字符串 `response` 保存为一行 JSONL。根据任务提示，回答应为 `true`/`false`、整数文本或字符串列表的 JSON 文本。

### 5. 评估预测结果

对上述单题示例进行边级验证任务的部分评测：

```bash
scifigbench evaluate --predictions predictions.jsonl --task edge_level_verification --data-root ./data --allow-partial
```

若预测结果完整覆盖全部五项任务：

```bash
scifigbench evaluate --predictions all_predictions.jsonl --task all --data-root ./data
```

使用合成数据离线验证全部五项 Jittor 评分器：

```bash
python examples/synthetic/run_demo.py --out-dir reports/synthetic
```

<a id="license-and-citation"></a>

## ✍️ 许可证与引用

SciFigBench 评测工具包采用 **MIT 许可证**，详见 [LICENSE](https://github.com/LeNcyer/SciFigBench/blob/main/LICENSE)。

如果这个工作对你有所帮助，请使用以下引用：

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
