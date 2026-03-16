![Python Version](https://badgen.net/static/Python/3.10.15-3.12/blue?icon=python)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![GitHub License](https://badgen.net/static/license/Apache%202.0/green)

**DGT** (Data Generation and Transformation, pronounced "digit") is a framework for building synthetic data pipelines that generate training data for fine-tuning large language models.

Write a handful of seed examples. Point DiGiT at a model. Get a dataset. Typical runs take under five minutes on a laptop with Ollama.

## What it does

High-quality, domain-specific training data is the biggest bottleneck in LLM fine-tuning. DiGiT addresses this by letting you:

- **Generate** new examples from a small seed set using any LLM as a teacher model
- **Transform** existing data (add chain-of-thought, score for difficulty, reformat, filter)
- **Compose** generation and transformation stages into multi-step pipelines

## Features

- **6 LM engines out of the box:** Ollama, OpenAI, Azure OpenAI, Anthropic, WatsonX, vLLM — switch with a one-line config change
- **Built-in quality controls:** deduplicators, syntactic validators, LLM-as-a-Judge scoring
- **Concurrent execution:** async batch requests across all providers for fast throughput
- **Local-first:** runs entirely on your machine for sensitive data and air-gapped environments
- **Extensible:** add a new databuilder in three files; plug into the same CLI and engine layer

## Get started in 5 minutes

```bash
# 1. Clone and install
git clone git@github.com:IBM/fms-dgt.git && cd fms-dgt
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# 2. Pull a local model (no API key needed)
ollama pull granite4:3b

# 3. Generate 20 geography QA pairs
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/qa/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

Output: `output/public/examples/geography_qa/final_data.jsonl`

Ready to do more? See [Quick Start](usage.md) for the rater example, cloud provider setup, and CLI reference.
