![Python Version](https://badgen.net/static/Python/3.10.15-3.12/blue?icon=python)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![GitHub License](https://badgen.net/static/license/Apache%202.0/green)

High-quality data is the backbone of modern AI development, but acquiring diverse, domain-specific, and scalable datasets remains a major bottleneck. Synthetic data generation addresses this challenge by enabling the creation of tailored datasets that are:

- Cost-effective and privacy-preserving
- Customizable for specific tasks and domains
- Scalable to meet evolving model needs

DGT (Data Generation and Transformation) [pronounced "digit"] is a horizontal framework designed to streamline and scale expert, domain-specific synthetic data generation via simplifying and standardizing essential components.

## Features

- 🤖 Standardize interface for ~5+ different LM engines (WatsonX, OpenAI, Azure OpenAI, vLLM, ollama, anthropic etc.) with retry/fallback logic
- 💡 Support for several domain-specific pipelines for tool calling, time series, question answering and more
- 🧪 Growing list of syntactic validators, deduplicators, LLMaJs (LLM-as-a-Judge)
- 🔒 Local execution capabilities for sensitive data and air-gapped environments
- 🤖 Plug-and-play [integrations][integrations] incl. Docling
- 💻 Simple and convenient CLI
