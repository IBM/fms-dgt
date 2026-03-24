# FMS-DGT

DGT (pronounced "digit") is a framework that enables different algorithms and models to be used to generate synthetic data.

![Python Version](https://badgen.net/static/Python/3.10.15-3.12/blue?icon=python)
[![Code style: black](https://badgen.net/static/Code%20Style/black/black)](https://github.com/psf/black)
![GitHub License](https://badgen.net/static/license/Apache%202.0/green)

| [Setup](#setup) | [Quick Start](#quick-start) | [Usage](#usage) | [Studio](#studio) |

This is the main repository for DiGiT, our **D**ata **G**eneration and **T**ransformation framework.

## Setup

First clone the repository:

```bash
git clone git@github.com:IBM/fms-dgt.git
cd fms-dgt
```

Set up a Python virtual environment (Python 3.10 or later):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

> [!TIP]
> Using [uv](https://github.com/astral-sh/uv)? Run `uv sync --extra all` instead.

> [!IMPORTANT]
> Install the pre-commit hooks before contributing:
>
> ```bash
> pip install pre-commit
> pre-commit install
> ```

### API Keys

Copy `.env.example` to `.env` and fill in the keys for whichever providers you plan to use:

```bash
cp .env.example .env
```

```bash
# OpenAI [Optional]
OPENAI_API_KEY=<your key>

# Anthropic [Optional]
ANTHROPIC_API_KEY=<your key>

# IBM watsonx [Optional]
WATSONX_API_KEY=<your key>
WATSONX_PROJECT_ID=<your project id>

# Azure OpenAI [Optional]
AZURE_OPENAI_API_KEY=<your key>
```

## Quick Start

The fastest path is Ollama — no API key needed, runs entirely on your machine.

**Step 1:** Pull a model (2.1 GB, Apache 2.0, runs on 8 GB RAM):

```bash
ollama pull granite4:3b
```

**Step 2:** Generate geography QA pairs from a handful of seed examples:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/qa/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

Output lands in `output/public/examples/geography_qa/final_data.jsonl`. That is it.

### Using OpenAI instead

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/qa/task.yaml \
  --config-path ./configs/public/examples/openai_qa.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

Requires `OPENAI_API_KEY` in your `.env`.

## Usage

The general CLI pattern is:

```bash
python -m fms_dgt.public \
  --task-paths <path/to/task.yaml> \
  --config-path <path/to/config.yaml> \   # optional: override LM engine / model
  --num-outputs-to-generate <N> \
  --restart                               # start fresh, discarding previous output
```

Use `--help` for the full list of flags.

### Built-in examples

| Example                 | Task path                              | Default engine         |
| ----------------------- | -------------------------------------- | ---------------------- |
| Geography QA generation | `tasks/public/examples/qa/task.yaml`   | Ollama (`granite4:3b`) |
| QA difficulty rating    | `tasks/public/examples/rate/task.yaml` | Ollama (`granite4:3b`) |

### Supported LM engines

| Engine                                                                                | Config `type`  | Env vars required                       |
| ------------------------------------------------------------------------------------- | -------------- | --------------------------------------- |
| [Ollama](https://ollama.com/)                                                         | `ollama`       | —                                       |
| [OpenAI](https://platform.openai.com/)                                                | `openai`       | `OPENAI_API_KEY`                        |
| [Anthropic](https://www.anthropic.com/)                                               | `anthropic`    | `ANTHROPIC_API_KEY`                     |
| [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) | `azure-openai` | `AZURE_OPENAI_API_KEY`                  |
| [IBM watsonx](https://www.ibm.com/products/watsonx)                                   | `watsonx`      | `WATSONX_API_KEY`, `WATSONX_PROJECT_ID` |
| [vLLM](https://github.com/vllm-project/vllm)                                          | `vllm`         | —                                       |

## Observability

Every run writes structured telemetry to the `telemetry/` directory:

- `events.jsonl`: lifecycle events (run start/finish, task start/finish, epoch boundaries, rejected data points)
- `traces.jsonl`: one record per LLM call with provider, model, latency, semaphore wait time, and token usage

Both files rotate at 100 MB and rotated files older than 14 days are deleted automatically.

```bash
# Disable telemetry entirely
DGT_TELEMETRY_DISABLE=1 python -m fms_dgt.public ...

# Record prompts and completions in spans (sensitive — review before enabling)
DGT_TELEMETRY_RECORD_PAYLOADS=1 python -m fms_dgt.public ...
```

See [Observability](https://ibm.github.io/fms-dgt/concepts/observability/) in the docs for the full event and span reference.

## Studio

DiGiT Studio is a local run-monitoring UI that works like TensorBoard: it auto-launches alongside a generation run and stays up after the run finishes so you can keep browsing your data.

**Install:**

```bash
pip install -e ".[studio]"
```

Node.js (>=24) is required only if you want to rebuild Studio from source. The pre-built assets in `studio/dist/` are committed to the repo, so end users do not need Node.

**Launch with a generation run:**

```bash
python -m fms_dgt --studio \
  --task-paths ./tasks/public/examples/qa/task.yaml \
  --num-outputs-to-generate 20
# DiGiT Studio is running at http://localhost:4720
```

**Launch Studio standalone** (to browse existing runs without starting a new generation):

```bash
digit-studio start
digit-studio start --output-dir /path/to/output
digit-studio start --port 4720
```

**Stop a running instance:**

```bash
digit-studio stop
digit-studio stop --port 5000  # if running on a custom port
```

Studio reads `DGT_OUTPUT_DIR` from your environment by default. Override the port via `DIGIT_STUDIO_PORT` in `.env` or on the command line. Studio intentionally outlives the generation run so you can keep browsing data after it finishes.

## The Team

FMS-DGT is currently maintained by [Max Crouse](https://github.com/mvcrouse), [Kshitij Fadnis](https://github.com/kpfadnis), [Siva Sankalp Patel](https://github.com/sivasankalpp), and [Pavan Kapanipathi](https://github.com/pavan046).

## License

FMS-DGT has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.
