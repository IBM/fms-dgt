# Installation

Before you start, please make sure you have **Python 3.10+** available.

Building DGT from source lets you make changes to the code base. To install from source, clone the repository and install with the following commands:

```shell
git clone git@github.com:IBM/fms-dgt.git
cd fms-dgt
```

Now let's set up your virtual environment.

=== "Python venv"

    ```shell
    python3.10 -m venv ssdg_venv
    source ssdg_venv/bin/activate
    ```

    To install packages, we recommend starting off with the following

    ```bash
    pip install -e ".[all]"
    ```

    If you plan on contributing, install the pre-commit hooks to keep code formatting clean

    ```bash
    pip install pre-commit
    pre-commit install
    ```

=== "uv"

    ```shell
    uv sync --extra all
    ```

    If you plan on contributing, install the pre-commit hooks to keep code formatting clean

    ```bash
    uv pip install pre-commit
    uv pre-commit install
    ```

### Large Language Models (LLMs) Dependencies

DGT uses Large Language Models (LLMs) to generate synthetic data. Following LLM inference engines are supported:

| Engine                                                                             | Additional Installation    | Environment Variables                         | Supported APIs                  |
| ---------------------------------------------------------------------------------- | -------------------------- | --------------------------------------------- | ------------------------------- |
| [Ollama](https://docs.ollama.com/)                                                 | -                          | -                                             | `completion`, `chat_completion` |
| [WatsonX](https://cloud.ibm.com/apidocs/watsonx-ai)                                | -                          | `WATSONX_API_KEY=""`, `WATSONX_PROJECT_ID=""` | `completion`, `chat_completion` |
| [OpenAI](https://platform.openai.com/docs/api-reference/introduction)              | -                          | `OPENAI_API_KEY=""`                           | `completion`, `chat_completion` |
| [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/overview) | -                          | `AZURE_OPENAI_API_KEY=""`                     | `completion`, `chat_completion` |
| [Anthropic Claude](https://docs.claude.com/en/api/overview)                        | -                          | `ANTHROPIC_API_KEY=""`                        | `chat_completion`               |
| [vLLM](https://github.com/vllm-project/vllm)                                       | `pip install -e ".[vllm]"` | -                                             | `completion`, `chat_completion` |

Most of the aforementioned LLM inference engines use environment variables to specify configuration settings. You can either export those environment variables prior to every run or save them in `.env` file at base of `fms-dgt` repository directory.

!!! warning
vLLM dependencies [requires Linux OS and CUDA](https://docs.vllm.ai/en/latest/getting_started/installation.html#requirements).
