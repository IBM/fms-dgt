# Changing the Language Model Engine

DiGiT provides built-in support for multiple language model (LM) engines through the `LMProvider` block: Ollama, WatsonX, OpenAI, Azure OpenAI, vLLM, and Anthropic. As described in the [architecture overview](../concepts/architecture.md), blocks are single-operation components initialized once per databuilder. Switching LM engines requires only a change to the databuilder YAML configuration.

The databuilder YAML is located in the same directory as `generate.py`. For the misconceptions generation databuilder built in [Building a Generation Databuilder](../tutorials/generate_data.md), the config is at:

`fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml`

The relevant section is:

```{.yaml .no-copy title="fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml" hl_lines="3 4"}
blocks:
  - name: generator # (1)!
    type: ollama    # (2)!
    model_id_or_path: granite4:3b # (3)!
    temperature: 0.7
    max_tokens: 128
    num_ctx: 4096
```

1. The name must match the class-level annotation in `generate.py` (`generator: LMProvider`).
2. The LM engine. Supported values: `ollama`, `watsonx`, `openai`, `azure-openai`, `anthropic`, `vllm`, `vllm-remote`.
3. The model identifier. The correct format depends on the engine.

## Switching models within Ollama

To use a different locally-hosted model, update `model_id_or_path` and pull the model first:

```bash
ollama pull llama3.2:3b
```

Then update the YAML:

```{.yaml title="fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml" hl_lines="4"}
blocks:
  - name: generator
    type: ollama
    model_id_or_path: llama3.2:3b
    temperature: 0.7
    max_tokens: 128
    num_ctx: 4096
```

`llama3.2:3b` is a good general-purpose alternative. It is licensed under the Llama 3.2 Community License, which explicitly permits synthetic data generation.

## Switching to a cloud provider

To switch from Ollama to a cloud-hosted model, change `type` and `model_id_or_path` and set the required environment variables. The generation parameters (`temperature`, `max_tokens`) carry over unchanged.

???+ warning "Review your provider's terms before using outputs for training"
    Cloud API providers may use API request and response data to improve their own models unless you have an enterprise agreement that opts out of this. WatsonX, OpenAI, and Anthropic all offer enterprise or API-tier contracts that disable training data collection. Review your agreement before using cloud-generated outputs in a training dataset at scale.

=== "WatsonX"

    ```{.yaml title="fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml" hl_lines="3 4"}
    blocks:
      - name: generator
        type: watsonx
        model_id_or_path: meta-llama/llama-3-3-70b-instruct
        temperature: 0.7
        max_new_tokens: 128
    ```

    Set the following environment variables before running:

    | Variable | Required | Description |
    |---|---|---|
    | `WATSONX_PROJECT_ID` | Yes | Your WatsonX project ID |
    | `WATSONX_API_KEY` | Yes | Your IBM Cloud API key |
    | `WATSONX_API_URL` | No | Defaults to `https://us-south.ml.cloud.ibm.com` |

    ```bash
    export WATSONX_PROJECT_ID=your-project-id
    export WATSONX_API_KEY=your-api-key
    ```

    WatsonX uses `max_new_tokens` instead of `max_tokens`. All other generation parameters are the same.

=== "OpenAI"

    ```{.yaml title="fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml" hl_lines="3 4"}
    blocks:
      - name: generator
        type: openai
        model_id_or_path: gpt-4o-mini
        temperature: 0.7
        max_tokens: 128
    ```

    Set the following environment variable before running:

    | Variable | Required | Description |
    |---|---|---|
    | `OPENAI_API_KEY` | Yes | Your OpenAI API key |

    ```bash
    export OPENAI_API_KEY=your-api-key
    ```

=== "Anthropic"

    ```{.yaml title="fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml" hl_lines="3 4"}
    blocks:
      - name: generator
        type: anthropic
        model_id_or_path: claude-3-5-haiku-20241022
        temperature: 0.7
        max_tokens: 128
    ```

    Set the following environment variable before running:

    | Variable | Required | Description |
    |---|---|---|
    | `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |

    ```bash
    export ANTHROPIC_API_KEY=your-api-key
    ```

    The Anthropic provider processes one request at a time. For large runs, expect lower throughput compared to Ollama or OpenAI.

## Running after switching

The run command is the same regardless of the engine:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/misconceptions/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

## Next steps

- To load seed examples from an external file instead of the task YAML, see [Loading Seed Examples from a File](loading_seed_examples_from_file.md).
- To add a validator that filters low-quality outputs, see [Creating a Validator](creating_validator.md).
