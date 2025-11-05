# Changing the Language Model (LM) Engine

DGT offers built-in support for over five different language model (LM) engines (WatsonX, OpenAI, Azure OpenAI, vLLM, Ollama, and Anthropic) through the `LMProvider` block. As described in this [section](../concepts/architecture.md), blocks are single-operation components that are initialized once per databuilder. This design makes it easy to switch between LM engines by simply updating the databuilder YAML configuration.

Typically, the YAML file for a databuilder is located in the same directory as `generate.py`, which serves as the entry point for that databuilder. Let’s revisit our earlier example of generating geography-based question-answer pairs.

The corresponding YAML file can be found at:
`fms_dgt/public/databuilders/examples/qa/qa.yaml`

Here’s a closer look at its contents:

```{.yaml .no-copy title="fms_dgt/public/databuilders/examples/qa/qa.yaml" hl_lines="11 12 13" }
######################################################
#                   MANDATORY FIELDS
######################################################
name: public/examples/geography_qa

######################################################
#                   RESERVED FIELDS
######################################################
blocks:
  # Language model connector
  - name: generator # (1)!
    type: ollama # (2)!
    model_id_or_path: mistral-small3.2 # (3)!
    temperature: 0.0
    max_tokens: 128
  # Built-in Rouge-L score based deduplicator
  - name: dedup
    type: rouge_scorer
    filter: true
    threshold: 1.0
    input_map:
      question: input
postprocessors:
  # Post-processors operate on all data points simultaneously
  - name: dedup
metadata:
  version: 1.0

```

1. Identifier for the LM block.
2. Specifies the LM engine. Supported values include watsonx, openai, azure-openai, anthropic, vllm, vllm-remote, and ollama.
3. The model identifier or path, which varies depending on the selected LM engine. Refer to the documentation for the specific engine to determine the correct value.

Let's try via changing the model used from `mistral-small3.2` to `gemma3:1b` as follows

```{.yaml title="fms_dgt/public/databuilders/examples/qa/qa.yaml" hl_lines="13" }
######################################################
#                   MANDATORY FIELDS
######################################################
name: public/examples/geography_qa

######################################################
#                   RESERVED FIELDS
######################################################
blocks:
  # Language model connector
  - name: generator
    type: ollama
    model_id_or_path: gemma3:1b
    temperature: 0.0
    max_tokens: 128
  # Built-in Rouge-L score based deduplicator
  - name: dedup
    type: rouge_scorer
    filter: true
    threshold: 1.0
    input_map:
      question: input
postprocessors:
  # Post-processors operate on all data points simultaneously
  - name: dedup
metadata:
  version: 1.0
```
