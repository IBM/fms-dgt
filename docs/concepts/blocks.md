# Blocks

A block is a single-operation component that a databuilder calls inside its `__call__` method. Blocks handle the computationally intensive work: LLM inference, validation, deduplication, field transformation. The databuilder code remains simple and sequential; all concurrency and parallelism lives inside the blocks.

## Why blocks exist

The simplest way to add an LLM call to a databuilder is to instantiate a client inline and call it directly. That works for one databuilder, but it creates problems at scale:

- **Duplicated initialization**: every databuilder that uses the same model creates its own connection and event loop.
- **No shared concurrency limits**: two databuilders calling the same API key can independently saturate the rate limit.
- **No reuse**: a validation function written for one databuilder cannot be used in another without copying code.

Blocks solve all three. A block is initialized once per databuilder, shared across all tasks that builder serves, and registered under a name that any databuilder can reference in its YAML config. The framework handles wiring: it reads the block definitions from YAML, instantiates them, and injects them as typed class attributes on the databuilder.

## Using a block from a databuilder

### Declaring a block

Declare the block as a class-level annotation in `generate.py`:

```python
class MyDataBuilder(GenerationDataBuilder):
    generator: LMProvider
    validator: MyValidatorBlock
```

The annotation name must match the `name` field in the corresponding YAML block entry.

### Configuring a block

Add the block to the `blocks` list in the builder YAML:

```yaml
blocks:
  - name: generator
    type: ollama
    model_id_or_path: granite4:3b
    temperature: 0.7
    max_tokens: 128
  - name: validator
    type: my_validator_type
    filter: true
```

The `type` field is the registered name of the block class (the string passed to `@register_block`).

### Calling a block

Blocks accept a list of dictionaries. Each dictionary represents one item to process. The block maps dictionary fields to its internal [`DATA_TYPE`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/base/block.py#L235) dataclass, runs its logic, and writes results back to the dictionaries:

```python
outputs = self.generator(
    [{"input": prompt, "reference": my_data_point}],
    method="chat_completion",
)
for output in outputs:
    result = output["result"]
    original = output["reference"]
```

Fields that are not part of the block's `DATA_TYPE` (like `reference` above) are passed through unchanged. This is the standard pattern for carrying data point context through a block call.

### Field remapping with `input_map` and `output_map`

When your dictionary uses different field names than the block's `DATA_TYPE` expects, use [`input_map` and `output_map`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/base/block.py#L65) to rename fields at the boundary. Maps can be set in the YAML config (applied to every call) or passed per call:

```yaml
# In YAML: rename "question" in your dict to "input" expected by the block
- name: dedup
  type: rouge_scorer
  filter: true
  threshold: 1.0
  input_map:
    question: input
```

```python
# At call time: rename "output" in the block result back to "answer" in your dict
results = self.generator(inputs, output_map={"result": "answer"})
```

The YAML convention is `{block_internal_field: your_field}`. DiGiT flips this internally for lookup, so the YAML reads as "the block's `input` field comes from my `question` field."

### Postprocessors

Blocks can also run as postprocessors after the main generation loop completes. Declare them in the `postprocessors` list by name:

```yaml
postprocessors:
  - name: dedup
```

The block must already be declared in `blocks`. Postprocessors run over the full accumulated dataset for each task once generation finishes, before final data is written.

## Built-in blocks

| Registry name                                                                     | Category     | What it does                                                                                                    |
| --------------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| `ollama`, `openai`, `azure-openai`, `anthropic`, `watsonx`, `vllm`, `vllm-server` | LM inference | LLM completion and chat completion via various backends                                                         |
| `rouge_scorer`                                                                    | Validator    | Filters records whose target field exceeds a ROUGE-L similarity threshold against existing data (deduplication) |
| `lm_judge`                                                                        | Validator    | Uses an LLM to judge whether a generated record meets a quality criterion                                       |
| `tool_call_validator`                                                             | Validator    | Validates tool-calling sequences in multi-turn data                                                             |
| `noop`                                                                            | Validator    | Always passes. Useful as a placeholder or for testing                                                           |
| `field_map`                                                                       | Utility      | Renames or copies fields between dictionary keys                                                                |
| `flatten_field`                                                                   | Utility      | Explodes a list-valued field into separate records                                                              |

## Implementing a new block

### Choosing a base class

All blocks inherit from [`Block`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/base/block.py#L89). For the most common case of filtering or validating records, inherit from [`ValidatorBlock`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/base/block.py#L421) instead. `ValidatorBlock` handles the iteration, filtering, and rejected-record storage loop for you. You only implement `_validate`.

### Define DATA_TYPE

`DATA_TYPE` declares the fields your block needs. The framework extracts these from incoming dictionaries automatically:

```python
@dataclass(kw_only=True)
class MyBlockData(ValidatorBlockData):
    input: str          # required (no default)
    threshold: float = 0.5  # optional (has default)
```

Fields with no default are required; the block will raise if they are missing from an input dictionary. Fields with defaults are optional.

### Register the block

```python
@register_block("my_namespace/my_block_name")
class MyBlock(ValidatorBlock):
    DATA_TYPE = MyBlockData

    def _validate(self, instance: MyBlockData) -> Tuple[bool, Optional[Dict]]:
        if len(instance.input) < 10:
            return False, {"reason": "Input too short."}
        return True, None
```

The string passed to `@register_block` is the `type` value used in YAML configs.

### Design guidance for block authors

**Keep blocks stateless.** Blocks are shared across tasks running in the same databuilder. Any instance variable you write during a call will be visible to other tasks. If you need per-task state, use the `store_names` mechanism to route data to per-task datastores, which the framework manages automatically.

**Do not put concurrency in `__call__`.** If your block needs to process items in parallel (for example, calling an external API for each item), implement the concurrency inside `execute` or `_validate`. The databuilder's `__call__` method should remain a simple sequential orchestration.

**Use `filter: true` and the rejected-record store.** When implementing a `ValidatorBlock`, always support `filter: true` and pass `store_names` through from the caller. This allows users to inspect rejected records without adding any extra code to your block:

```python
outputs = self.validator(
    [
        {
            "input": dp.text,
            "reference": dp,
            "store_names": self.get_block_store_names(
                block_name=self.validator.name,
                task_name=dp.task_name,
            ),
        }
        for dp in candidates
    ]
)
```

**Prefer narrow `DATA_TYPE` definitions.** Only declare fields your block actually uses. Narrow types make blocks easier to reuse across databuilders with different schemas, since callers can use `input_map` to satisfy the requirements from whatever fields they have.
