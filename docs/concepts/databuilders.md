# Data Builders

[Tasks](./tasks.md) define what data to consume and produce, while Data Builders define how that data is produced. A Data Builder is a class responsible for implementing the logic that generates or transforms data. Each Data Builder exposes a **call** method, which operates on:

- Input: Accepts a list of dataclass instances.
- Output: Returns a list of dataclass instances.

This design enables Data Builders to be reused across multiple tasks, as long as those tasks operate on the same type of dataclass. By decoupling data generation logic from task design, the framework promotes modularity and flexibility.

There is built-in support for two prominent patterns data processing patterns viz. 1) Generation and 2) Transformation

|                      | **Generation**                                                     | **Transformation**                               |
| -------------------- | ------------------------------------------------------------------ | ------------------------------------------------ |
| **Purpose**          | Create new synthetic data instances from scratch or seed examples  | Modify or convert existing data into a new form  |
| **Input**            | Often starts with empty list or minimal seed data                  | Requires existing data instances                 |
| **Output**           | Newly generated synthetic data                                     | Transformed version of input data                |
| **Typical Use Case** | Data augmentation, synthetic dataset creation                      | Translation, normalization, feature extraction   |
| **Dependency**       | May rely on generative models or iterative refinement              | Depends on transformation logic or mapping rules |
| **Processing Style** | **Iterative**: continues until target number of datapoints reached | **Single-pass**: processes each input once       |
| **Example**          | Generate synthetic sentences from seed text                        | Translate English sentences to French            |

### Effeciency via Blocks

We designed the framework to be non-prescriptive, allowing flexible implementation of the `__call__` function. However, we strongly encourage the use of blocks for computationally intensive operations, such as batch processing with LLMs (predefined blocks for LLMs are available [here](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/core/blocks/llm)).

Adding a block to a data builder is straightforward. The framework automatically handles block initialization. To include a block, simply declare a class variable in the data builder class and configure it in the accompanying YAML file by adding an entry to the blocks list. For instance, the [GeographyQA data builder](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/examples/qa) uses a `generator` block:

```{.python .no-copy title="fms_dgt/public/databuilders/examples/qa/generate.py" hl_lines="8"}
@register_data_builder("public/examples/geography_qa")
class GeographyQADataBuilder(GenerationDataBuilder):
    """Geography QA data builder"""

    TASK_TYPE: GeographyQATask = GeographyQATask

    # NOTE: generator is the language model that we will use to produce the synthetic examples
    generator: LMProvider

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    .
```

And the corresponding YAML configuration:

```{.yaml .no-copy title="fms_dgt/public/databuilders/examples/qa/qa.yaml" hl_lines="10-15"}
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
    model_id_or_path: mistral-small3.2
    temperature: 0.0
    max_tokens: 128
  .
```

??? warning
Make sure the value specified for `model_id_or_path` field matches models available for the specified LLM backends.

### Effeciency via Parallelism

Keep in mind that data builders can handle task parallelism, meaning multiple tasks can be processed simultaneously by the same data builder. This results in a mixed list of input data in the `__call__` function. When combining data from multiple tasks (e.g., for instruction following), ensure you track the origin of each data point.

One can opt out of task parallelism by overriding the `call_with_task_list` method for `Generation` pattern, as shown below:

```{.python}
def call_with_task_list(self, tasks: List[Task], *args, **kwargs) -> Iterable[DataPoint]:
  for task in tasks:
    # Tracks number of attempts made
    request_idx = 0

    data_pool = task.get_batch_examples()
    while data_pool:
      yield self(*[request_idx, data_pool], **dict())

      # Fetch next batch of examples
      data_pool = task.get_batch_examples()

      # Update request index
      request_idx += 1
```
