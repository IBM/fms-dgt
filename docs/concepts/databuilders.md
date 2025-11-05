# Data Builders

Data builders (see [here](https://github.ibm.com/DGT/fms-dgt/blob/main/fms_dgt/core/databuilders/simple/generate.py) for an example) contain the means by which our framework generates data. They consist of some number of _blocks_. Blocks are most often _generators_ or _validators_. Generators are, roughly speaking, things that take in inputs and generate some output (e.g., most often an LLM taking in a prompt and then returning a string). Correspondingly, validators are things that inspect an object and return True or False to signify whether that object is valid (e.g., validating the output of an LLM for well-formedness constraints in the case of code generation).

Each data builder is defined with a \_\_call\_\_ function. Importantly, the call function takes as input a list of the dataclass instances described above. This leads to an inherent composability of data builders, where the outputs of one data builder can be fed as the inputs to another (ideally leading to more code reuse across the repository).

```python
def __call__(
    self,
    request_idx: int,
    instruction_data: List[ExampleSdgData],
) -> Tuple[List[ExampleSdgData], int]:

    ... code goes here ...

    return outputs, discarded
```

As with task definitions, we aimed to be very non-prescriptive in how the \_\_call\_\_ functions are defined. That being said, we do encourage any computationally expensive calls that leverage batch processes (e.g., LLM calls) to go through blocks (with blocks for LLMs already being provided [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/core/blocks/generators/llm)).

An important aspect to keep in mind when defining a new data builder is the notion of _task parallelism_. That is, to make things more efficient, all tasks that can be executed by the same data builder will be run simultaneously. Thus, the inputs to the \_\_call\_\_ function will be a mixed list of instruction data (i.e., elements of the list can come from one of _N_ tasks). When doing things like combining instructions together (e.g., to serve as an ICL example to produce a new output), one has to make sure to keep track the provenance of the data.

Data builders are very configurable. An example of a configuration can be found [here](https://github.ibm.com/DGT/fms-dgt/blob/main/fms_dgt/core/databuilders/simple/simple.yaml) (see below for the relevant snippet).

```yaml
name: simple
blocks:
  - name: llm1
    type: rits
    base_url: https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01/v1
    model_id_or_path: mistralai/mixtral-8x7B-instruct-v0.1
    temperature: 0.0
    max_new_tokens: 512
    min_new_tokens: 1
  - name: val1
    type: rouge_scorer
    filter: true
    threshold: 1.0
metadata:
  version: 1.0
```

In this, the `blocks` field shows the default settings for a generator and a validator, respectively. This allows for trivial substitution of parameters like model types, LLM backends (e.g., `rits`, `watsonx`, `vllm`, `openai`) without having to change the underlying code.

> **IMPORTANT**
>
> Make sure the value specified for `model_id_or_path` field matches models available for the specified LLM backends (`rits`, `watsonx`, `vllm`, `openai`).
