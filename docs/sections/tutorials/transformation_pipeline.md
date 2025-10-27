# Transformation Pipelines

This tutorial will cover how to design your own transformation pipeline. In this tutorial, we will walk through the process of turning GSM8K into a chain-of-thought dataset.

## Prerequisites

To successfully run this, you will need to have completed the following:

1. Successfully completed the [Starting From Scratch](./from_scratch.md) tutorial
2. Read through the [Databuilders](../key_concepts/databuilders.md) section

## Transformation Data Builders

In addition to synthetic data generation where one uses a small number of examples to generate a much larger set of examples, DGT also has built-in support for data transformation. A transformation task involves taking an existing dataset (or set of datasets) and converting it to a new format. For instance, one might take an older dataset produced initially for a slot-filling task and convert it to an instruction-tuning dataset for API function calls. Often this will require the same infrastructure that a data generation task requires (e.g., ability to make LLM calls, storing and loading data from the same sources, etc.).

In DGT, transformation pipelines are treated as data builders. The main distinguishing characteristic of a transformation-specific data builder is that its default behavior will be to iterate over an input dataset only once and apply its transformation (as opposed to a general data builder, where generation iterates over seed / machine-generated examples until a target number of outputs is generated).

## Defining a Transformation Data Builder

As in the other tutorials, we'll first create our base directory for our transformation pipeline. Starting from the base of the repo, create a directory `fms_dgt/research/databuilders/gsm8k_cot`  

```bash
# from repo root
mkdir fms_dgt/research/databuilders/gsm8k_cot
```

In that directory, we add our data-specifying objects. These are very simple in our case, as GSM8K originally only loads with `question` and `answer` fields. Add the following code to a `fms_dgt/research/databuilders/gsm8k_cot/task.py` file (see `NOTE:` comments for explanation)

```python
# Standard
from dataclasses import dataclass

# Local
from fms_dgt.base.task import SdgData, TransformTask


@dataclass(kw_only=True)
class Gsm8kInitialTransformData(SdgData):

    question: str
    answer: str

    def __post_init__(self):
        self.question = self.question.strip()
        # NOTE: GSM8k on huggingface already has the answer included with an explanation, we'll strip out the explanation and just keep the number
        self.answer = self.answer.split("####")[-1].strip()


@dataclass(kw_only=True)
class Gsm8kCotTransformData(SdgData):

    input: str
    output: str
    thought: str


# NOTE: we have a pre-transform and post-transform data type here for convenience
class Gsm8kCotTransformTask(TransformTask):

    INPUT_DATA_TYPE = Gsm8kInitialTransformData
    OUTPUT_DATA_TYPE = Gsm8kCotTransformData
```


Now that the task has been defined, we'll add the data transformation code. Create a `fms_dgt/research/databuilders/gsm8k_cot/generate.py` file with the following code

```python
# Standard
from typing import Any, Iterable, List

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.research.databuilders.gsm8k_cot.task import (
    Gsm8kInitialTransformData,
    Gsm8kCotTransformData,
    Gsm8kCotTransformTask,
)

_PROMPT = """You are an intelligent tutoring assistant that helps students with math homework. Given a question and its answer, explain how to solve the question step-by-step to achieve the answer. When you are explaining the answer to the student, please preface your explanation with "Let's think step-by-step."

Here are some examples:

Question: { { question } }
Answer: { { answer } }
Explanation: Let's think step-by-step. 
""".strip()


# NOTE: importantly, transformation data builders are STRONGLY ENCOURAGED to inherit from TransformationDataBuilder, rather than from BaseDataBuilder. This is because the default behavior of TransformationDataBuilder is much more suited to transformation tasks
@register_data_builder("gsm8k_cot")
class Gsm8kCotTransformDataBuilder(TransformationDataBuilder):
    """Class for GSM8K chain-of-thought task"""

    TASK_TYPE: Gsm8kCotTransformTask = Gsm8kCotTransformTask

    # NOTE: this is the same llm1 as in our config yaml
    llm1: LMProvider

    # NOTE: this can be removed, but we've kept it for those who will copy-paste this as a template
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __call__(
        self,
        qa_data: List[Gsm8kInitialTransformData],
    ) -> Iterable[Gsm8kCotTransformData]:

        llm_inputs = []
        for qa_pair in tqdm(qa_data, desc="GSM8K Transformation"):
            # NOTE: since we have obtained this from huggingface, the actual answer is marked by "... #### <number>", so we'll extract that here

            new_inp = _PROMPT.replace("{ { question } }", qa_pair.question).replace(
                "{ { answer } }", qa_pair.answer
            )
            llm_inputs.append(
                {"input": new_inp, "stop": ["Question:"], "data": qa_pair}
            )

        # NOTE: unlike in the other tutorials, we have provided 'arg_fields' / 'kwarg_fields' / 'result_field' in the data builder's config, thus we do not need to specify them here
        llm_outputs = self.llm1.generate(llm_inputs)

        for output in llm_outputs:
            orig_qa: Gsm8kInitialTransformData = output["data"]
            # NOTE: we don't do any validation of the generated 'thought', however, in general that would be a good idea
            thought = output["output"].strip()
            # NOTE: here we yield from the data builder so that the data is saved immediately
            yield Gsm8kCotTransformData(
                **{
                    "task_name": orig_qa.task_name,
                    "input": orig_qa.question,
                    "output": orig_qa.answer,
                    "thought": thought,
                }
            )
```

Lastly, we define a config file (as was done with the generation data builder). Create a `fms_dgt/research/databuilders/gsm8k_cot/transform_gsm8k_cot.yaml`

```yaml
name: gsm8k_cot
blocks:
  - name: llm1
    type: rits
    base_url: https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01/v1
    model_id_or_path: mistralai/mixtral-8x7B-instruct-v0.1
    temperature: 0.0
    max_new_tokens: 512
    min_new_tokens: 1
    model_id_or_path: mistralai/mixtral-8x7b-instruct-v01
metadata:
  version: 1.0
```

## Creating a Task

Our transformation code is complete, so now we can define a task file that specifies our input dataset. Create the directory `fms_dgt/research/gsm8k_cot`

```bash
# from repo root
mkdir fms_dgt/research/gsm8k_cot
```

Within that directory, add a task.yaml file with the following contents

```yaml
task_name: gsm8k_cot
created_by: IBM
data_builder: gsm8k_cot
seed_datastore:
  type: default
  # NOTE: data_path will take either a string or a list of strings to be treated as *args to datasets.load_dataset(*args)
  data_path: ['gsm8k', 'main']
task_description: Transformation of gsm8k into chain-of-thought dataset
```

## Running your SDG Code

With `fms_dgt/research/databuilders/gsm8k_cot` and `tasks/research/gsm8k_cot/task.yaml` defined, this can now be run by executing (from the base of the repo)

`python -m fms_dgt.research --task-path ./tasks/research/gsm8k_cot/task.yaml`

Once this completes, you should be able to find the output of your system at

`output/gsm8k_cot/data.jsonl`