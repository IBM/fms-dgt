# Start From Scratch

In this section, we'll go through the process of developing your very own data builder from scratch. We'll go through all of the steps from coding a data builder to defining a task file.

For the purposes of this exercise, we'll assume we want to generate data pertaining to a simple geography question-answering task.

## Prerequisites

To successfully run this, you will need to have completed the following:

1. Followed the [installation guide](../getting_started/installation.md) to set up your virtual environment
2. Read through the [Data Builders](../key_concepts/databuilders.md) and [Task](../key_concepts/tasks.md) sections

## Defining a Data Builder

We'll start by creating the base directory which will contain all of the code used to execute our SDG process. DGT supports both *generation* and *transformation* pipelines, but for this work, we'll be creating just a generation pipeline. First, create a directory `src/databuilders/generation/geography_qa`  


```bash
# from repo root
mkdir fms_dgt/research/databuilders/geography_qa
```

In that directory, we will first add the data-specifying objects that our code will operate on. In DGT, we assume there to be a class for the overall `task` as well as a class for an individual data instance. Since we're doing a simple question-answering task, we don't need to instantiate much. Add the following code to a `fms_dgt/research/databuilders/geography_qa/task.py` file (see `NOTE:` comments for explanation)

```python
# Standard
from dataclasses import dataclass
from typing import Any

# Local
from fms_dgt.base.task import SdgData, SdgTask


# NOTE: this class holds the information needed for a single SDG example
@dataclass(kw_only=True)
class GeographySdgData(SdgData):
    """This class is intended to hold the seed / machine generated instruction data"""

    question: str
    answer: str


# NOTE: this class holds the information needed for the overall geography QA task
class GeographySdgTask(SdgTask):
    """This class is intended to hold general task information"""

    # We must always specify both the type of data that will be accepted as well as the type of data that will be generated
    INPUT_DATA_TYPE = GeographySdgData
    OUTPUT_DATA_TYPE = GeographySdgData

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
```

With the task file defined, we can now work on the code that will be used to actually execute data generation. Create a `fms_dgt/research/databuilders/geography_qa/generate.py` file with the following code

```python
# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.research.databuilders.geography_qa.task import (
    GeographySdgData,
    GeographySdgTask,
)


_LLM_PROMPT = """You are a geography question-answering data generator. Your task is to come up with geography-related question-answer pairs that can be used to train a question-answering system. 

Here are some examples:

"""


# NOTE: we register the data builder with the below decorator so that we can reference it in an input data file later on
@register_data_builder("geography_qa")
class GeographyQADataBuilder(DataBuilder):
    """Geography QA data builder"""

    TASK_TYPE: GeographySdgTask = GeographySdgTask

    # NOTE: llm1 is the language model that we will use to produce the synthetic examples
    llm1: LMProvider

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        request_idx: int,
        geo_data: List[GeographySdgData],
    ) -> List[GeographySdgData]:

        llm_inputs: List[Dict] = []
        for gd in geo_data:
            # example of how to form an argument to the LLM generator
            prompt = _LLM_PROMPT + f"Q: {gd.question}\n\nA: {gd.answer}\n\nQ: "
            inp = {"input": prompt, "gen_kwargs": {"stop": ["Q:"]}, "data": gd}
            llm_inputs.append(inp)

        # NOTE: what blocks do (e.g., llm1) is iterate over each row of the input list and extract out elements of LMBlockData from each input
        llm_outputs = self.llm1(llm_inputs)

        outputs = []
        for llm_output in llm_outputs:
            # NOTE: the output of data generation will be found in "result" field (as indicated in LMBlockData) and the original data will be found in the "data" field
            llm_response = llm_output["result"]
            orig_data = llm_output["data"]
            qa_pair = llm_response.split("A:")
            if len(qa_pair) == 2:
                question, answer = qa_pair
                task_name = orig_data.task_name
                new_qa_pair = GeographySdgData(
                    question=question.strip(),
                    answer=answer.strip(),
                    task_name=task_name,
                )
                outputs.append(new_qa_pair)

        return outputs
```

Lastly, we must define a config file in this directory that let's us define information about the data builder. Create a `fms_dgt/research/databuilders/geography_qa/geo_qa.yaml`

```yaml
name: geography_qa
blocks:
  - name: llm1
    type: rits
    base_url: https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01/v1
    model_id_or_path: mistralai/mixtral-8x7B-instruct-v0.1
    temperature: 0.0
    max_new_tokens: 512
    min_new_tokens: 1
metadata:
  version: 1.0
```

## Creating a Task

Now that we have code for our SDG in place, we can define a simple task file that can be processed by this code. All SDG begins with a `qna.yaml` task file. To begin, create the directory `tasks/research/geography_qa`

```bash
# from repo root
mkdir tasks/research/geography_qa
```

Within that directory, add a task.yaml file with the following contents

```yaml
task_name: geography_qa
created_by: IBM
data_builder: geography_qa
seed_examples:
  - answer: "Mount Everest"
    question: "What is the name of the tallest mountain in the world?"
  - answer: "Atlantic, Pacific, Indian, Arctic, and the Antarctic"
    question: "What are the names of the five oceans of the world?"
task_description: "A task for geography question-answering"
```

## Running your SDG Code

With `fms_dgt/research/databuilders/geography_qa` and `tasks/research/geography_qa/task.yaml` defined, this can now be run by executing (from the base of the repo)

`python -m fms_dgt.research --task-path ./tasks/research/geography_qa/task.yaml`

Once this completes, you should be able to find the output of your system at

`output/geography_qa/data.jsonl`