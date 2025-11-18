# Data Transformation

In this example, we'll walk through the process of designing a custom transformation databuilder using DGT. Specifically, we'll demonstrate adding custom chain-of-thought (CoT) to the Grade School Math (GSM) dataset, which is useful for tasks that benefit from intermediate reasoning steps.

### What Are Transformation Data Builders?

While generation databuilders focuses on expanding a small set of examples into a larger dataset, transformation databuilders are designed to modify existing datasets. These modifications can include:

- Format conversions (e.g., from raw text to structured JSON)
- Quality improvements (e.g., filtering noisy samples)
- Anonymization (e.g., removing sensitive information)
- Task adaptation (e.g., converting slot-filling data into instruction-tuning format)

DGT supports transformation tasks using the same infrastructure as generation tasks—including LLM integration, data loading, and output handling. The key difference is that transformation builders typically process each input example once, applying a defined transformation, rather than iterating over again and again till stopping criteria is met.

### Set Up Your Transformation Databuilder

To begin, we'll create the base directory for our transformation databuilder. From the root of your repository, run:

```bash
# from root of your repository
mkdir -p fms_dgt/public/databuilders/test/gsm_cot
```

This directory will contain the logic for transforming Grade School Math (GSM) dataset from OpenAI into a chain-of-thought format.

### Define the Task and Data Classes

Next, we will define the data classes that represent the input and output formats. GSM examples typically consist of a question and an answer. For our transformation, we will retain these fields under new names, making minor modifications to the "answer" field to remove any existing reasoning steps. Additionally, we will introduce a new field called "thought," which will contain custom-generated reasoning steps.

Create a new file at:
`fms_dgt/public/databuilders/test/gsm_cot/task.py`

Add the following code:

```python
# Standard
from dataclasses import dataclass

# Local
from fms_dgt.base.data_objects import DataPoint
from fms_dgt.base.task import TransformationTask


@dataclass(kw_only=True)
class GsmData(DataPoint):
    question: str
    answer: str

    def __post_init__(self):
        self.question = self.question.strip()
        # NOTE: GSM8k on huggingface already has the answer included with an explanation, we'll strip out the explanation and just keep the number
        self.answer = self.answer.split("####")[-1].strip()


@dataclass(kw_only=True)
class GsmCotData(DataPoint):
    input: str
    output: str
    thought: str


class GsmCotTask(TransformationTask):
    # We must always specify both the type of data i.e. pre-transform (INPUT_DATA_TYPE) and post-transform (OUTPUT_DATA_TYPE)

    INPUT_DATA_TYPE = GsmData
    OUTPUT_DATA_TYPE = GsmCotData
```

### Implement the Data Transformation Logic

Now that we've defined our task and data classes, it's time to implement the actual data transformation logic. This is where we use a language model to synthesize new chain-of-thought (COT) reasoning steps for each question answer pair in the original dataset.

Create a new file at: `fms_dgt/public/databuilders/test/gsm_cot/generate.py` and add the following code to define your custom transformation databuilder:

```{.python title="fms_dgt/public/databuilders/test/gsm_cot/generate.py"}
# Standard
from typing import Any, Iterable, List

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.public.databuilders.test.gsm_cot.task import (
    GsmData,
    GsmCotData,
    GsmCotTask,
)
from fms_dgt.base.prompt import JinjaPromptTemplate


@register_data_builder("public/test/gsm_cot")
class GsmCotDataBuilder(TransformationDataBuilder):
    """Class for GSM chain-of-thought task"""

    TASK_TYPE: GsmCotTask = GsmCotTask

    lm: LMProvider

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # configure prompt
        self._prompt_template = JinjaPromptTemplate(
            template=(
                "You are an intelligent tutoring assistant that helps students with math homework."
                " Given a question and its answer, explain how to solve the question step-by-step to achieve the answer."
                ' When you are explaining the answer to the student, please preface your explanation with "Let\'s think step-by-step."\n\n'
                "Question: {{ question }}\nAnswer: {{ answer }}\nExplanation: "
            )
        )

    def __call__(
        self,
        data_points: List[GsmData],
    ) -> Iterable[GsmCotData]:
        # Build LM inputs
        llm_inputs = [
            {
                "input": self._prompt_template.encode(
                    render_dict={"question": data_point.question, "answer": data_point.answer}
                ),
                "stop": ["Question:"],
                "source": data_point,
            }
            for data_point in data_points
        ]

        # Invoke LM
        lm_outputs = self.lm(llm_inputs)

        # Process LM outputs
        for lm_output in lm_outputs:
            source_data_point: GsmData = lm_output["source"]
            # NOTE: we don't do any validation of the generated 'thought', however, in general that would be a good idea
            thought = lm_output["result"].strip()
            # NOTE: here we yield from the data builder so that the data is saved immediately in intermediate datastore
            yield GsmCotData(
                task_name=source_data_point.task_name,
                is_seed=False,
                input=source_data_point.question,
                output=source_data_point.answer,
                thought=thought,
            )
```

### Define the Builder Configuration

Next step is to define a configuration file that describes how the databuilder should operate. This file provides metadata, specifies the components involved in transformation. Create a new file at: `fms_dgt/public/databuilders/test/gsm_cot/gsm_cot.yaml` and add the following contents:

```{.yaml title="fms_dgt/public/databuilders/test/gsm_cot/gsm_cot.yaml"}
######################################################
#                   MANDATORY FIELDS
######################################################
name: public/test/gsm_cot

######################################################
#                   RESERVED FIELDS
######################################################
blocks:
  # Language model connector
  - name: lm
    type: ollama
    model_id_or_path: mistral-small3.2
    temperature: 0.0
    max_tokens: 512
metadata:
  version: 1.0
```

This configuration sets up the transformation databuilder using a language model (mistral-small3.2). The name field uniquely identifies your databuilder, and the metadata section can be extended as needed.

### Create a Task File

With the databuilder code and configuration in place, the final step is to define a task file that will drive the SDG process. Every SDG workflow begins with a task.yaml file that specifies the task name, description, data, and the associated databuilder. Start by creating the task directory:

```bash
# from repo root
mkdir -p tasks/public/test/gsm_cot
```

Inside this directory, create a file named task.yaml with the following contents:

```{.yaml title="tasks/public/test/gsm_cot/task.yaml" hl_lines="12-14"}
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: public/test/gsm_cot
task_description: A task for adding custom Chain-of-Thought (COT) to Grade School Math (GSM) dataset from OpenAI
created_by: IBM

data_builder: public/test/gsm_cot
######################################################
#                   RESERVED FIELDS
######################################################
data:
  type: default
  data_path: ${DGT_DATA_DIR}/public/test/gsm8k_cot/train.jsonl
```

!!! warning

    You will need to have local copy of Grade School Math (GSM) dataset from OpenAI in `data/public/test/gsm8k_cot` directory. You can download it from OpenAI's Github respository with following command.

    ```bash
    wget https://raw.githubusercontent.com/openai/grade-school-math/refs/heads/master/grade_school_math/data/train.jsonl
    ```

### Running your Code

From the root of your repository, execute the following command:

```bash
python -m fms_dgt.research --task-path ./tasks/public/test/gsm_cot/task.yaml
```

Once the process completes, the generated data will be available at:

`output/public/test/gsm_cot/final_data.jsonl`
