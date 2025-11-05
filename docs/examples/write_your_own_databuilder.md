# Write your own generation databuilder

In this section, we'll go through the process of developing your very own data builder from scratch. We'll go through all of the steps from coding a data builder to defining a task file.

For the purposes of this exercise, we'll assume we want to generate data pertaining to a simple geography question-answering task.

## Defining a Data Builder

We'll start by creating the base directory which will contain all of the code used to execute our SDG process. DGT supports both _generation_ and _transformation_ databuilders, but for this work, we'll be creating just a generation databuilder. First, create a directory `fms_dgt/public/databuilders/test/geography_qa`

```bash
# from the root of the cloned repository
mkdir -p fms_dgt/public/databuilders/test/geography_qa
```

In that directory, we will first add the data-specifying objects that our code will operate on. In DGT, we assume there to be a class for the overall `task` as well as a class for an individual data instance. Since we're doing a simple question-answering task, we don't need to instantiate much. Add the following code to a `fms_dgt/public/databuilders/test/geography_qa/task.py` file

```{.python title="fms_dgt/public/databuilders/test/geography_qa/task.py"}
# Standard
from dataclasses import dataclass
from typing import Any

# Local
from fms_dgt.base.data_objects import DataPoint
from fms_dgt.base.task import GenerationTask


@dataclass(kw_only=True)
class GeographyQAData(DataPoint):
    """This class is intended to hold the seed / machine generated instruction data"""

    question: str
    answer: str


class GeographyQATask(GenerationTask):
    """
    In this example, we wish to create 50 geographical question answer pairs. Hence, we choose to extend from `GenerationTask`.
    """

    # We must always specify both the type of data that will be accepted as well as the type of data that will be generated
    # For our example, we will be providing some seed examples to large languge model to create new synthetic data in the similar format.
    # Therefore, our `INPUT_DATA_TYPE` and `OUTPUT_DATA_TYPE` are identical.
    #
    # CAUTION: Be careful when you use different `INPUT_DATA_TYPE` and `OUTPUT_DATA_TYPE`. By default, `GenerationTask` type task are expected
    # to keep looping over a mixture of seed and synthetic data till it produces requested number of synthetic data points.

    INPUT_DATA_TYPE = GeographyQAData
    OUTPUT_DATA_TYPE = GeographyQAData

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Additional arguments specified in this constructor can be set using associated `task.yaml`
        """
        super().__init__(*args, **kwargs)

    def instantiate_input_example(self, **kwargs) -> GeographyQAData:
        """
        This helper method is called automatically on each seed data point provided to the generation task.

        By default, it will try to instantiate object of `INPUT_DATA_TYPE` dataclass from each loaded data point. But, the
        databuilder developer has ability to change the default behavior via overriding this method.

        Returns:
            GeographyQAData: object of `GeographyQAData` from seed data
        """
        return GeographyQAData(
            task_name=self.name,
            is_seed=True,
            question=kwargs.get("question"),
            answer=kwargs.get("answer"),
        )
```

With the task file defined, we can now work on the code that will be used to actually execute data generation. Create a `fms_dgt/research/databuilders/geography_qa/generate.py` file with the following code

```{.python title="fms_dgt/public/databuilders/test/geography_qa/generate.py"}
# Standard
from typing import Any, Dict, List
import random

# Local
from fms_dgt.base.databuilder import GenerationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.public.databuilders.test.geography_qa.task import (
    GeographyQAData,
    GeographyQATask,
)

# NOTE: we register the data builder with the below decorator so that we can reference it in an input data file later on
@register_data_builder("public/test/geography_qa")
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

        self._prompt_template = (
            "You are a geography question-answering data generator."
            " Your task is to come up with geography-related question-answer pairs that can be used to train a question-answering system."
            "\n\nHere are few examples:\n\n"
        )

    def __call__(
        self,
        request_idx: int,
        seed_data: List[GeographyQAData],
    ) -> List[GeographyQAData]:
        # Build generator inputs
        generator_inputs: List[Dict] = []
        for _ in range(len(seed_data)):
            # Randomly select in-context learning (icl) examples
            icl_examples = random.choices(seed_data, k=3)

            # Build prompt
            prompt = f'{self._prompt_template}{"\n\n".join([f"Question: {icl_example.question}\nAnswer: {icl_example.answer}" for icl_example in icl_examples])}\n\nNow generate a single different question-answer pair in the similar format.\n\nQuestion: '


            # Build generator inputs
            # input (str | List[Dict[str, Any]]): (Reserved field) prompt to be passed to `/completion` endpoint or messages to be passed to `/chat/completion` endpoint
            # gen_kwargs (Optional[Dict[str, Any]]): (Reserved field) Additional generation specific parameters to be passed to `/completion` or `/chat/completion` endpoint
            # reference (Optional[Any]): We recommend passing data used to build prompt for future use. DiGiT returns all non-reserved field in output from a block.
            generator_inputs.append(
                {
                    "input": prompt,
                    "reference": icl_examples,
                }
            )

        # Execute block
        # LMProvider block is optimized to perform asynchronous invocation of `/completion` or `/chat/completion` endpoint to enable batch processing.
        generator_outputs = self.generator(generator_inputs)

        # Process outputs from block
        outputs = []
        for generator_output in generator_outputs:
            # Extract icl examples passed to LMProvider block
            icl_examples = generator_output["reference"]

            # LMProvider block return output from `/completion` or `/chat/completion` endpoint in "result" field.
            question_answer_pair = generator_output["result"].split("Answer:")

            # Minimal check to guarantee well formed response
            if len(question_answer_pair) == 2:
                # For well-formed response, build "GeographyQAData" objects
                # As you can observed, having "reference" (icl examples) is handy to able to set correct "task_name"
                outputs.append(
                    GeographyQAData(
                        task_name=icl_examples[0].task_name,
                        is_seed=False,
                        question=question_answer_pair[0].strip().rstrip("\n"),
                        answer=question_answer_pair[1].strip().rstrip("\n"),
                    )
                )

        # Return generated synthetic data points
        return outputs
```

Lastly, we must define a config file in this directory that let's us define information about the data builder. Create a `fms_dgt/public/databuilders/test/geography_qa/geography_qa.yaml`

```{.yaml title="fms_dgt/public/databuilders/test/geography_qa/geography_qa.yaml"}
######################################################
#                   MANDATORY FIELDS
######################################################
name: public/test/geography_qa

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

## Creating a Task

Now that we have code for our SDG in place, we can define a simple task file that can be processed by this code. All SDG begins with a `task.yaml` task file. To begin, create the directory `tasks/public/test/geography_qa`

```bash
# from repo root
mkdir -p tasks/public/test/geography_qa
```

Within that directory, add a task.yaml file with the following contents

```yaml
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: public/test/geography_qa
task_description: A task for geography question-answering
created_by: IBM

data_builder: public/test/geography_qa

######################################################
#                   RESERVED FIELDS
######################################################
seed_examples:
  - question: What is the name of the tallest mountain in the world?
    answer: Mount Everest
  - question: What are the names of the five oceans of the world?
    answer: Atlantic, Pacific, Indian, Arctic, and the Antarctic
  - question: What is the largest desert in the world?
    answer: The Antarctic Desert
  - question: What is the longest river in Africa?
    answer: The Nile River
  - question: What is the smallest country in the world by land area?
    answer: Vatican City
  - question: What is the capital of Australia?
    answer: Canberra
  - question: What is the longest mountain range in South America?
    answer: The Andes mountain range
  - question: What are well known dense forests around the world?
    answer: Amazon Rainforest, Congo Basin, La Mosquitia jungle are few examples of dense rainforests with thick, nearly impenetrable vegetation.
  - question: Which country has the largest population in the world?
    answer: China
  - question: What American city is the Golden Gate Bridge located in?
    answer: San Francisco
  - question: What is the capital of Mexico?
    answer: Mexico City
  - question: What is the name of the largest ocean in the world?
    answer: The Pacific Ocean
  - question: What country has the most natural lakes?
    answer: Canada
  - question: What continent is Britain part of?
    answer: Europe
  - question: Which European country is closest to Africa?
    answer: Spain
  - question: In what country is the Taj Mahal located?
    answer: India
  - question: What do you call a chain of mountains?
    answer: A range
  - question: How many time zones does Russia have?
    answer: 11
  - question: What is the name of the only tropical rainforest in the United States?
    answer: Puerto Rico’s El Yunque National Forest
  - question: What country formerly ruled Iceland?
    answer: Denmark
```

## Running your SDG Code

With `fms_dgt/public/databuilders/test/geography_qa` and `tasks/public/test/geography_qa/task.yaml` defined, this can now be run by executing (from the base of the repo)

`python -m fms_dgt.public --task-path ./tasks/public/test/geography_qa/task.yaml`

Once this completes, you should be able to find the output of your system at

`output/public/test/geography_qa/final_data.jsonl`
