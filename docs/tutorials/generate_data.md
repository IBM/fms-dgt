# Building a Generation Databuilder

This tutorial walks through building a generation databuilder from scratch. You will define the data schema, implement the generation logic, configure the builder, and run it. By the end you will have a working databuilder that generates new examples from a small set of seeds using an LLM.

The domain used here is common misconceptions: given a few seed examples of a widely-held false belief paired with a correction, the builder generates more examples in the same format.

## What you will build

A databuilder registered as `public/examples/misconceptions` that:

- Accepts seed examples with a `misconception` field and a `correction` field
- Prompts a language model with a few in-context learning examples to generate new pairs
- Deduplicates outputs using Rouge-L scoring

All files go under `fms_dgt/public/databuilders/examples/misconceptions/` and `tasks/public/examples/misconceptions/`.

## Step 1: Create the directory

```bash
mkdir -p fms_dgt/public/databuilders/examples/misconceptions/prompt_templates
mkdir -p tasks/public/examples/misconceptions
```

## Step 2: Define the data schema

Create `fms_dgt/public/databuilders/examples/misconceptions/data_objects.py`:

```{.python title="fms_dgt/public/databuilders/examples/misconceptions/data_objects.py"}
# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass

# Local
from fms_dgt.base.data_objects import DataPoint


@dataclass(kw_only=True)
class MisconceptionData(DataPoint):
    """Holds one misconception-correction pair, either from seed data or generated."""

    misconception: str
    correction: str
```

`DataPoint` is DiGiT's base class for all data records. It provides the `task_name` and `is_seed` fields automatically. You only need to declare the fields specific to your domain.

## Step 3: Define the task

Create `fms_dgt/public/databuilders/examples/misconceptions/task.py`:

```{.python title="fms_dgt/public/databuilders/examples/misconceptions/task.py"}
# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# Local
from fms_dgt.base.task import GenerationTask
from fms_dgt.public.databuilders.examples.misconceptions.data_objects import MisconceptionData


class MisconceptionTask(GenerationTask):
    """
    Generation task for misconception-correction pairs.

    GenerationTask loops over a mixture of seed and synthetic data until
    the requested number of outputs is reached.
    """

    INPUT_DATA_TYPE = MisconceptionData
    OUTPUT_DATA_TYPE = MisconceptionData

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def instantiate_input_example(self, **kwargs: Any) -> MisconceptionData:
        return MisconceptionData(
            task_name=self.name,
            is_seed=True,
            misconception=kwargs.get("misconception"),
            correction=kwargs.get("correction"),
        )
```

`INPUT_DATA_TYPE` and `OUTPUT_DATA_TYPE` are both `MisconceptionData` because the generator produces examples in the same format as the seeds. DiGiT uses these type annotations to validate data flowing through the pipeline.

## Step 4: Write the prompt template

Create `fms_dgt/public/databuilders/examples/misconceptions/prompt_templates/prompt.txt`:

```{.text title="fms_dgt/public/databuilders/examples/misconceptions/prompt_templates/prompt.txt"}
You are a misconception correction data generator. Your task is to come up with
commonly-held false beliefs paired with accurate corrections, for use in training
a language model to identify and correct misinformation.

Please follow these guiding principles:
 * State the misconception as a simple declarative sentence.
 * State the correction clearly and concisely without being condescending.
 * Do not use list markers, bullets, or numbering in either field.

Here are a few examples:

{{ examples }}

Now generate a single different misconception-correction pair in the same format.

```

The `{{ examples }}` placeholder is filled at runtime by the databuilder with formatted in-context learning examples.

## Step 5: Implement the generation logic

Create `fms_dgt/public/databuilders/examples/misconceptions/generate.py`:

```{.python title="fms_dgt/public/databuilders/examples/misconceptions/generate.py"}
# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Any, Dict, List
import random

# Local
from fms_dgt.base.databuilder import GenerationDataBuilder
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.public.databuilders.examples.misconceptions.data_objects import MisconceptionData
from fms_dgt.public.databuilders.examples.misconceptions.task import MisconceptionTask


@register_data_builder("public/examples/misconceptions")
class MisconceptionDataBuilder(GenerationDataBuilder):
    """Generates misconception-correction pairs via in-context learning."""

    TASK_TYPE: MisconceptionTask = MisconceptionTask

    # DiGiT injects the LMProvider block declared in the YAML config.
    generator: LMProvider

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Load the prompt template from the prompt_templates directory.
        template_path = Path(Path(__file__).parent, "prompt_templates", "prompt.txt")
        self._prompt = JinjaPromptTemplate(template_path=template_path)

    def __call__(
        self,
        request_idx: int,
        seed_data: List[MisconceptionData],
    ) -> List[MisconceptionData]:

        # Build one LM request per seed example, each with 3 random ICL examples.
        generator_inputs: List[Dict] = []
        for _ in range(len(seed_data)):
            icl_examples = random.choices(seed_data, k=3)

            encoded_examples = "\n\n".join(
                [
                    f"Misconception: {ex.misconception}\nCorrection: {ex.correction}"
                    for ex in icl_examples
                ]
            )

            # input: messages list for chat_completion.
            # reference: passed through unchanged so we can recover task_name below.
            generator_inputs.append(
                {
                    "input": [
                        {
                            "role": "user",
                            "content": self._prompt.encode(
                                render_dict={"examples": encoded_examples}
                            ),
                        }
                    ],
                    "reference": icl_examples,
                }
            )

        # LMProvider runs all requests asynchronously for throughput.
        generator_outputs = self.generator(generator_inputs, method="chat_completion")

        outputs = []
        for output in generator_outputs:
            icl_examples = output["reference"]

            result = output["result"]
            if isinstance(result, dict):
                result = result.get("content") or ""

            # Parse the "Misconception: ... \n Correction: ..." format.
            parts = result.split("Correction:")
            if len(parts) == 2:
                misconception = parts[0].replace("Misconception:", "").strip().rstrip("\n")
                correction = parts[1].strip().rstrip("\n")
                outputs.append(
                    MisconceptionData(
                        task_name=icl_examples[0].task_name,
                        is_seed=False,
                        misconception=misconception,
                        correction=correction,
                    )
                )

        return outputs
```

A few things worth noting:

- `@register_data_builder("public/examples/misconceptions")` registers this class under a name that must match the `name` field in the YAML config and the `data_builder` field in the task YAML.
- `generator: LMProvider` is a class-level annotation. DiGiT reads the YAML config and injects the configured LM block automatically.
- The `reference` field passes ICL examples through the LM call so we can recover `task_name` when building output objects.

## Step 6: Write the builder config

Create `fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml`:

```{.yaml title="fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml"}
######################################################
#                   MANDATORY FIELDS
######################################################
name: public/examples/misconceptions

######################################################
#                   RESERVED FIELDS
######################################################
blocks:
  - name: generator
    type: ollama
    model_id_or_path: granite4:3b
    temperature: 0.7
    max_tokens: 128
    num_ctx: 4096
  - name: dedup
    type: rouge_scorer
    filter: true
    threshold: 1.0
    input_map:
      misconception: input
postprocessors:
  - name: dedup
metadata:
  version: 1.0
```

The `dedup` block filters out generated misconceptions that are too similar to existing ones, using Rouge-L overlap. The `input_map` tells the block which field to compare (`misconception` maps to the block's `input` field).

## Step 7: Create the task file

Create `tasks/public/examples/misconceptions/task.yaml`:

```{.yaml title="tasks/public/examples/misconceptions/task.yaml"}
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: public/examples/misconceptions
task_description: Generate misconception-correction pairs for training a model to identify and correct misinformation.
created_by: IBM

data_builder: public/examples/misconceptions

######################################################
#                   RESERVED FIELDS
######################################################
seed_examples:
  - misconception: Lightning never strikes the same place twice.
    correction: Lightning frequently strikes the same place multiple times. Tall structures like the Empire State Building are struck dozens of times per year.
  - misconception: Humans only use 10 percent of their brains.
    correction: Brain imaging studies show that virtually all regions of the brain are active at some point, and most are active almost all the time.
  - misconception: Swallowed chewing gum stays in your stomach for seven years.
    correction: While gum base is not digestible, it passes through the digestive system and is excreted within a few days, just like other indigestible matter.
  - misconception: Goldfish have a memory span of only three seconds.
    correction: Research has shown that goldfish can remember things for months and can be trained to navigate mazes and recognize their owners.
  - misconception: The Great Wall of China is visible from space with the naked eye.
    correction: The Great Wall is too narrow to be seen from low Earth orbit without aid. Astronauts have confirmed this repeatedly.
```

## Step 8: Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/misconceptions/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

Generated data is written to `output/public/examples/misconceptions/final_data.jsonl`. Each record looks like:

```json
{
  "task_name": "public/examples/misconceptions",
  "is_seed": false,
  "misconception": "Carrots improve your night vision.",
  "correction": "Carrots contain vitamin A, which is necessary for normal vision, but consuming extra carrots does not enhance night vision beyond normal levels. The association originated from World War II British propaganda."
}
```

## Next steps

- To switch to a cloud provider, see [Changing the LM Engine](../tutorials/changing_lm_engine.md).
- To add a validator that filters low-quality outputs, see [Creating a Validator](../tutorials/creating_validator.md).
- To build a transformation databuilder, see [Building a Transformation Databuilder](transform_data.md).
