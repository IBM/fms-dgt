# Creating a Validator

### Recap of Blocks

Blocks are one of the way to contribute specialized algorithms or tools to DGT, making them accessible for other individuals to use. Each block takes as input a list of dictionary-like objects (e.g., [a pandas table, a list of dictionaries, etc.](https://github.com/IBM/fms-dgt/blob/ec3ce21f341bf9938426a082b1e63431da031e03/fms_dgt/constants.py#L9)).

Additionally, blocks can accept input_map and output_map as arguments ([see here](https://github.com/IBM/fms-dgt/blob/ec3ce21f341bf9938426a082b1e63431da031e03/fms_dgt/base/block.py#L338)), or these can be set during block initialization ([see here](https://github.com/IBM/fms-dgt/blob/ec3ce21f341bf9938426a082b1e63431da031e03/fms_dgt/base/block.py#L65)).

Internally, a block is expected to iterate over each input element and extract instances of its associated [`DATA_TYPE`](https://github.com/IBM/fms-dgt/blob/ec3ce21f341bf9938426a082b1e63431da031e03/fms_dgt/base/block.py#L235). The block’s output is then written back to the input elements (typically dictionaries), as specified by the output_map.

### Creating a New Block

In this example, we’ll define a [Validator Block](https://github.com/IBM/fms-dgt/blob/ec3ce21f341bf9938426a082b1e63431da031e03/fms_dgt/base/block.py#L421). Validator blocks are used to verify whether an input element, often a newly generated data point from SDG, is valid and should be returned to the user.

Let’s revisit the [Data Generation example](../examples/generate_data.md), where the goal was to build a geography question-answering pipeline. We’ll continue with that context, but modify the objective: we now want to restrict the generated questions to factoid-type questions only.

To enforce this, we’ll apply a length constraint on the answers. If an answer exceeds a certain length, it will be flagged as invalid.

To implement this, create a file at:

`fms_dgt/public/databuilders/test/geography_qa/blocks/length_constraint/block.py`

and add the following code:

```{.python title="fms_dgt/public/databuilders/test/geography_qa/blocks/length_constraint/block.py"}
# Standard
from dataclasses import dataclass
from typing import Tuple, Dict

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.base.block import ValidatorBlock
from fms_dgt.base.data_objects import ValidatorBlockData


@dataclass(kw_only=True)
class LengthValidatorData(ValidatorBlockData):
    input: str


@register_block("public/test/geography_qa/length_constraint")
class LengthValidator(ValidatorBlock):
    """Class for length-constraint validator"""

    # We must associate LengthValidatorData as this class's DATA_TYPE for the input dictionaries to be mapped to instances of LengthValidatorData
    DATA_TYPE = LengthValidatorData

    def __init__(
        self,
        *args,
        max_num_words: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        if max_num_words is None or max_num_words < 0:
            raise ValueError("Expected 'max_num_words' parameter to be a non-negative number")

        self._max_num_words = max_num_words

    def _validate(self, instance: LengthValidatorData) -> Tuple[bool, Dict | None]:
        # Calculate number of words in the input
        num_words = len(instance.input.split())

        # Perform validity check
        is_valid = num_words <= self._max_num_words

        # Return
        return is_valid, (
            {
                "reason": f"Number of words in input ({num_words}) exceeds limit ({self._max_num_words})."
            }
            if not is_valid
            else None
        )

```

### Integrating the New Block into the Data Builder

Next, we need to update the data builder and its configuration to incorporate the newly created block.

Open the file: `fms_dgt/public/databuilders/test/geography_qa/generate.py`

and update it with the following code:

```{.python title="fms_dgt/public/databuilders/test/geography_qa/generate.py" hl_lines="28-29 104-120"}
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
from fms_dgt.public.databuilders.test.geography_qa.blocks.length_constraint.block import (
    LengthValidator,
)


# NOTE: we register the data builder with the below decorator so that we can reference it in an input data file later on
@register_data_builder("public/test/geography_qa")
class GeographyQADataBuilder(GenerationDataBuilder):
    """Geography QA data builder"""

    TASK_TYPE: GeographyQATask = GeographyQATask

    # Generator is the language model that we will use to produce the synthetic examples
    generator: LMProvider

    # Validator is the validator we defined in our `blocks` directory
    validator: LengthValidator

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
            encoded_icl_examples = "\n\n".join(
                [
                    f"Question: {icl_example.question}\nAnswer: {icl_example.answer}"
                    for icl_example in icl_examples
                ]
            )
            prompt = f"{self._prompt_template}{encoded_icl_examples}\n\nNow generate a single different question-answer pair in the similar format.\n\n"

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
        generated_outputs = []
        for generator_output in generator_outputs:
            # Extract icl examples passed to LMProvider block
            icl_examples = generator_output["reference"]

            # LMProvider block return output from `/completion` or `/chat/completion` endpoint in "result" field.
            question_answer_pair = generator_output["result"].split("Answer:")

            # Minimal check to guarantee well formed response
            if len(question_answer_pair) == 2:
                # For well-formed response, build "GeographyQAData" objects
                # As you can observed, having "reference" (icl examples) is handy to able to set correct "task_name"
                generated_outputs.append(
                    GeographyQAData(
                        task_name=icl_examples[0].task_name,
                        is_seed=False,
                        question=question_answer_pair[0]
                        .split("Question:")[-1]
                        .strip()
                        .rstrip("\n"),
                        answer=question_answer_pair[1].strip().rstrip("\n"),
                    )
                )

        # Arguments that are not in our block's DATA_TYPE class are ignored, so we wrap out GeographySdgData objects in a dictionary with a 'reference' key
        # "store_names" is an optional parameter designed to store the results of filtered data.
        validated_outputs = self.validator(
            [
                {
                    "input": generated_output.answer,
                    "reference": generated_output,
                    "store_names": self.get_block_store_names(
                        block_name=self.validator.name, task_name=generated_output.task_name
                    ),
                }
                for generated_output in generated_outputs
            ]
        )

        # Return validated synthetic data points
        return [validated_output["reference"] for validated_output in validated_outputs]
```

This update ensures that the length_constraint block is executed as part of the data generation pipeline. Specifically, it will validate each generated answer and filter out those that exceed the allowed length, helping enforce the factoid-only constraint.

Your code now makes use of your new validator block, however, you must also make it visible in the data builder config. Open up `fms_dgt/research/databuilders/geography_qa/geography_qa.yaml` and update the config to be the following

```{.yaml title="fms_dgt/research/databuilders/geography_qa/geography_qa.yaml" hl_lines="16-20"}
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
  # Factoid answer validator (using length as proxy)
  - name: validator
    type: public/test/geography_qa/length_constraint
    max_num_words: 3
    filter: true
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

### Running your Code

From the root of your repository, execute the following command:

```bash
python -m fms_dgt.public --task-path ./tasks/public/test/geography_qa/task.yaml --restart-generation
```

Once the process completes, the generated data will be available at:

`output/public/test/geography_qa/final_data.jsonl`
