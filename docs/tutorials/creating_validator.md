# Creating a Validator

This tutorial extends the misconceptions databuilder built in [Building a Generation Databuilder](../tutorials/generate_data.md) with a custom validator block. The validator filters out generated outputs that are malformed: questions instead of statements, empty strings, or outputs that are implausibly short to be a real misconception.

## Recap: Blocks

Blocks are single-operation components that can be reused across databuilders. A `ValidatorBlock` takes a list of items, evaluates each one, and returns only the items that pass. Rejected items are optionally written to a separate store for inspection.

Each block has a [`DATA_TYPE`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/base/block.py#L235) dataclass that maps incoming dictionary fields to typed attributes. The block's `_validate` method receives one instance and returns `(is_valid, reason_dict)`. For validators that call an LM or external API, override `_validate_batch` instead to process the full input list in one call.

Blocks also accept [`input_map` and `output_map`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/base/block.py#L65) arguments that rename fields at the boundary between the caller and the block, and a [`ValidatorBlock`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/base/block.py#L421) base class handles the filtering loop so you only need to implement `_validate` or `_validate_batch`.

For full details, see [Blocks](../concepts/blocks.md).

## Step 1: Create the block file

Create `fms_dgt/public/databuilders/examples/misconceptions/blocks/statement_check/block.py`:

```{.python title="fms_dgt/public/databuilders/examples/misconceptions/blocks/statement_check/block.py"}
# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Local
from fms_dgt.base.block import ValidatorBlock
from fms_dgt.base.data_objects import ValidatorBlockData
from fms_dgt.base.registry import register_block


@dataclass(kw_only=True)
class StatementCheckData(ValidatorBlockData):
    """Input data type for the statement check validator."""

    input: str


@register_block("public/examples/misconceptions/statement_check")
class StatementCheckValidator(ValidatorBlock):
    """Rejects misconception strings that are questions, empty, or too short."""

    DATA_TYPE = StatementCheckData

    def __init__(
        self,
        *args,
        min_words: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._min_words = min_words

    def _validate(self, instance: StatementCheckData) -> Tuple[bool, Optional[Dict]]:
        text = instance.input.strip()

        if not text:
            return False, {"reason": "Empty string."}

        if text.endswith("?"):
            return False, {"reason": "Output is a question, not a statement."}

        word_count = len(text.split())
        if word_count < self._min_words:
            return False, {
                "reason": f"Too short: {word_count} words (minimum {self._min_words})."
            }

        return True, None
```

Three checks in order: empty string, question mark at the end (the model sometimes generates a question instead of a statement), and a minimum word count to catch truncated outputs.

## Step 2: Update `generate.py`

Open `fms_dgt/public/databuilders/examples/misconceptions/generate.py` and add the validator import and class-level annotation:

```{.python title="fms_dgt/public/databuilders/examples/misconceptions/generate.py" hl_lines="14 15 28 90-113"}
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
from fms_dgt.public.databuilders.examples.misconceptions.blocks.statement_check.block import (
    StatementCheckValidator,
)
from fms_dgt.public.databuilders.examples.misconceptions.data_objects import MisconceptionData
from fms_dgt.public.databuilders.examples.misconceptions.task import MisconceptionTask


@register_data_builder("public/examples/misconceptions")
class MisconceptionDataBuilder(GenerationDataBuilder):
    """Generates misconception-correction pairs via in-context learning."""

    TASK_TYPE: MisconceptionTask = MisconceptionTask

    generator: LMProvider
    validator: StatementCheckValidator

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        template_path = Path(Path(__file__).parent, "prompt_templates", "prompt.txt")
        self._prompt = JinjaPromptTemplate(template_path=template_path)

    def __call__(
        self,
        request_idx: int,
        seed_data: List[MisconceptionData],
    ) -> List[MisconceptionData]:

        generator_inputs: List[Dict] = []
        for _ in range(len(seed_data)):
            icl_examples = random.choices(seed_data, k=3)

            encoded_examples = "\n\n".join(
                [
                    f"Misconception: {ex.misconception}\nCorrection: {ex.correction}"
                    for ex in icl_examples
                ]
            )

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

        generator_outputs = self.generator(generator_inputs, method="chat_completion")

        outputs = []
        for output in generator_outputs:
            icl_examples = output["reference"]

            result = output["result"]
            if isinstance(result, dict):
                result = result.get("content") or ""

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

        # Run the statement check validator on the misconception field.
        validated = self.validator(
            [
                {
                    "input": dp.misconception,
                    "reference": dp,
                    "store_names": self.get_block_store_names(
                        block_name=self.validator.name,
                        task_name=dp.task_name,
                    ),
                }
                for dp in outputs
            ]
        )

        return [item["reference"] for item in validated]
```

The validator call passes the `misconception` field as `input` (matching `StatementCheckData.input`) and returns only the items that pass. The `reference` field carries the full `MisconceptionData` object through unchanged so it can be recovered in the return value.

## Step 3: Update the builder YAML

Add the `validator` block to `fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml`:

```{.yaml title="fms_dgt/public/databuilders/examples/misconceptions/misconceptions.yaml" hl_lines="16-19"}
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
  - name: validator
    type: public/examples/misconceptions/statement_check
    min_words: 5
    filter: true
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

`filter: true` tells the block to drop rejected items rather than returning them. Rejected items are written to a separate store under `output/public/examples/misconceptions/` for inspection.

## Step 4: Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/misconceptions/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

Only outputs that pass the statement check proceed to the Rouge-L deduplication step. Rejected items are saved to `output/public/examples/misconceptions/block_store_validator_*.jsonl` alongside the reason for rejection.

## Next steps

- To switch to a different LM engine, see [Changing the Language Model Engine](changing_lm_engine.md).
- To load seed examples from an external file, see [Loading Seed Examples from a File](loading_seed_examples_from_file.md).
- To build a transformation databuilder, see [Building a Transformation Databuilder](../tutorials/transform_data.md).
