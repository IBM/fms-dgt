# Adding New Blocks

In this section, we'll go through the process of adding a new block to DGT. This will build off the data builder introduced in the [Starting From Scratch](./from_scratch.md) section.

## Prerequisites

To successfully run this, you will need to have completed the following:

1. Successfully completed the [Starting From Scratch](./from_scratch.md) tutorial
2. Read through the [Blocks](../key_concepts/blocks.md) section

## Recap of Blocks

Blocks are how one can contribute specialized algorithms or tools into DGT for other teams to use. Each block accepts as input a list of dict-like objects (e.g., [a pandas table, a list of dictionaries, etc.](https://github.ibm.com/DGT/fms-dgt/blob/5d6226fa2fa19aeb6dedffa9c58a6b17a53c9699/fms_dgt/constants.py#L9)). In addition, you can also pass as arguments [`input_map` / `output_map`](https://github.ibm.com/DGT/fms-dgt/blob/main/fms_dgt/base/block.py#L285) (or you can set these [in the init of the block](https://github.ibm.com/DGT/fms-dgt/blob/5d6226fa2fa19aeb6dedffa9c58a6b17a53c9699/fms_dgt/base/block.py#L40)).

Internally, a block is expected to iterate over each element of its input and extract instances of its associated [`DATA_TYPE`](https://github.ibm.com/DGT/fms-dgt/blob/5d6226fa2fa19aeb6dedffa9c58a6b17a53c9699/fms_dgt/base/block.py#L179). The result of the block is written onto the input elements (usually the input dictionaries) as specified by `output_map`.

## Defining a New Block

In this example, we'll define a [Validator Block](https://github.ibm.com/DGT/fms-dgt/blob/main/fms_dgt/core/blocks/validators/__init__.py). Validators are used to validate that the input element (most often being a newly generated data point from SDG) is valid and should be returned to the user.

Recall, in the [Starting From Scratch](./from_scratch.md) tutorial the objective was to define a geography question-answering SDG pipeline. We will keep that as the goal, however, with a modified objective being that we want to restrict the types of questions to be only factoid questions. This will be achieved by putting a length restriction on the answers our system generates (with answers that are too long being flagged as invalid). Create a `fms_dgt/research/blocks/validators/length_constraint.py` file with the following code

```python
# Standard
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.core.blocks.validators import BaseValidatorBlock, BaseValidatorBlockData


@dataclass(kw_only=True)
class LengthValidatorData(BaseValidatorBlockData):
    input: str


@register_block("length_constraint")
class LengthValidator(BaseValidatorBlock):
    """Class for length-constraint validator"""

    # NOTE: we must associate LengthValidatorData as this class's DATA_TYPE for the input dictionaries to be mapped to instances of LengthValidatorData
    DATA_TYPE = LengthValidatorData

    def __init__(
        self,
        max_len: int = 5,
        # NOTE: `filter` is a kwarg to BaseValidatorBlock, so it could be excluded here
        filter: bool = False,
        # NOTE: `input_map` / `output_map` are kwargs to BaseBlock, so they could be excluded here
        input_map: Optional[Dict] = None,
        output_map: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            filter=filter,
            input_map=input_map,
            output_map=output_map,
            **kwargs,
        )

        if max_len is None or max_len < 0:
            raise ValueError(
                f"Expected 'max_len' parameter to be a non-negative number"
            )

        self._max_len = max_len

    def execute(self, inputs: Iterable[LengthValidatorData]):
        """Checks whether an input has more than _max_len number of words for the field specified by `arg_fields`"""

        outputs = []
        for inp in inputs:
            # is_valid is inherited from BaseValidatorBlockData
            inp.is_valid = self._validate(inp)
            # NOTE: if self._filter_invalids is True, we do NOT return the input instance back to the user
            if inp.is_valid or not self._filter_invalids:
                outputs.append(inp)

        return outputs

    def _validate(self, inp: LengthValidatorData) -> bool:
        return len(inp.input.split()) <= self._max_len

```

Next, we must update our data builder and data builder config to actually make use of this new block. Update your `fms_dgt/research/databuilders/geography_qa/generate.py` file with the following code

```python
# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.blocks.llm import LMProvider
from fms_dgt.utils import sdg_logger
from fms_dgt.research.blocks.validators.length_constraint import LengthValidator
from fms_dgt.databuilders.geography_qa.task import (
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

    # NOTE: val1 is the validator we defined in our `blocks` directory
    val1: LengthValidator

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
            inp = {"input": prompt, "data": gd, "gen_kwargs": {"stop": ["Q:"]}}
            llm_inputs.append(inp)

        # NOTE: what blocks do (e.g., llm1) is iterate over each row of the input list and extract *args and **kwargs
        llm_outputs = self.llm1.generate(llm_inputs)

        outputs = []
        for llm_output in llm_outputs:
            # NOTE: the output of data generation will be found in "result" field and the original data will be found in the "data" field
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

        # NOTE: arguments that are not in our block's DATA_TYPE class are ignored, so we wrap out GeographySdgData objects in a dictionary with a 'data' key
        val_inputs = [{"to_check": out.answer, "data": out} for out in outputs]
        val_outputs = self.val1.generate(
            val_inputs, input_map={"to_check": "input"}
        )
        final_outputs = [out["data"] for out in val_outputs]

        # NOTE: useful to have logging information like this to sanity-check that filtering is actually occurring
        sdg_logger.info(
            "Discarded %s outputs due to violated length constraints, %s outputs returned",
            len(val_inputs) - len(final_outputs),
            len(final_outputs),
        )

        return final_outputs
```

Your code now makes use of your new validator block, however, you must also make it visible in the data builder config. Open up `fms_dgt/research/databuilders/geography_qa/geo_qa.yaml` and update the config to be the following

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
    model_id_or_path: mistralai/mixtral-8x7b-instruct-v01
  - name: val1
    type: length_constraint
    max_len: 5
    filter: true
metadata:
  version: 1.0
```

## Running your SDG Code

With those three files written / updated, you can run your code just as before. We will restart executing and increase the number of outputs generated, from the base of the repo execute the following command

`python -m fms_dgt.research --task-path ./tasks/research/geography_qa/qna.yaml --restart --num-outputs 100`

Once this completes, you should be able to find the output of your system at

`output/geography_qa/data.jsonl`

