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
from fms_dgt.public.databuilders.examples.qa.data_objects import GeographyQAData
from fms_dgt.public.databuilders.examples.qa.task import GeographyQATask


# NOTE: we register the data builder with the below decorator so that we can reference it in an input data file later on
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

        # There are multiple ways you can define one or more prompts used during generation
        # 1. Via variable[s] (as shown here / a `prompts.py` where multiple prompts are specified)
        self._prompt_template = (
            "You are a geography question-answering data generator."
            " Your task is to come up with geography-related question-answer pairs that can be used to train a question-answering system."
            "\n\nHere are few examples:\n\n"
        )

        # 2. Via text file[s]
        self._prompts = {}
        for template_path in Path(Path(__file__).parent, "prompt_templates").glob("*.txt"):
            self._prompts[template_path.name[:-4]] = JinjaPromptTemplate(
                template_path=template_path
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

            encoded_icl_examples = "\n\n".join(
                [
                    f"Question: {icl_example.question}\nAnswer: {icl_example.answer}"
                    for icl_example in icl_examples
                ]
            )

            # Build prompt
            # Option A (chat_completion): pass as a messages list.  The model returns a
            # dict {"role": "assistant", "content": "..."} in the "result" field.
            prompt = self._prompts["prompt"].encode(render_dict={"examples": encoded_icl_examples})

            # Option B (completion): pass as a plain string with a trailing primer so the
            # model continues the text directly.  Useful for base / instruction models
            # that give more control via raw text completion (e.g. Mixtral, Qwen base).
            # To use this instead, change the input below to `prompt_completion` and
            # pass method="completion" to self.generator(...) below.
            #
            # prompt_completion = (
            #     f"{self._prompt_template}{encoded_icl_examples}"
            #     "\n\nNow generate a different question-answer pair in the similar format."
            #     "\n\nQuestion: "
            # )

            # Build generator inputs
            # input (str | List[Dict[str, Any]]): (Reserved field) messages for
            #   /chat_completion, or a plain string for /completion.
            # gen_kwargs (Optional[Dict[str, Any]]): (Reserved field) per-request
            #   overrides for generation params (e.g. {"temperature": 0.9, "max_tokens": 256}).
            #   Not set here because the defaults from qa.yaml are sufficient.
            # reference (Optional[Any]): We recommend passing data used to build prompt
            #   for future use. DiGiT returns all non-reserved fields in block output.
            # task_name (str): Not used by the block. Included for telemetry — allows
            #   DiGiT to attribute token usage to this task in traces.
            generator_inputs.append(
                {
                    "input": [{"role": "user", "content": prompt}],
                    "reference": icl_examples,
                    "task_name": icl_examples[0].task_name,
                }
            )

        # Execute block
        # LMProvider runs requests asynchronously for throughput.
        # Switch method="completion" and pass a plain string input (Option B above)
        # to use the text-completion endpoint instead.
        generator_outputs = self.generator(generator_inputs, method="chat_completion")

        # Process outputs from block
        outputs = []
        for generator_output in generator_outputs:
            # Extract icl examples passed to LMProvider block
            icl_examples = generator_output["reference"]

            # For chat_completion, "result" is a dict {"role": "assistant", "content": "..."}.
            # Extract the text content before parsing.
            result = generator_output["result"]
            if isinstance(result, dict):
                result = result.get("content") or ""

            question_answer_pair = result.split("Answer:")

            # Minimal check to guarantee well formed response
            if len(question_answer_pair) == 2:
                # For well-formed response, build "GeographyQAData" objects
                # As you can observed, having "reference" (icl examples) is handy to able to set correct "task_name"
                outputs.append(
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

        # Return generated synthetic data points
        return outputs
