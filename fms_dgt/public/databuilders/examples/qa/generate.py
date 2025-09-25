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

            # Build prompt
            # 1. From variable
            prompt = f'{self._prompt_template}{"\n\n".join([f"Question: {icl_example.question}\nAnswer: {icl_example.answer}" for icl_example in icl_examples])}\n\nNow generate a different question-answer pair in the similar format.\n\nQuestion: '

            # OR
            # 2. Using PromptTemplate class
            prompt = self._prompts["prompt"].encode(
                render_dict={
                    "examples": "\n\n".join(
                        [
                            f"Question: {icl_example.question}\nAnswer: {icl_example.answer}"
                            for icl_example in icl_examples
                        ]
                    )
                }
            )

            # Build generator inputs
            # input (str | List[Dict[str, Any]]): (Reserved field) prompt to be passed to `/completion` endpoint or messages to be passed to `/chat/completion` endpoint
            # gen_kwargs (Optional[Dict[str, Any]]): (Reserved field) Additional generation specific parameters to be passed to `/completion` or `/chat/completion` endpoint
            # reference (Optional[Any]): We recommend passing data used to build prompt for future use. DiGiT returns all non-reserved field in output from a block.
            generator_inputs.append(
                {
                    "input": prompt,
                    "gen_kwargs": {"stop": ["Question:"]},
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
