# Standard
from pathlib import Path
from typing import Any
import os

# Local
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.base.task import GenerationTask
from fms_dgt.public.databuilders.instructlab.skills.data_objects import SkillsData
from fms_dgt.utils import dgt_logger


class SkillsTask(GenerationTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = SkillsData
    OUTPUT_DATA_TYPE = SkillsData

    def __init__(
        self,
        *args: Any,
        num_icl_examples_per_prompt: int = 3,
        num_questions_to_generate_per_prompt: int = 5,
        prompt_templates_dir: str | None = None,
        **kwargs: Any,
    ):
        # Initialize parent
        super().__init__(*args, **kwargs)

        # Save necessary variables
        self.num_icl_examples_per_prompt = num_icl_examples_per_prompt
        self.num_questions_to_generate_per_prompt = num_questions_to_generate_per_prompt

        # Load prompts
        if prompt_templates_dir:
            prompt_templates_dir = os.path.expandvars(prompt_templates_dir)

            if not os.path.exists(prompt_templates_dir):
                dgt_logger.warning(
                    'Failed to locate prompt templates directory at "%s". Loading default templates from "%s"',
                    prompt_templates_dir,
                    Path(Path(__file__).parent, "prompt_templates"),
                )
                prompt_templates_dir = Path(Path(__file__).parent, "prompt_templates")
        else:
            prompt_templates_dir = Path(Path(__file__).parent, "prompt_templates")

        self.prompt_templates = {}
        for template_path in Path(prompt_templates_dir).glob("*.txt"):
            self.prompt_templates[template_path.name[:-4]] = JinjaPromptTemplate(
                template_path=template_path, stop=["</s>", "[/EXAMPLE]"]
            )

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            task_description=self.task_description,
            instruction=kwargs.get("question", kwargs.get("instruction")),
            response=kwargs.get("answer", kwargs.get("response")),
            context=kwargs.get("context", None),
        )

    def instantiate_output_example(self, **kwargs):
        if not kwargs["context"]:
            kwargs.pop("context", None)
        return self.OUTPUT_DATA_TYPE(**kwargs)
