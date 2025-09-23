# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import ast
import glob
import json
import os
import re

# Local
from fms_dgt.base.block import Block, BlockData
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.constants import TYPE_KEY
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.public.blocks.magpie.tag.prompts import MagpieTransformPrompt
from fms_dgt.utils import dgt_logger

# ===========================================================================
#                       CONSTANTS
# ===========================================================================
DEFAULT_TASKS = [
    "quality",
    "sample_quality",
    "difficulty",
]
TAG_DICT = {
    "quality": "input_quality",
    "sample_quality": "score",
    "difficulty": "difficulty",
    "classification": "primary_tag",
}


# ===========================================================================
#                       UTILITY FUNCTIONS
# ===========================================================================
def extract_first_enclosed_braces(string: str):
    start_idx, end_idx = None, None
    left_braces = None
    for char_idx, char in enumerate(string):
        if char == "{":
            if left_braces is None:
                left_braces = 0
                start_idx = char_idx
            left_braces += 1
        elif char == "}":
            if left_braces:
                left_braces -= 1
        if left_braces == 0:
            end_idx = char_idx
            break
    return start_idx, end_idx


# ===========================================================================
#                       BLOCK
# ===========================================================================
@dataclass(kw_only=True)
class MagpieTaggerBlockData(BlockData):
    """
    Data type for Magpie tagging block
    """

    # ===========================================================================
    #                       INPUT FIELDS
    # ===========================================================================
    magpie_input: Optional[str] = None
    magpie_output: Optional[str] = None

    # For multi-turn setting, the input must be array of dictionaries
    # with each item in dictionary containing "text"/"content" field
    magpie_mt_input: Optional[List[Dict[str, Any]]] = None

    # ===========================================================================
    #                       OUTPUT FIELDS
    # ===========================================================================
    magpie_tags: Optional[Dict] = None


@register_block("magpie_tag")
class MagpieTagger(Block):
    r"""Class for Magpie Tagging

    Args:
        lm_config (dict): language model configuration
        tasks (List[str]): tagging tasks to perform. The available choices are "quality", "sample_quality", "difficulty", "classification", "conversation_quality".

    .. code-block:: python

        # Initialize tagger
        tagger = MagpieTagger(lm_config={"type": "genai", ...}, input_map={"question": "magpie_input", "answer": "magpie_output"})

        # Sample data
        data = [
                {
                    "question": "what is capital of the United States of America?",
                    "answer": "Washington D.C"
                },
                {
                    "question": "What is biggest star in our solar system?",
                    "answer": "Sun is the biggest star in our solar system."
                }
            ]

        # Invoke tagger
        tagger(data)


    """

    DATA_TYPE = MagpieTaggerBlockData

    def __init__(
        self,
        lm_config: Dict,
        tasks: Optional[List[str]] = None,
        input_map: Optional[Union[List, Dict]] = None,
        output_map: Optional[Union[List, Dict]] = None,
        **kwargs: Any,
    ) -> None:
        # Step 1: Set default values for "input_map" & "output_map", if necessary
        if input_map is None:
            input_map = {
                "question": "magpie_input",
                "answer": "magpie_output",
                "messages": "magpie_mt_input",
            }

        if output_map is None:
            output_map = {
                "magpie_tags": "magpie_tags",
            }

        # Step 2: Initialize parent
        super().__init__(
            input_map=input_map,
            output_map=output_map,
            **kwargs,
        )

        # Step 3: Assert all necessary information is available
        if TYPE_KEY not in lm_config:
            raise ValueError(
                f"Must specify {TYPE_KEY} in 'lm_config' field of {self.name} block"
            )

        # Step 4: Initialize langauge model
        self._llm_generator: LMProvider = get_block(
            lm_config.get(TYPE_KEY), **lm_config
        )
        self._blocks.append(self._llm_generator)

        # Step 5: Load prompts
        self._prompts = {}
        for template_path in glob.glob(
            f"{os.path.join(os.path.split(os.path.abspath(__file__))[0], 'prompt_templates')}/*.txt"
        ):
            prompt_name = template_path.split("/")[-1][:-4]
            if "_mt" in prompt_name:
                prompt_type = "multi_turn"
            else:
                prompt_type = "single_turn"
            self._prompts[prompt_name] = MagpieTransformPrompt(
                template_path=template_path, prompt_type=prompt_type
            )

        # Step 6: Save mission
        if tasks is None:
            tasks = DEFAULT_TASKS

        self.tasks = []
        for task in tasks:
            if task.strip() not in [
                "quality",
                "sample_quality",
                "difficulty",
                "classification",
                "conversation quality",
            ]:
                raise ValueError(
                    f"Invalid task type ({task.strip()}) is passed to MagpieTagger block."
                )
            else:
                self.tasks.append(task.strip())

    # ===========================================================================
    #                       PROMPT GENERATOR
    # ===========================================================================
    def template_generator_mt(self, conversation, task: str):
        if task == "quality":
            return self._prompts["input_quality_rating_mt"].encode(input=conversation)
        elif task == "sample_quality":
            return self._prompts["sample_quality_rating_mt"].encode(input=conversation)
        elif task == "conversation_quality":
            return self._prompts["conversations_quality_rating_mt"].encode(
                input=conversation
            )
        else:
            raise ValueError(
                "Invalid mission. Available missions: quality, sample_quality, conversation_quality"
            )

    def template_generator(self, input_text: str, task: str, output_text: str = ""):
        """
        Generate prompt template for specified tagging task

        Args:
            text (str): input text
            task (str): requested tagging task
            output (str, optional): output text. Defaults to "".

        Raises:
            ValueError: invalid task type

        Returns:
            str: generated prompt
        """
        if task == "difficulty":
            return self._prompts["input_difficulty_rating"].encode(input_text)
        elif task == "quality":
            return self._prompts["input_quality_rating"].encode(input_text)
        elif task == "classification":
            return self._prompts["input_classification"].encode(input_text)
        elif task == "sample_quality":
            return self._prompts["sample_quality_rating"].encode(
                input_text, output_text
            )
        else:
            raise ValueError(
                "Invalid tagging task. Available tagging tasks are 'difficulty', 'quality', 'classification' and 'sample_quality'"
            )

    # ===========================================================================
    #                       UTILITY FUNCTIONS
    # ===========================================================================
    def parse(self, response: str, instance: MagpieTaggerBlockData, tag_task: str):
        """
        Parse language model response to prepare final tagged outputs

        Args:
            response (str): generated response to parse
            instance (MagpieTaggerBlockData): instance being tagged
            tag_task (str): tagging task to perform. The available choices are "quality", "sample_quality", "difficulty", "classification", "conversation_quality"
        """

        def salvage_response(response, label):
            if label != "score":
                pattern = r'"{label}":\s*"([\w\s]*)"'
                match = re.search(pattern, response)
            else:
                pattern = r'"score":\s*"?\[?"?(\d+)'
                match = re.search(pattern, response)
                if not match:
                    pattern = r'"score":\s*"(.*)"'
                    match = re.search(pattern, response)

            if match:
                return match.group(1)
            else:
                return "None"

        def salvage_tag(rating):
            if isinstance(rating, str):
                try:
                    rating_eval = ast.literal_eval(rating)
                except Exception:
                    rating_eval = None
                if isinstance(rating_eval, List):
                    rating = str(rating_eval[0])
                    return str(rating)

                if "[" in rating and "]" in rating:

                    pattern = r"(\[[^\]]+\])"
                    match = re.search(pattern, rating)
                    if match:
                        new_rating = match.group(1)
                        try:

                            if isinstance(ast.literal_eval(new_rating), List):
                                rating = str(ast.literal_eval(rating)[0])
                                return str(rating)
                        except Exception:
                            return rating

                pattern = r".*(\d+)"
                match = re.search(pattern, rating)
                if not match:
                    pattern = r".*(very poor|poor|average|good|excellent|very easy|easy|medium|hard|very hard).*"
                    match = re.search(pattern, rating)
                if match:
                    return match.group(1)
                else:
                    return rating
            elif isinstance(rating, List):
                rating = str(rating[0])
                return rating
            elif isinstance(rating, float):
                return str(int(rating))
            elif isinstance(rating, int):
                return str(rating)
            else:
                return rating

        def clean(text: str):
            if text.startswith("```json") and text.endswith("```"):
                return text[7:-3].strip()
            if text.startswith("```") and text.endswith("```"):
                return text[3:-3].strip()
            return text

        def set_error_values(
            entry: Dict, tag_task: str, model_name: str, error: Any, response: Any
        ):
            if tag_task == "quality":
                entry["input_quality"] = None
                entry["input_quality_explanation"] = None
                entry["metadata"]["label_model"] = [model_name]
                entry["metadata"]["quality_err"] = error
                entry["metadata"]["quality_raw"] = response
            elif tag_task == "sample_quality":
                entry["judge_quality_score"] = None
                entry["judge_quality_explanation"] = None
                entry["metadata"]["label_model"] = [model_name]
                entry["metadata"]["sample_quality_err"] = error
                entry["metadata"]["sample_quality_raw"] = response
            elif tag_task == "difficulty":
                entry["intent"] = None
                entry["knowledge"] = None
                entry["difficulty"] = None
                entry["metadata"]["label_model"] = [model_name]
                entry["metadata"]["difficulty_err"] = error
                entry["metadata"]["difficulty_raw"] = response
            elif tag_task == "classification":
                entry["task_category"] = None
                entry["metadata"]["label_model"] = [model_name]
                entry["metadata"]["classification_err"] = error
                entry["metadata"]["classification_raw"] = response
            elif tag_task == "conversation_quality":
                entry["conversation_score"] = None
                entry["conversation_explanation"] = None
                entry["metadata"]["label_model"] = [model_name]
                entry["metadata"]["conv_quality_err"] = error
                entry["metadata"]["conv_quality_raw"] = response

        # Step 1: Extract model name used to generate
        model_name = self._llm_generator.model_id_or_path

        # Step 2: Initialize "metadata" for tag's field in the instance, if non existent
        if (
            not "metadata" in instance.magpie_tags
            or not instance.magpie_tags["metadata"]
        ):
            instance.magpie_tags["metadata"] = {}

        # Step 3: Process response
        # Step 3.a: Clean response
        response = clean(response)

        # Step 3.b: Try to convert response to dictionary, otherwise try to salvage from explanation string
        try:
            response_dict = json.loads(response)
            if isinstance(response_dict, list):
                response_dict = response_dict[0]

            # Step 3.b.i: Salvage response, if necessary
            if TAG_DICT[tag_task] not in response_dict:
                response_dict[TAG_DICT[tag_task]] = salvage_response(
                    response, TAG_DICT[tag_task]
                )

        except json.decoder.JSONDecodeError:
            response_dict = {}
            response_dict[TAG_DICT[tag_task]] = salvage_response(
                response, TAG_DICT[tag_task]
            )
            response_dict["explanation"] = response
            if tag_task == "difficulty":
                response_dict["intent"] = None
                response_dict["knowledge"] = None

        # Step 3.c: Process for "quality" tag task
        if tag_task == "quality":
            # Step 3.c.ii: Validate
            if "input_quality" not in response_dict:
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Failed to rate for {tag_task} task. Defaulting to None.",
                    response=response,
                )
                return instance

            # Step 3.c.ii: Salvage tag if it is not in valid list
            if response_dict["input_quality"] not in [
                "very poor",
                "poor",
                "average",
                "good",
                "excellent",
            ]:
                response_dict["input_quality"] = salvage_tag(
                    response_dict["input_quality"]
                )

            # Step 3.c.iii: Validate format
            if not isinstance(response_dict["input_quality"], str):
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Invalid Format type in response. type: {type(response_dict['input_quality'])} value: {response_dict['input_quality']}",
                    response=response,
                )
                return instance

            # Step 3.c.iv: Save
            instance.magpie_tags["input_quality"] = [
                {f"{model_name}": response_dict["input_quality"]}
            ]
            if "explanation" in response_dict:
                instance.magpie_tags["input_quality_explanation"] = [
                    {f"{model_name}": response_dict["explanation"]}
                ]
            instance.magpie_tags["metadata"]["label_model"] = [model_name]

        # Step 3.d: Process for "sample_quality" tag task
        elif tag_task == "sample_quality":
            # Step 3.d.i: Validate
            if "score" not in response_dict:
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Failed to rate for {tag_task} task. Defaulting to None.",
                    response=response,
                )
                return instance

            # Step 3.d.ii: Salvage tag if it is not in valid list
            if response_dict["score"] not in ["1", "2", "3", "4", "5"]:
                response_dict["score"] = salvage_tag(response_dict["score"])

            # Step 3.d.iii: Validate format
            if not (
                (
                    isinstance(response_dict["score"], str)
                    and response_dict["score"].isdigit()
                )
                or isinstance(response_dict["score"], (int, float))
            ):
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Invalid Format type in response. type: {type(response_dict['score'])} value: {response_dict['score']}",
                    response=response,
                )
                return instance

            # Step 3.d.iv: Save
            instance.magpie_tags["judge_quality_score"] = [
                {f"{model_name}": str(response_dict["score"])}
            ]

            if "explanation" in response_dict:
                instance.magpie_tags["judge_quality_explanation"] = [
                    {f"{model_name}": response_dict["explanation"]}
                ]
            instance.magpie_tags["metadata"]["label_model"] = [model_name]

        # Step 3.e: Process for "difficulty" tag task
        elif tag_task == "difficulty":
            # Step 3.e.i: Validate
            if "difficulty" not in response_dict:
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Failed to rate for {tag_task} task. Defaulting to None.",
                    response=response,
                )
                return instance

            # Step 3.e.ii: Salvage tag if it is not in valid list
            if response_dict["difficulty"] not in [
                "very easy",
                "easy",
                "medium",
                "hard",
                "very hard",
            ]:
                response_dict["difficulty"] = salvage_tag(response_dict["difficulty"])

            # Step 3.e.iii: Validate format
            if not isinstance(response_dict["difficulty"], str):
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Invalid Format type in response. type: {type(response_dict['difficulty'])} value: {response_dict['difficulty']}",
                    response=response,
                )
                return instance

            # Step 3.e.iv: Save
            if "intent" in response_dict:
                instance.magpie_tags["intent"] = [
                    {f"{model_name}": response_dict["intent"]}
                ]
            if "knowledge" in response_dict:
                instance.magpie_tags["knowledge"] = [
                    {f"{model_name}": response_dict["knowledge"]}
                ]
            instance.magpie_tags["difficulty"] = [
                {f"{model_name}": response_dict["difficulty"]}
            ]
            instance.magpie_tags["metadata"]["label_model"] = [model_name]

        # Step 3.f: Process for "classification" tag task
        elif tag_task == "classification":
            # Step 3.f.i: Validate
            if "primary_tag" not in response_dict:
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Failed to rate for {tag_task} task. Defaulting to None.",
                    response=response,
                )
                return instance

            # Step 3.f.ii: Validate format
            if "primary_tag" not in response_dict or not isinstance(
                response_dict["primary_tag"], str
            ):
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Invalid Format type in response. type: {type(response_dict['primary_tag'])} value: {response_dict['primary_tag']}",
                    response=response,
                )
                return instance

            # Step 3.f.iii: Save
            instance.magpie_tags["task_category"] = [
                {f"{model_name}": response_dict["primary_tag"]}
            ]
            instance.magpie_tags["metadata"]["label_model"] = [model_name]

        # Step 3.g: Process for "conversation_quality" tag task
        elif tag_task == "conversation_quality":
            # Step 3.g.i: Validate format
            if "score" not in response_dict:
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Failed to rate for {tag_task} task. Defaulting to None.",
                    response=response,
                )
                return instance

            # Step 3.g.ii: Salvage tag if it is not in valid list
            if response_dict["score"] not in ["1", "2", "3", "4", "5"]:
                response_dict["score"] = salvage_tag(response_dict["score"])

            # Step 3.g.iii: Validate format
            if not (
                (
                    isinstance(response_dict["score"], str)
                    and response_dict["score"].isdigit()
                )
                or isinstance(response_dict["score"], (int, float))
            ):
                set_error_values(
                    instance.magpie_tags,
                    tag_task,
                    model_name,
                    error=f"Invalid Format type in response. type: {type(response_dict['score'])} value: {response_dict['score']}",
                    response=response,
                )
                return instance

            # Step 3.g.iv: Save
            instance.magpie_tags["conversation_score"] = [
                {f"{model_name}": str(response_dict["score"])}
            ]

            if "explanation" in response_dict:
                instance.magpie_tags["conversation_explanation"] = [
                    {f"{model_name}": response_dict["explanation"]}
                ]
            instance.magpie_tags["metadata"]["label_model"] = [model_name]

        return instance

    # ===========================================================================
    #                       MAIN PROCESS
    # ===========================================================================
    def execute(
        self, inputs: List[MagpieTaggerBlockData], *args, **kwargs
    ) -> List[MagpieTaggerBlockData]:
        """
        Top-level process method to perform Magpie tagging

        Args:
            inputs (List[MagpieTaggerBlockData]): inputs to be tagged

        Returns:
            List[MagpieTaggerBlockData]: tagged inputs
        """
        # Step 1: Prepare dictionary containing prompt, input for every task
        instances = []

        # Warning about roles and content not supported
        dgt_logger.warning(
            "In the case of multi-turn only messages with the field 'content'/'text'/'value' are supported by Magpie. Others will be skipped "
        )
        dgt_logger.warning(
            "In the case of multi-turn only 'user' and 'assistant' messages are supported by Magpie. Others will be skipped "
        )

        for item_idx, item in enumerate(inputs):
            # Step 1: Validate necessary fields are provided
            if (item.magpie_input is None) and (item.magpie_mt_input is None):
                dgt_logger.warning(
                    "Failed to tag synthetic data due to missing 'magpie_input' & 'magpie_output' OR 'magpie_mt_input' field. Make sure to update input_map field in the databuilder yaml under magpie_tagger block.",
                )
                continue
            else:
                if item.magpie_input is not None and item.magpie_output is None:
                    dgt_logger.warning(
                        "magpie_output is missing. Make sure to update input_map field in the databuilder yaml under magpie_tagger block.",
                    )

                for task in self.tasks:
                    if item.magpie_mt_input:
                        try:
                            # Multi-turn input
                            if len(item.magpie_mt_input) > 2 and task in [
                                "quality",
                                "sample_quality",
                                "conversation_quality",
                            ]:
                                prompt = self.template_generator_mt(
                                    conversation=item.magpie_mt_input,
                                    task=task,
                                )
                            else:  # Single-turn input
                                user_utterance_texts = []
                                assistant_utterance_texts = []

                                # extract user and assistant utterance texts
                                for utterance in item.magpie_mt_input:
                                    # Identify role field
                                    if "from" in utterance:
                                        role_field = "from"
                                    elif "role" in utterance:
                                        role_field = "role"
                                    elif "speaker" in utterance:
                                        role_field = "speaker"
                                    else:
                                        raise ValueError(
                                            "conversation should have a 'from' field or a 'role' field or a 'speaker' field to signify whether it was a user or assistant utterance"
                                        )

                                    # Identify text/content field
                                    if "value" in utterance:
                                        txt_field = "value"
                                    elif "content" in utterance:
                                        txt_field = "content"
                                    elif "text" in utterance:
                                        txt_field = "text"
                                    else:
                                        # Skipping messages without content
                                        continue

                                    if utterance[role_field] == "user":
                                        user_utterance_texts.append(
                                            utterance[txt_field]
                                        )
                                    elif utterance[role_field] in [
                                        "assistant",
                                        "agent",
                                    ]:
                                        assistant_utterance_texts.append(
                                            utterance[txt_field]
                                        )
                                    else:
                                        # Skipping roles that are not 'user' or 'assistant'
                                        continue

                                if len(user_utterance_texts) < 1:
                                    dgt_logger.warning("Missing user utterances")
                                    continue

                                if (
                                    len(user_utterance_texts)
                                    != len(assistant_utterance_texts)
                                ) and len(user_utterance_texts) > 1:
                                    dgt_logger.warning(
                                        "Mismatch in number of user (%d) and assistant (%d) utterances",
                                        len(user_utterance_texts),
                                        len(assistant_utterance_texts),
                                    )
                                    continue

                                if len(assistant_utterance_texts) < 1:
                                    assistant_utterance_texts = [None]

                                prompt = self.template_generator(
                                    input_text=user_utterance_texts[0],
                                    task=task,
                                    output_text=assistant_utterance_texts[0],
                                )
                        except KeyError as err:
                            dgt_logger.warning(
                                f"Failed to tag multi-turn synthetic data due to missing {err.args[0]} field."
                            )
                    else:
                        prompt = self.template_generator(
                            input_text=item.magpie_input,
                            task=task,
                            output_text=item.magpie_output,
                        )

                    instances.append(
                        {
                            "input": prompt,
                            "src": item,
                            "task": task,
                            "index": item_idx,
                        }
                    )

        dgt_logger.info(
            "The total number of samples in the progress bar is no. of tasks (%d) x total no. of inputs (%d) ",
            len(self.tasks),
            len(inputs),
        )
        # Step 2: Run prompts
        outputs = self._llm_generator(
            instances,
            input_map={"input": "input", "task": "task", "index": "index"},
            **kwargs,
        )

        # Step 3: Process results
        results = []

        # Step 3.a: Group outputs by index (input)
        outputs_per_input = {}
        for output in outputs:
            try:
                outputs_per_input[output["index"]].append(output)
            except KeyError:
                outputs_per_input[output["index"]] = [output]

        # Step 3.b: Process outputs for an input
        for llm_outputs in outputs_per_input.values():
            # Step 3.b.i: Initialize response instance
            instance: MagpieTaggerBlockData = llm_outputs[0]["src"]
            if instance.magpie_tags is None:
                instance.magpie_tags = {}

            # Step 3.b.ii: Collate all outputs
            for llm_output in llm_outputs:
                # Step 3.b.ii.*: Fetch LLM output
                model_response = llm_output["result"].strip()

                # Step 3.b.ii.**: Expected response in dictionary format.
                #           Identify starting and ending character index for a first dictionary in LLM output
                start_idx, end_idx = extract_first_enclosed_braces(model_response)

                # Step 3.b.ii.***: Skip processing, if failed to find dictionary
                if start_idx is None or end_idx is None:
                    continue
                else:
                    model_response = model_response[start_idx : end_idx + 1]

                # Step 3.b.ii.****: Parse response
                self.parse(model_response, instance, llm_output["task"])

            # # Step 3.b.iii: Add result
            results.append(instance)

        return results
