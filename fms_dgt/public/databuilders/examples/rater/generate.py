# Standard
from pathlib import Path
from typing import Any, Dict, List
import ast
import json
import re

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.public.databuilders.examples.rater.data_objects import (
    InputData,
    OutputData,
)
from fms_dgt.public.databuilders.examples.rater.task import RatingTask


# NOTE: we register the data builder with the below decorator so that we can reference it in an input data file later on
@register_data_builder("public/examples/qa_rater")
class RatingDataBuilder(TransformationDataBuilder):
    """Rating geographica question-answer pairs for complexity"""

    TASK_TYPE: RatingTask = RatingTask

    # NOTE: rator is the language model that we will use to rate the synthetic examples
    rater: LMProvider

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self._prompts = {}
        for template_path in Path(Path(__file__).parent, "prompt_templates").glob(
            "*.txt"
        ):
            self._prompts[template_path.name[:-4]] = JinjaPromptTemplate(
                template_path=template_path
            )

    def __call__(self, data_points: List[InputData]) -> List[OutputData]:
        # Build rater inputs
        rater_inputs: List[Dict] = []
        for data_point in data_points:
            # Build rater inputs
            # input (str | List[Dict[str, Any]]): (Reserved field) prompt to be passed to `/completion` endpoint or messages to be passed to `/chat/completion` endpoint
            # gen_kwargs (Optional[Dict[str, Any]]): (Reserved field) Additional generation specific parameters to be passed to `/completion` or `/chat/completion` endpoint
            # reference (Optional[Any]): We recommend passing data used to build prompt for future use. DiGiT returns all non-reserved field in output from a block.
            rater_inputs.append(
                {
                    "input": self._prompts["judge"].encode(
                        render_dict={
                            "question": data_point.question,
                            "answer": data_point.answer,
                        }
                    ),
                    "reference": data_point,
                }
            )

        # Execute block
        # LMProvider block is optimized to perform asynchronous invocation of `/completion` or `/chat/completion` endpoint to enable batch processing.
        rater_outputs = self.rater(rater_inputs)

        # Process outputs from block
        outputs = []
        for rater_output in rater_outputs:
            # Extract data point passed to LMProvider block
            data_point = rater_output["reference"]

            # LMProvider block return output from `/completion` or `/chat/completion` endpoint in "result" field.
            ratings = self.parse(response=rater_output["result"])

            # Add to outputs
            outputs.append(
                OutputData(
                    task_name=data_point.task_name,
                    is_seed=False,
                    question=data_point.question,
                    answer=data_point.answer,
                    ratings=ratings,
                )
            )

        # Return rated data points
        return outputs

    # ===========================================================================
    #                       UTILITY FUNCTIONS
    # ===========================================================================
    def parse(self, response: str):
        """
        Parse language model response to prepare final tagged outputs

        Args:
            response (str): generated response to parse
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

        # Clean response
        response = clean(response)

        # Try to convert response to dictionary, otherwise try to salvage from explanation string
        try:
            response_dict = json.loads(response)
            if isinstance(response_dict, list):
                response_dict = response_dict[0]

            # Salvage response, if necessary
            if "difficulty" not in response_dict:
                response_dict["difficulty"] = salvage_response(response, "difficulty")

        except json.decoder.JSONDecodeError:
            response_dict = {}
            response_dict["difficulty"] = salvage_response(response, "difficulty")
            response_dict["knowledge"] = None

        # Salvage value, if necessary
        if response_dict["difficulty"] not in [
            "very easy",
            "easy",
            "medium",
            "hard",
            "very hard",
        ]:
            response_dict["difficulty"] = salvage_tag(response_dict["difficulty"])

        # Return
        return response_dict
