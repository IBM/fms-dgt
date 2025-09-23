# Standard
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Local
from fms_dgt.base.block import ValidatorBlock
from fms_dgt.base.data_objects import ValidatorBlockData
from fms_dgt.base.registry import register_block


# ===========================================================================
#                       MAGPIE FILTER BLOCK DATA TYPE
# ===========================================================================
@dataclass(kw_only=True)
class MagpieFilterBlockData(ValidatorBlockData):
    """
    Data type for Magpie tagging block
    """

    field: Dict

    # UUID for each entry
    id: Optional[str] = None

    magpie_input: Optional[str] = None
    magpie_output: Optional[str] = None

    # For multi-turn setting, the input must be array of dictionaries
    # with each item in dictionary containing "text"/"content" field
    magpie_mt_input: Optional[List[Dict[str, Any]]] = None


# ===========================================================================
#                       MAGPIE HELPER FUNCTIONS
# ===========================================================================
def flatten(instance: MagpieFilterBlockData) -> MagpieFilterBlockData:
    """
    Flatten scores to only contain labels assigned by first model (e.g., mistralai/Mistral-7B-Instruct-v0.3)

    Args:
        instance (MagpieFilterBlockData): data to simplify

    Returns:
        MagpieData: data with simplified scores
    """
    # Create a copy
    flattened_instance = deepcopy(instance)

    # Unpack labels assigned by one model (e.g., mistralai/Mistral-7B-Instruct-v0.3) for simplicity, if present
    if (
        flattened_instance.field
        and "input_quality" in flattened_instance.field
        and flattened_instance.field["input_quality"]
    ):
        flattened_instance.field["input_quality"] = flattened_instance.field[
            "input_quality"
        ][0].popitem()[1]

    if (
        flattened_instance.field
        and "difficulty" in flattened_instance.field
        and flattened_instance.field["difficulty"]
    ):
        flattened_instance.field["difficulty"] = flattened_instance.field["difficulty"][
            0
        ].popitem()[1]

    if (
        flattened_instance.field
        and "judge_quality_score" in flattened_instance.field
        and flattened_instance.field["judge_quality_score"]
    ):
        flattened_instance.field["judge_quality_score"] = flattened_instance.field[
            "judge_quality_score"
        ][0].popitem()[1]

    if (
        flattened_instance.field
        and "task_category" in flattened_instance.field
        and flattened_instance.field["task_category"]
    ):
        flattened_instance.field["task_category"] = flattened_instance.field[
            "task_category"
        ][0].popitem()[1]

    return flattened_instance


def has_valid_scores(instance: MagpieFilterBlockData) -> Tuple[bool, Dict | None]:
    if instance.field and "input_quality" in instance.field:
        if instance.field["input_quality"] not in [
            "very poor",
            "poor",
            "average",
            "good",
            "excellent",
        ]:
            return (
                False,
                {
                    "reason": f'Invalid value: {instance.field["input_quality"]} for "input_quality" score'
                },
            )

    if instance.field and "difficulty" in instance.field:
        if instance.field["difficulty"] not in [
            "very easy",
            "easy",
            "medium",
            "hard",
            "very hard",
        ]:
            return (
                False,
                {
                    "reason": f'Invalid value: {instance.field["difficulty"]} for "difficulty" score'
                },
            )

    if instance.field and "judge_quality_score" in instance.field:
        if instance.field["judge_quality_score"] not in ["1", "2", "3", "4", "5"]:
            return (
                False,
                {
                    "reason": f'Invalid value: {instance.field["judge_quality_score"]} for "judge_quality_score" score'
                },
            )

    if (
        instance.field
        and "task_category" in instance.field
        and instance.field["task_category"]
    ):
        if len(instance.field["task_category"]) < 1:
            return (
                False,
                {
                    "reason": f'Invalid value: {instance.field["task_category"]} for "task_category"'
                },
            )

    return True, None


def has_valid_data(instance: MagpieFilterBlockData) -> Tuple[bool, Dict | None]:
    if not instance.magpie_input and not instance.magpie_mt_input:
        return (
            False,
            {
                "reason": 'Either "magpie_input" or "magpie_mt_input" field must be specified.',
            },
        )

    if instance.magpie_mt_input:
        user_utterance_texts = []
        assistant_utterance_texts = []

        # Extract user and assistant utterance texts
        for utterance in instance.magpie_mt_input:
            # Identify role field
            if "from" in utterance:
                role_field = "from"
            elif "role" in utterance:
                role_field = "role"
            elif "speaker" in utterance:
                role_field = "speaker"
            else:
                return (
                    False,
                    {
                        "reason": f'Failed to identify role for uttereance ({utterance}). Allowed role field keys are "from", "role" or "speaker"',
                    },
                )

            # Identify text/content field
            if "value" in utterance:
                txt_field = "value"
            elif "content" in utterance:
                txt_field = "content"
            elif "text" in utterance:
                txt_field = "text"
            else:
                return (
                    False,
                    {
                        "reason": f'Failed to identify text for uttereance ({utterance}). Allowed text field keys are "value", "content" or "text"',
                    },
                )

            if utterance[role_field] == "user":
                if len(utterance[txt_field]) > 0:
                    user_utterance_texts.append(utterance[txt_field])
            elif utterance[role_field] in [
                "assistant",
                "agent",
            ]:
                if len(utterance[txt_field]) > 0:
                    assistant_utterance_texts.append(utterance[txt_field])
            elif utterance[role_field] not in ["system", "developer", "tool"]:
                return (
                    False,
                    {
                        "reason": f'Unsupported role for utterance ({utterance}). Allowed values are "user", "assistant", "agent", "system", "developer" and "tool"',
                    },
                )

        if len(user_utterance_texts) < 1:
            return (
                False,
                {
                    "reason": "No user utterance[s]",
                },
            )

        if (
            len(user_utterance_texts) != len(assistant_utterance_texts)
            and len(user_utterance_texts) > 1
        ):
            return (
                False,
                {
                    "reason": f"Mismatched number of user utterances ({len(user_utterance_texts)}) and assistant utterances ({len(assistant_utterance_texts)})",
                },
            )
    else:
        if not instance.magpie_input:
            return (
                False,
                {
                    "reason": 'Empty value for "magpie_input"',
                },
            )
        if instance.magpie_output and len(instance.magpie_output) == 0:
            return (
                False,
                {
                    "reason": 'Missing value for "magpie_output"',
                },
            )

    return True, None


def is_qualified(
    instance: MagpieFilterBlockData,
    remove_duplicates: bool,
    filter_criteria: Optional[Dict] = None,
) -> Tuple[bool, Dict | None]:
    if filter_criteria:
        input_quality_requirements = filter_criteria.get(
            "input_quality", ["good", "excellent"]
        )
        judge_quality_score_requirements = filter_criteria.get("sample_quality", ["5"])
        difficulty_requirements = filter_criteria.get(
            "difficulty",
            [
                "very easy",
                "easy",
                "medium",
                "hard",
                "very hard",
            ],
        )
    else:
        input_quality_requirements = ["good", "excellent"]
        judge_quality_score_requirements = ["5"]
        difficulty_requirements = [
            "very easy",
            "easy",
            "medium",
            "hard",
            "very hard",
        ]

    if instance.field and "input_quality" in instance.field:
        if instance.field["input_quality"] not in input_quality_requirements:
            return False, {"reason": "Failed to meet input quality requirements"}

    if instance.field and "judge_quality_score" in instance.field:
        if (
            instance.field["judge_quality_score"]
            not in judge_quality_score_requirements
        ):
            return False, {"reason": "Failed to meet judge quality requirements"}

    if instance.field and "difficulty" in instance.field:
        if instance.field["difficulty"] not in difficulty_requirements:
            return False, {"reason": "Failed to meet difficulty requirements"}

    if instance.field and "min_similar_uuid" in instance.field and remove_duplicates:
        if (
            instance.field["min_similar_uuid"] is not None
            and instance.id != instance.field["min_similar_uuid"]
        ):
            return False, {"reason": "Duplicate instance"}

    return True, None


# ===========================================================================
#                       MAGPIE FILTER BLOCK
# ===========================================================================
@register_block("magpie_filter")
class MagpieFilter(ValidatorBlock):
    r"""Class for Magpie based validator

    Args:
        filter_type (List[str]): types of filtering to apply. Possible values are `invalid_scores`, `high_quality_filter` and `all`.
        filter_criteria (Optional[dict]): validity criteria. Defaults to `{"input_quality": ["good", "excellent"], "sample_quality": ["5"]}`
        remove_duplicates (Optional[bool]): Should remove duplicate instances. Defaults to `true`

    .. code-block:: python

        # Initialize filter
        filter = MagpieFilter()

        # Sample data
        data = [
                {
                    "id": "UUID 1",
                    "question": "what is capital of the United States of America?",
                    "answer": "Washington D.C",
                    "field": {
                        "input_quality": [{"Model A": "poor"}, {"Model B": "very poor"}],
                        "difficulty": [{"Model A": "hard"}, {"Model B": "medium"}],
                        "judge_quality_score": [{"Model A": "5"}, {"Model B": "3"}],
                        "task_category": [{"Model A": "Task 1"}, {"Model B": "Task 1"}]
                    }
                },
                {
                    "id": "UUID 2",
                    "question": "What is biggest star in our solar system?",
                    "answer": "Sun is the biggest star in our solar system.",
                    "field": {
                        "input_quality": [{"Model A": "good"}, {"Model B": "excellent"}],
                        "difficulty": [{"Model A": "medium"}, {"Model B": "medium"}],
                        "judge_quality_score": [{"Model A": "5"}, {"Model B": "4"}],
                        "task_category": [{"Model A": "Task 2"}, {"Model B": "Task 2"}]
                    }
                }
            ]

        # Run filtering
        filter(data)
    """

    DATA_TYPE = MagpieFilterBlockData

    def __init__(
        self,
        filter_type: List[str],
        filter_criteria: Optional[Dict] = None,
        remove_duplicates: Optional[bool] = True,
        input_map: Optional[Union[List, Dict]] = None,
        **kwargs: Any,
    ) -> None:
        # Set default values for "input_map", if necessary
        if input_map is None:
            input_map = {
                "question": "magpie_input",
                "answer": "magpie_output",
                "messages": "magpie_mt_input",
                "id": "id",
                "tags": "field",
            }

        # Initialize parent
        super().__init__(input_map=input_map, **kwargs)

        self._filter_type = filter_type
        self._filter_criteria = filter_criteria
        self._remove_duplicates = remove_duplicates

    def _validate(
        self, instance: MagpieFilterBlockData, *args: Any, **kwargs: Any
    ) -> bool | Tuple[bool, Dict | None]:
        # Flatten instance
        flattened_instance = flatten(instance=instance)

        # Validate data
        valid, metadata = has_valid_data(instance=flattened_instance)
        if not valid:
            return False, metadata

        # Validate score, if necessary
        if self._filter_type in ["all", "invalid_scores"]:
            valid, metadata = has_valid_scores(instance=flattened_instance)
            if not valid:
                return False, metadata

        # Validate qualification, if necessary
        if self._filter_type in ["all", "high_quality_filter"]:
            valid, metadata = is_qualified(
                instance=flattened_instance,
                remove_duplicates=self._remove_duplicates,
                filter_criteria=self._filter_criteria,
            )

            if not valid:
                return False, metadata

        return True
