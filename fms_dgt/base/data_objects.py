# Standard
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List
import json
import os

# Local
from fms_dgt.utils import validate_block_sequence


# ===========================================================================
#                       DATA
# ===========================================================================
@dataclass
class DataPoint:
    task_name: str
    is_seed: bool = False  # whether or not a particular example is a seed example

    def to_dict(self) -> Dict:
        """Returns output dictionary representation of dataclass. Designed to be overridden with custom logic.

        Returns:
            Dict: Dictionary representation of dataclass
        """
        return asdict(self)

    @classmethod
    def get_field_names(cls):
        return [field.name for field in fields(cls)]


# ===========================================================================
#                       BLOCK DATA
# ===========================================================================
@dataclass
class BlockData:
    """Internal data type for Block

    Attributes:
        SRC_DATA (Any): This attribute is used to store the original data. It SHOULD NOT be overwritten
        store_names (Optional[List[str]]): This attribute is used to provide datastore names to identify appropriate datastore.
    """

    SRC_DATA: Any
    store_names: List[str] | None = None

    def dict_factory(self, data) -> Dict:
        """
        This method is used to serialize data point in `asdict()` method.

        Args:
            data (DataPoint): data point to serialize

        Returns:
            Dict: serializable data point
        """
        result = {}
        for k, v in data:
            if isinstance(v, dict):
                result[k] = self.dict_factory(v.items())
            else:
                try:
                    json.dumps(v)
                    result[k] = v
                except (TypeError, json.JSONDecodeError):
                    continue

        return result

    def to_dict(self):
        return asdict(self, dict_factory=self.dict_factory)


@dataclass(kw_only=True)
class ValidatorBlockData(BlockData):
    """Default class for base validator data

    Attributes:
        is_valid (Optional[bool]): Whether or not the particular data passed to the validator block is valid
        metadata (Optional[dict]): Additional metadata
    """

    is_valid: bool | None = None
    metadata: Dict | None = None


# ===========================================================================
#                       TASK
# ===========================================================================
@dataclass
class TaskRunnerConfig:
    """Base configuration

    Attributes:
        output_dir (Optional[str]): The directory where the generated outputs will be saved.
        save_formatted_output (Optional[bool]): A boolean indicating whether to save outputs that have been reformatted
        restart_generation (Optional[bool]): A boolean indicating whether to restart generation from scratch.
    """

    output_dir: str | None = None
    save_formatted_output: bool | None = False
    restart_generation: bool | None = False

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = os.getenv("DGT_OUTPUT_DIR", "output")


@dataclass
class GenerationTaskRunnerConfig(TaskRunnerConfig):
    """Configuration for a generation task

    Attributes:
        seed_batch_size (Optional[int]): The batch size used for seed examples.
        machine_batch_size (Optional[int]): The batch size used for machine examples.
        num_outputs_to_generate (Optional[int]): The number of outputs to generate.
    """

    seed_batch_size: int | None = None
    machine_batch_size: int | None = None
    num_outputs_to_generate: int | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.num_outputs_to_generate is None:
            self.num_outputs_to_generate = 2

        if self.seed_batch_size is None:
            self.seed_batch_size = min(100, self.num_outputs_to_generate)

        if self.machine_batch_size is None:
            self.machine_batch_size = 10


@dataclass
class TransformationTaskRunnerConfig(TaskRunnerConfig):
    """Configuration for a transformation task

    Attributes:
        transform_batch_size (Optional[int]): The batch size used for data to transform.
    """

    transform_batch_size: int | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.transform_batch_size is None:
            self.transform_batch_size = 100


# ===========================================================================
#                       DATA BUILDER
# ===========================================================================
@dataclass
class DataBuilderConfig(dict):
    """Configuration for a data builder.

    Attributes:
        name (Optional[str]): The name of the data builder.
        blocks (Optional[List[Dict]]): A list of block configurations.
        postprocessors (Optional[List[str]]): A list of names of the blocks that should be used during postprocessing.
        metadata (Optional[Dict[str, Any]]): Metadata for the data builder. Allows for users to pass arbitrary info to data builders.
    """

    name: str | None = None
    blocks: dict | None = None
    postprocessors: List[str | Dict] | None = None
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if self.blocks is None:
            self.blocks = []

        if self.postprocessors is None:
            self.postprocessors = []

        validate_block_sequence(self.postprocessors)
