# Standard
from typing import Any, Dict, Mapping, Union

# Local
from fms_dgt.base.task import TransformationTask, TransformationTaskRunnerConfig
from fms_dgt.public.databuilders.time_series.data_objects import (
    TimeSeriesInputData,
    TimeSeriesOutputData,
)
from fms_dgt.utils import init_dataclass_from_dict


# NOTE: this class holds the information needed for the overall time series generation task
class TimeSeriesTask(TransformationTask):

    # We must always specify both the type of data that will be accepted as well as the type of data that will be generated
    INPUT_DATA_TYPE = TimeSeriesInputData
    OUTPUT_DATA_TYPE = TimeSeriesOutputData

    def __init__(
        self,
        *args: Any,
        runner_config: Union[Mapping, TransformationTaskRunnerConfig] = None,
        data_params: Dict[str, Any],
        sdforger_params: Dict[str, Any],
        **kwargs: Any,
    ):
        runner_config = init_dataclass_from_dict(runner_config, TransformationTaskRunnerConfig)
        runner_config.transform_batch_size = data_params.get("train_length", 5000)
        runner_config.restart_generation = True
        self.data_params = data_params
        self.sdforger_params = sdforger_params
        super().__init__(*args, runner_config=runner_config, **kwargs)

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            task_description=self.task_description,
            observations=kwargs,
        )
