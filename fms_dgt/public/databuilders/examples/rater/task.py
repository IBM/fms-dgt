# Local
from fms_dgt.base.task import TransformationTask
from fms_dgt.public.databuilders.examples.rater.data_objects import (
    InputData,
    OutputData,
)


class RatingTask(TransformationTask):

    INPUT_DATA_TYPE = InputData
    OUTPUT_DATA_TYPE = OutputData

    def instantiate_input_example(self, **kwargs) -> InputData:
        """
        This helper method is called automatically on each data point provided under `data` field to the tranformation task.

        By default, it will try to instantiate object of `INPUT_DATA_TYPE` dataclass from each loaded data point. But, the
        databuilder developer has ability to change the default behavior via overriding this method.

        Returns:
            InputData: object of `InputData` from data
        """
        return InputData(
            task_name=self.name,
            is_seed=True,
            question=kwargs.get("question"),
            answer=kwargs.get("answer"),
        )
