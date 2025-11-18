# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Local
from fms_dgt.base.task import DataPoint, TransformationTask


@dataclass(kw_only=True)
class MagpieTransformData(DataPoint):
    input: List[Dict[str, Any]]
    magpie_tags: Optional[List[Dict[str, Any]]] = None


class MagpieTransformTask(TransformationTask):

    INPUT_DATA_TYPE = MagpieTransformData
    OUTPUT_DATA_TYPE = MagpieTransformData

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(task_name=self.name, input=kwargs)

    def get_batch_examples(self) -> List[DataPoint]:
        """Returns batch of examples from dataloader.

        Returns:
            List[DataPoint]: List of examples to be used by SDG process.
        """
        examples = []

        # Load examples from seed data loader sequentially
        while True:
            example = self.get_example()
            if example is None:
                break
            examples.append(example)

        # Return
        return examples

    def save_final_data(self):
        loaded_data = self.datastore.load_data() or []
        if loaded_data:
            self.final_datastore.save_data(
                [
                    {**data_point["input"], "magpie_tags": data_point["magpie_tags"]}
                    for data_point in loaded_data
                ]
            )
