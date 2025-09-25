# Standard
from typing import Any, Dict, List, Optional

# Local
from fms_dgt.base.task import GenerationTask
from fms_dgt.core.databuilders.simple.data_objects import SimpleData


class SimpleTask(GenerationTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = SimpleData
    OUTPUT_DATA_TYPE = SimpleData

    def __init__(
        self,
        *args,
        seed_datastore: Optional[Dict] = None,
        seed_examples: Optional[List[Any]] = None,
        **kwargs,
    ):
        # Step 1: Raise error if seed examples or seed datastore are not specified
        if (seed_examples is None or not seed_examples) and seed_datastore is None:
            raise ValueError(
                "Missing mandatory value for seed_examples or seed_datastore. Please provide at least one seed example in the task.yaml file before running."
            )

        # Step 2: Initialize parent
        super().__init__(
            *args, seed_datastore=seed_datastore, seed_examples=seed_examples, **kwargs
        )

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            taxonomy_path=self.name,
            task_description=self.task_description,
            instruction=kwargs.get("question", kwargs.get("instruction")),
            input=kwargs.get("context", kwargs.get("input", "")),
            output=kwargs.get("answer", kwargs.get("output")),
            document=kwargs.get("document", None),
        )
