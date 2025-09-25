"""
In DiGiT, `Task` is one of the fundamental concept. There are two kinds of task natively supported in DiGiT.

1. Generation Task: These tasks create new synthetic data from scratch. For example, question answers pairs over different knowledge sources,
multi-turn task oriented conversations, tool calling demontrations etc. Typically end-user wish to generate `N` number of synthetic data points for
one or many generation task.

2. Transformation Task: These tasks extend, shrink or alter data. For example, quality rater marking each data point with some criteria, changing formats etc.

Both `Generation` and `Transformation` tasks extend base `Task` class which controls almost all of the I/O operations. For example, loading seed data, saving synthetic data etc.
Additionally task constructor is an easiest way for a databuilder developers to expose customizability.
"""

# Standard

# Standard
from typing import Any

# Local
from fms_dgt.base.task import GenerationTask
from fms_dgt.public.databuilders.examples.qa.data_objects import GeographyQAData


class GeographyQATask(GenerationTask):
    """
    In this example, we wish to create 50 geographical question answer pairs. Hence, we choose to extend from `GenerationTask`.
    """

    # We must always specify both the type of data that will be accepted as well as the type of data that will be generated
    # For our example, we will be providing some seed examples to large languge model to create new synthetic data in the similar format.
    # Therefore, our `INPUT_DATA_TYPE` and `OUTPUT_DATA_TYPE` are identical.
    #
    # CAUTION: Be careful when you use different `INPUT_DATA_TYPE` and `OUTPUT_DATA_TYPE`. By default, `GenerationTask` type task are expected
    # to keep looping over a mixture of seed and synthetic data till it produces requested number of synthetic data points.

    INPUT_DATA_TYPE = GeographyQAData
    OUTPUT_DATA_TYPE = GeographyQAData

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Additional arguments specified in this constructor can be set using associated `task.yaml`
        """
        super().__init__(*args, **kwargs)

    def instantiate_input_example(self, **kwargs) -> GeographyQAData:
        """
        This helper method is called automatically on each seed data point provided to the generation task.

        By default, it will try to instantiate object of `INPUT_DATA_TYPE` dataclass from each loaded data point. But, the
        databuilder developer has ability to change the default behavior via overriding this method.

        Returns:
            GeographyQAData: object of `GeographyQAData` from seed data
        """
        return GeographyQAData(
            task_name=self.name,
            is_seed=True,
            question=kwargs.get("question"),
            answer=kwargs.get("answer"),
        )
