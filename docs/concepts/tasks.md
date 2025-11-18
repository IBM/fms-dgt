# Tasks

Tasks are simply classes which allows developers to define required assets (e.g., schemas, seed data, models), stopping criteria (e.g., number of records, time limits, quality thresholds), output data formatter as well as other high-level specifications that govern the overall data creation process.

All Tasks in DGT must inherit from [`Task`](https://github.com/IBM/fms-dgt/blob/ec3ce21f341bf9938426a082b1e63431da031e03/fms_dgt/base/task.py#L49) base class. Tasks are automatically instantiated using information provided in a configuration YAML file.

The base [`Task`](https://github.com/IBM/fms-dgt/blob/ec3ce21f341bf9938426a082b1e63431da031e03/fms_dgt/base/task.py#L49) class is intentionally minimal to avoid being overly prescriptive or burdensome for developers.

### Data Initialization

The Task class is responsible for initializing Datapoint objects, which are then passed to the data builder's `__call__` method.

#### Input

By default, DGT uses the instantiate_input_example method to create input examples. This method receives kwargs constructed from either:

- A combination of seed_examples and synthetic_examples (for generation tasks), or
- Individual data instances (for transformation tasks)

```python
def instantiate_input_example(self, **kwargs: Any) -> INPUT_DATA_TYPE:
        """Instantiate an input example for this task. Designed to be overridden with custom initialization.

        Args:
            kwargs (Dict, optional): Kwargs used to instantiate an input example object.

        Returns:
            INPUT_DATA_TYPE: An instance of INPUT_DATA_TYPE.
        """
        return self.INPUT_DATA_TYPE(task_name=kwargs.pop("task_name", self.name), **kwargs)
```

`INPUT_DATA_TYPE` is a dataclass that inherits from [Datapoint](https://github.com/IBM/fms-dgt/blob/ec3ce21f341bf9938426a082b1e63431da031e03/fms_dgt/base/data_objects.py#L15) and is defined as a class variable in task.py.

#### Output

Similarly, output examples are created using the instantiate_output_example method. It receives kwargs built from the data returned by the data builder.

```python
def instantiate_output_example(self, **kwargs: Any) -> OUTPUT_DATA_TYPE:
        """Instantiate an output example for this task. Designed to be overridden with custom initialization.

        Args:
            kwargs (Dict, optional): Kwargs used to instantiate an output example object.

        Returns:
            OUTPUT_DATA_TYPE: An instance of OUTPUT_DATA_TYPE.
        """
        return self.OUTPUT_DATA_TYPE(**kwargs)
```

`OUTPUT_DATA_TYPE` is also a dataclass that inherits from [Datapoint](https://github.com/IBM/fms-dgt/blob/ec3ce21f341bf9938426a082b1e63431da031e03/fms_dgt/base/data_objects.py#L15) and is defined as a class variable in task.py.

### Stopping Criteria

DGT uses sensible default stopping criteria depending on the task type, but developers are encouraged to override them as needed.

- `Generation Tasks`: Completes once the minimum requested number of outputs is generated. Note that this number acts as a lower bound—depending on the data builder's design, the final number of samples may exceed this value.

- `Transformation Tasks`: Completes after a single pass over all data to be transformed.

To customize this behavior, developers can override the is_complete method.

??? warning - `is_complete` function does not take any arguments

### Saving Data

In DGT, the Task determines how and when data is saved. By default, DGT stores various types of data in a directory specified by the `DGT_OUTPUT_DIR` environment variable. If not set, it defaults to an output folder at the `root` of the repository. Data is saved using [datastores](./datastores.md), and includes the following

- `task_card`: Contains all configuration details passed to the run.
- `data`: Generated during the iterative loop execution. Each loop's output is appended and stored as intermediate data, saved before any post-processing occurs.
- `final_data`: The validated and post-processed data. If no post-processor or validator is defined, this will be identical to `data`.
- `postproc_data_*`: Data points after post-processing steps.
- `formatted_data`: If a custom formatter is specified, this contains the final_data after formatting.
- `task_results`: Logs task execution metadata such as start/end time, process ID, number of data points processed, and any custom metrics.

You can override the default datastore behavior via the `YAML` configuration file. For example:

```{.yaml hl_lines=12-14}
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: <TASK_NAME>
task_description: <TASK_DESCRIPTION>
created_by: <CREATOR>
data_builder: <DATABUILDER_NAME> # Must match exactly with the databuilder name

######################################################
#                   RESERVED FIELDS
######################################################
datastore:
    type: default
    output_data_format: jsonl | parquet # (1)!
```

### Fields

DGT's runtime automatically initializes `Task` objects from task configuration YAML files. It does this by passing the contents of the YAML configuration as `kwargs` to the `Task` constructor.

#### Mandatory Fields

- `task_name (str)`: A unique identifier for the task. Recommend following directory structure as naming convention.
- `task_description (str)`: A detailed explanation of the use case.
- `created_by (str)`: Information about the task creator.
- `data_builder (str)`: Specifies the data builder to be used.
- `seed_examples (List[dict] | None)`: In-context learning (ICL) examples used during the generative cycle. Applicable only when extending from GenerationTask.
- `seed_datastore (Dict | None)`: Configuration for a datastore containing ICL examples used in the generative cycle. Applicable only when extending from GenerationTask.
- `data (List[dict] | Dict)`: Input data to be transformed. Can be a list of dictionaries or a datastore configuration. Applicable only when extending from TransformationTask.

#### Reserved Fields

- `formatter (Dict | None)`: Defines the formatter and its configuration
- `runner_config (Dict | TaskRunnerConfig)`: Specifies parameters that control execution behavior, such as `save_formatted_output`, `seed_batch_size`, and `transform_batch_size`. Many of these can be overridden via command-line arguments.
- `datastore (Dict | None)`: Provides configuration for storing intermediate, final, and formatted data in a datastore.

#### Extra Fields

Databuilder developers can easily pass custom fields to a task by modifying its constructor. As mentioned earlier, all entries from the YAML configuration file are provided to the constructor as `kwargs`.

##### Example: Passing a Random Seed

Suppose we want to pass a `random_seed` value for selecting in-context learning (ICL) examples. First, add the field to your YAML configuration:

```{.yaml hl_lines="46"}
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: core/simple/logical_reasoning/causal # Must be unique. Recommend following directory structure as naming convention.
task_description: To teach a language model about Logical Reasoning - causal relationships
created_by: IBM
data_builder: simple # Must match exactly with the databuilder name

######################################################
#                   RESERVED FIELDS
######################################################
seed_examples:
  - answer:
      "While days tend to be longer in the summer, just because it is not summer
      doesn't mean days are necessarily shorter.

      "
    question:
      "If it is summer, then the days are longer. Are the days longer if it
      is not summer ?

      "
  - answer:
      'No, we cannot conclusively conclude that some cats are black based solely
      on the given premises. The statement "some mammals are black" does not necessarily
      guarantee that among those mammals are cats.

      '
    question:
      If all cats are mammals and some mammals are black, can we conclude that
      some cats are black?
  - answer:
      "Yes, we can conclude that all squares have four sides based on the given
      premises.

      "
    question:
      "If all squares are rectangles and a rectangle has four sides, can we
      conclude that all squares have four sides?

      "

######################################################
#                   TASK FIELDS
######################################################
random_seed: 42
```

Next, update the associated [`task.py`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/core/databuilders/simple/task.py) by adding a new argument to the `SimpleTask` constructor:

```{.python hl_lines="19 34-35"}
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
        random_seed: Optional[int] = None
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

        # Save 3: Save task sepcified fields for later use
        self._random_seed = random_seed

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
```

Now, the `_random_seed` attribute is available on the `SimpleTask` instance and can be used in any method within the task.
