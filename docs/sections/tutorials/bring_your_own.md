# Bring Your Own Pipeline

DiGiT supports the integration of fully customized pipelines leveraging the interface offered by the [`DataBuilder` class](../../fms-dgt/fms_dgt/base/databuilder.py).
The integration relies on the design decision of not imposing any limitation on the number of _generators_ and / or _validators_ considered as _blocks_ in the `DataBuilder` definition.

As a practical example for pipeline on-boarding let's consider the [`sqlinstruct` case](../../src/databuilders/generation/sqlinstruct/).

Since the pipeline is already available as a python package, we start by including it as an optional dependency to the [pyproject.toml](../../pyproject.toml):

```toml
[project.optional-dependencies]
...
# here we include the actual package dependencies as well as specific package requirements
sql = ["pydantic>=2.4.2", "pydantic-settings>=2.0.3", "pyyaml>=6.0.1", "sqlglot==23.11.2", "sqlinstruct @ git+ssh://git@github.ibm.com/flowpilot/sql-synthetic-data-generator@main"]
...
# we make sure to include it  also in the "all" group
all = [
    ...
    "fms_dgt[sql]",
    ...
]
```

Afterwards we can proceed by implementing a dedicated `DataBuilder`. In `sqlinstruct`'s case, as we are integrating the data generation pipeline, we place it in the dedicated directory:

```console
# from repo root
$ ls -1 src/databuilders/generation
...
sqlinstruct
```

To implement the `DataBuilder`, we require the following directory structure:

```console
ls -1 src/databuilders/generation/sqlinstruct
README.md
sqlinstruct.yaml
generate.py
```

The `README.md` acts as a manifest describing pipeline functionalities and schema for its data configuration and can be directly copied and updated from the [template](../../fms-dgt/templates/databuilder/README.md).
For the `sqlinstruct` case see the [dedicated one](../../src/databuilders/generation/sqlinstruct/README.md).

The `sqlinstruct.yaml` is the configuration file used to initialize the `DataBuilder`, and in the case considered, as we are relying entirely on custom _generators_ and _validators_ provided by the installed dependency we have an extremely simple configuration:

```yaml
name: sqlinstruct
# all generators and validators are configured via sqlinstruct library
blocks: []
metadata:
  version: 1.0
  # NOTE: this is used to provide additional configurations to the sqlinstruct pipeline that go beyond DiGiT-specific ones
  # sqlinstruct_pipeline_configuration: {}
```

The `generate.py` implements the actual logic (see `NOTE:` comments for explanation):

```python
# Standard
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Iterable, List, Set, Tuple
import time

# Third Party
# NOTE: here we have the import from the installed dependency
from sqlinstruct.models import SQLDataGenerationSchema
from sqlinstruct.pipeline import (
    SQLDataGenerationPipeline,
    SQLDataGenerationPipelineConfiguration,
)

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import SdgTask
# NOTE: here we rely on types that are natively available in fms_dgt, it is straightforward to implement new ones
from fms_dgt.databuilders.generation.nl2sql.task import SqlSdgData, SqlSdgTask
from fms_dgt.databuilders.generation.simple.task import InstructLabSdgData
from fms_dgt.utils import sdg_logger

# NOTE: registering the DataBuilder allows to usage in the framework
@register_data_builder("sqlinstruct")
class SQLInstructDataBuilder(DataBuilder):  # NOTE: inheriting from the base DataBuilder is fundamental
    """Class for running SQL-Instruct as a data builder."""

    # NOTE: it is important to explicitly define the right task type
    TASK_TYPE: SdgTask = SqlSdgTask

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        # NOTE: the data-builder access directly the data, for an example: data/code/sql/nl2sql/orders/qna.yaml
        instruction_data: List[SqlSdgData],
    ) -> List[InstructLabSdgData]:

        outputs: List[InstructLabSdgData] = []
        discarded: int = 0

        # NOTE: here we define our custom logic
        sdg_logger.info("Starting data generation...")
        for instruction_data_item in instruction_data:
            # We just need to add some "task_description" as it is redacted by data configuration loading.
            data_generation_schema_dict = asdict(instruction_data_item)
            data_generation_schema_dict["task_description"] = (
                instruction_data_item.task_description
            )
            data_generation_schema = SQLDataGenerationSchema(
                **data_generation_schema_dict
            )
            sdg_logger.info(
                f"Running generation pipeline with data configuration: {data_generation_schema.model_dump_json(indent=2)}"
            )
            # NOTE: we allow to customize the pipeline configuration via the DataBuilder metadata
            pipeline_configuration_dictionary = (
                self.config.metadata.get("sqlinstruct_pipeline_configuration", {})
                if self.config.metadata is not None
                else {}
            )
            pipeline_configuration = SQLDataGenerationPipelineConfiguration(
                **pipeline_configuration_dictionary
            )
            sdg_logger.info(
                f"Running generation pipeline with parameters: {pipeline_configuration.model_dump_json(indent=2)}"
            )
            pipeline = SQLDataGenerationPipeline(
                pipeline_configuration=pipeline_configuration
            )
            for instruction in pipeline.generate(
                data_generation_schema=data_generation_schema
            ):
                # NOTE: convert the generated instructions to a format compatible with fms_dgt.
                converted_instruction = InstructLabSdgData(
                    # NOTE: coming from the package configuration
                    task_name=instruction_data_item.taxonomy_path,
                    # NOTE: info coming from taxonomy
                    taxonomy_path=instruction_data_item.taxonomy_path,
                    task_description=instruction_data_item.task_description,
                    # NOTE: info coming from generated entries
                    instruction=instruction.user,
                    input="",
                    document=None,
                    output=instruction.assistant,
                )
                outputs.append(converted_instruction)
        sdg_logger.info("Data generation completed.")

        # NOTE: we return the outputs in a DiGiT-compatible format
        return outputs

    # NOTE: as the input data for sqlinstruct are simply configuring the pipeline run with information on db schema, optional ground-truth,
    # optional query-logs, we need to make sure properly handle this input/output mismatch.
    def call_with_task_list(self, request_idx: int, tasks: List[SdgTask]) -> Iterable:
        # this data builder outputs data in a different format than the input, so only the original seed data should be used
        _ = request_idx
        data_pool = [e for task in tasks for e in task.seed_data]
        return self(data_pool)
```

At this point we are ready to run the registered pipeline via DiGit by pointing it to the relevant data :rocket::

```bash
fms_dgt --data-paths data/code/sql/nl2sql/orders/qna.yaml
```

Note that the registered `DataBuilder` will be picked automatically from the data entry itself using the `data_builder` field:

```yaml
task_name: nl2sql_orders
created_by: IBM Research
task_description: Natural Language to SQL data generation task for a Postgres database of orders
data_builder: sqlinstruct
...
```
