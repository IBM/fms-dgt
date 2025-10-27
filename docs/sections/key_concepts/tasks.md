# Tasks

Data files are used to instantiate data generation tasks. An example of one can be found [here](https://github.ibm.com/DGT/fms-dgt/blob/main/tasks/core/logical_reasoning/causal/task.yaml) (see below for the relevant snippet).

```yaml
task_name: causal_logical_reasoning
created_by: IBM
data_builder: simple
seed_examples:
  - answer:
      "While days tend to be longer in the summer, just because it is not summer
      doesn't mean days are necessarily shorter."
    question:
      "If it is summer, then the days are longer. Are the days longer if it
      is not summer ?"
task_description: To teach a language model about Logical Reasoning - causal relationships
```

Our goal was to be as non-prescriptive as possible, allowing people to load their own data with their own fields without having to modify it to fit into the framework. As such, in the YAML, the only components that must **always** be specified are the `created_by`, `task_name`, `data_builder`, and `task_description` fields. Beyond those, there are no constraints to a data file, i.e., the designer of a task can include whatever they want here.

Internally, the data of a YAML file will populate [Task / Data objects](https://github.ibm.com/DGT/fms-dgt/blob/main/fms_dgt/base/task.py) (see below for relevant snippet)

```python
@dataclass
class ExampleSdgData(SdgData):
    """This class is intended to hold the seed / machine generated instruction data"""

    task_description: str
    instruction: str
    input: str
    output: str
    document: str

class ExampleSdgTask(SdgTask):
    """This class is intended to hold general task information"""

    DATA_TYPE = ExampleSdgData

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def instantiate_example(self, **kwargs: Any):
        return self.DATA_TYPE(
            task_name=self.name,
            task_description=self.task_description,
            instruction=kwargs.get("question"),
            input=kwargs.get("context", ""),
            output=kwargs.get("answer"),
            document=kwargs.get("document", None),
        )
```

You can see from the above that the task object allows you to define your own example instantiation code. This can be used to add in global data (e.g., anything outside of the `seed_examples` field), to each data instance without having to store it redundantly in the YAML data file (i.e., copies of the same field for every entry of `seed_examples`).

As stated before, each iteration of generation will have the data builders taking in dataclass instances (both the seed examples and the examples generated thus far).
