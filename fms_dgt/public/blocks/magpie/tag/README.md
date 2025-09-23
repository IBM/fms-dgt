# Magpie Tagging transformation

Data builder used for tagging data using Magpie .
Modified version of [**Magpie**](https://magpie-align.github.io/) to enable working with opensource models.

It generates scores and tags using the open-source Mixtral-8x7B model as the teacher (generator) and prompt templates.

## Data specification

This data builder supports generation defining the following parameters:

### Required

```
- `task_name` : magpie_tag_transform
- `created_by`: creator of the task.
- `task_description`: description of the task.
- `data_builder`: magpie_tag_transform
- `seed_datastore`:
  `type`: default
  `data_path`: path to file
```

### Format of Data

see example data format [here](../../../../../../data/granite/magpie_tranform/data.jsonl)

The data should have "input", "output" field or "messages" field which is a list of dictionaries with alternating

```
[{'role': 'user', 'content':'something'}, {'role':'assistant', 'content':'something'}]
```

If there is a "messages" field then it will ignore the "input" and "output" field and tag

### Explanation

Tagging the input & output (in case of single turn) or the conversation (in case of multi turn) in terms of :

```
quality (question) : [
"very poor",
"poor",
"average",
"good",
"excellent",
]

sample_quality score(question and response) : ["1", "2", "3", "4", "5"]

difficulty : [
"very easy",
"easy",
"medium",
"hard",
"very hard",
]

classification of task: []

```

This is Step 1 in [Magpie Dataset Filtering](https://github.com/magpie-align/magpie?tab=readme-ov-file#dataset-filtering)

To perform all the tagging use

```
mission: all
```

To perform only the tagging used by magpie (quality, sample quality and difficulty) use

```
mission: magpie
```

### Script Help

Make sure to install dependencies by following the [README](../../../../../../README.md) of the fms-dgt repo and then do

```
pip install ".[magpie]"
```

An example script is

```python
python -m fms_dgt.granite --task-paths ./tasks/granite/magpie/task.yaml --restart-generation
```

## Generators and validators

Default configuration for generators used is available [here](./magpie_tag_transform.yaml).

Another method of using models to generate is vllm.

In the above config you can change the `type: vllm` and model id to any hfmodel. Make sure to run on a node with enough GPUs to spin up that model and perform `huggingface-cli login` in case it is a restricted model.

If you have a model hosted on a node using vllm you can use ,

```
type: vllm-remote
base_url: <url of the machine it is hosted in>
```
