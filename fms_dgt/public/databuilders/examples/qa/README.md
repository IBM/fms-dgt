# Generation of Geography QA pairs

**[Task Specification](#task-specification) | [Generators and post-processors](#generators-and-post-processors) | [Usage](#usage) | [Contributors](#contributors)**

Data builder used for generating question-answer pairs over geographical data.

## Task specification

This data builder supports generation defining the following parameters:

### Required

- `question`: question pertinent to geographical data
- `answer`: answer for the corresponding question

An example can be found [here](../../../../../tasks/public/examples/qa/task.yaml).

## Generators and post-processors

Default configuration for generator used by the data builder is available [here](./qa.yaml).

### Generators

- `mistral-small3.2` via `ollama`.

### Post-processors

- `rouge_scorer`

## Usage

To try out the databuilder, run the following command:

```
python -m fms_dgt.public --task-paths ./tasks/public/examples/qa/task.yaml --restart-generation --num-outputs-to-generate 15
```

This launches a data generation job by passing seed examples data using the YAML specified via `--task-paths` argument.

By default, the output (\*.jsonl) is generated in sub-directories under output/public/examples/geography_qa. Here's a sample output:

```json
{
  "task_name": "public/examples/geography_qa",
  "is_seed": false,
  "question": "What is the largest desert in the world?",
  "answer": "The Antarctic Desert"
}
```

## Contributors

**Author and Maintainer**: Kshitij Fadnis
