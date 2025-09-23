# Adding difficulty rating to Geography QA pairs

**[Task Specification](#task-specification) | [Rater](#rater) | [Usage](#usage) | [Contributors](#contributors)**

Data builder used for rating generated geographical question-answer pairs for difficulty using large language model.

## Task specification

This data builder supports generation defining the following parameters:

### Input

- `question`: question pertinent to geographical data
- `answer`: answer for the corresponding question

### Output

- `question`: question pertinent to geographical data
- `answer`: answer for the corresponding question
- `ratings`: difficulty ratings [very easy/easy/medium/hard/very hard] and knowledge required to answer question

An example can be found [here](../../../../../tasks/public/examples/rate/task.yaml).

## Rater

Default configuration for generator used by the data builder is available [here](./rater.yaml).

### Rater

- `meta-llama/llama-3-3-70b-instruct` via `watsonx`.

## Usage

To try out the databuilder, run the following command:

```
python -m fms_dgt.public --task-paths ./tasks/public/examples/rate/task.yaml --restart-generation
```

This launches a data transformation job by passing examples to be transformed via `data` field in the YAMLs specified via `--task-paths` argument.

By default, the output (\*.jsonl) is generated in sub-directories under output/public/examples/qa_ratings. Here's a sample output:

```json
{
  "task_name": "public/examples/qa_ratings",
  "is_seed": false,
  "question": "What is the longest river in South America?",
  "answer": "Amazon River",
  "ratings": {
    "knowledge": "To solve this problem, the models need to know the longest river in South America.",
    "difficulty": "very easy"
  }
}
```

## Contributors

**Author and Maintainer**: Kshitij Fadnis
