# Logical and Temporal Reasoning

This example shows how to generate reasoning data without writing any Python. The `SimpleDataBuilder` is a ready-to-use generation databuilder that accepts a task YAML with seed examples and produces new examples in the same style using in-context learning.

The two tasks covered here, causal logical reasoning and temporal ordering, ship with the repository and run out of the box with Ollama.

## What is SimpleDataBuilder?

`SimpleDataBuilder` is a general-purpose generation databuilder that implements the self-instruct pipeline. Given a set of seed question-answer examples, it prompts a language model to produce new examples in the same format, then filters near-duplicates using Rouge-L scoring. It requires no custom code: the task YAML is the entire configuration.

The databuilder is registered as `public/instructlab/simple`.

## Logical reasoning: causal relationships

This task teaches a model to reason about cause and effect. Given conditional statements, can the model determine what follows and what does not?

### Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/instructlab/simple/logical_reasoning/causal/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

### Task specification

The task file is at `tasks/public/instructlab/simple/logical_reasoning/causal/task.yaml`. Each seed example has a `question` (the reasoning prompt) and an `answer` (the correct inference):

```yaml
task_name: public/instructlab/simple/logical_reasoning/causal
task_description: To teach a language model about Logical Reasoning - causal relationships
created_by: IBM
data_builder: public/instructlab/simple

seed_examples:
  - question: |
      If it is summer, then the days are longer. Are the days longer if it is not summer?
    answer: |
      While days tend to be longer in the summer, just because it is not summer
      doesn't mean days are necessarily shorter.
  - question: |
      If all cats are mammals and some mammals are black, can we conclude that some cats are black?
    answer: |
      No, we cannot conclusively conclude that some cats are black based solely
      on the given premises. The statement "some mammals are black" does not necessarily
      guarantee that among those mammals are cats.
  - question: |
      If all squares are rectangles and a rectangle has four sides, can we
      conclude that all squares have four sides?
    answer: |
      Yes, we can conclude that all squares have four sides based on the given premises.
```

### Sample output

Generated data lands in `output/public/instructlab/simple/logical_reasoning/causal/final_data.jsonl`. Each record looks like:

```json
{
  "task_name": "public/instructlab/simple/logical_reasoning/causal",
  "is_seed": false,
  "instruction": "If all birds can fly and penguins are birds, can we conclude that penguins can fly?",
  "input": "",
  "output": "No. While the premise states all birds can fly, this is factually incorrect. Penguins are birds but they cannot fly. The logical form of the argument is valid, but the conclusion is false because the first premise is false."
}
```

## Temporal reasoning: ordering events in time

This task teaches a model to establish chronological order from a set of relative constraints, a core skill for tasks involving scheduling, narrative understanding, and planning.

### Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/instructlab/simple/temporal_reasoning/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

### Task specification

The task file is at `tasks/public/instructlab/simple/temporal_reasoning/task.yaml`:

```yaml
task_name: public/instructlab/simple/temporal_reasoning
task_description: To teach a language model about chronological order of time.
created_by: IBM
data_builder: public/instructlab/simple

seed_examples:
  - question: |
      Consider I went to a party. I reached before John. David arrived after.
      Joe reached before me. John arrived after David. Who reached first and who reached last?
    answer: |
      1. Joe reached first (before you).
      2. You arrived next.
      3. David arrived after you.
      4. John reached last (after David).

      Joe was the first one to arrive at the party, followed by you, then David, and finally John.
```

### Sample output

```json
{
  "task_name": "public/instructlab/simple/temporal_reasoning",
  "is_seed": false,
  "instruction": "Alice finished her report before Bob started his. Carol submitted hers after Bob but before Dave. In what order did they finish?",
  "input": "",
  "output": "The order is: Alice, Bob, Carol, Dave. Alice finished first, then Bob started and finished, then Carol submitted, and Dave was last."
}
```

## Running both tasks together

You can pass multiple task paths in a single run:

```bash
python -m fms_dgt.public \
  --task-paths \
    ./tasks/public/instructlab/simple/logical_reasoning/causal/task.yaml \
    ./tasks/public/instructlab/simple/temporal_reasoning/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

DiGiT runs both tasks concurrently and writes each to its own output directory.

## Next steps

- To use a different model or provider, see [Changing the LM Engine](../tutorials/changing_lm_engine.md).
- To understand how `SimpleDataBuilder` works under the hood, read the [source](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/instructlab/simple).
- To build a custom generation databuilder from scratch, see [Data Generation](../tutorials/generate_data.md).
