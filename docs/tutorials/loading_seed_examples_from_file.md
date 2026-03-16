# Loading Seed Examples from a File

In synthetic data generation, seed examples (also called in-context learning examples) are provided in the prompt given to the teacher model. DiGiT's `GenerationTask` base class supports two ways to supply them: inline in the task YAML, or from an external file in `.jsonl`, `.json`, or `.parquet` format.

This tutorial continues with the misconceptions databuilder built in [Building a Generation Databuilder](../tutorials/generate_data.md). The seed examples are currently defined inline in the task YAML at `tasks/public/examples/misconceptions/task.yaml`:

```{.yaml .no-copy title="tasks/public/examples/misconceptions/task.yaml" hl_lines="13-23"}
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: public/examples/misconceptions
task_description: Generate misconception-correction pairs for training a model to identify and correct misinformation.
created_by: IBM

data_builder: public/examples/misconceptions

######################################################
#                   RESERVED FIELDS
######################################################
seed_examples:
  - misconception: Lightning never strikes the same place twice.
    correction: Lightning frequently strikes the same place multiple times. Tall structures like the Empire State Building are struck dozens of times per year.
  - misconception: Humans only use 10 percent of their brains.
    correction: Brain imaging studies show that virtually all regions of the brain are active at some point, and most are active almost all the time.
  - misconception: Swallowed chewing gum stays in your stomach for seven years.
    correction: While gum base is not digestible, it passes through the digestive system and is excreted within a few days, just like other indigestible matter.
  - misconception: Goldfish have a memory span of only three seconds.
    correction: Research has shown that goldfish can remember things for months and can be trained to navigate mazes and recognize their owners.
  - misconception: The Great Wall of China is visible from space with the naked eye.
    correction: The Great Wall is too narrow to be seen from low Earth orbit without aid. Astronauts have confirmed this repeatedly.
```

Keeping seed examples in the task YAML works well for small sets. For larger collections, or when you want to share seeds across multiple tasks, an external file is easier to manage.

## Step 1: Create the seed file

Save the following as `data/public/examples/misconceptions/seed_examples.jsonl`:

```{.json title="data/public/examples/misconceptions/seed_examples.jsonl"}
{"misconception": "Lightning never strikes the same place twice.", "correction": "Lightning frequently strikes the same place multiple times. Tall structures like the Empire State Building are struck dozens of times per year."}
{"misconception": "Humans only use 10 percent of their brains.", "correction": "Brain imaging studies show that virtually all regions of the brain are active at some point, and most are active almost all the time."}
{"misconception": "Swallowed chewing gum stays in your stomach for seven years.", "correction": "While gum base is not digestible, it passes through the digestive system and is excreted within a few days, just like other indigestible matter."}
{"misconception": "Goldfish have a memory span of only three seconds.", "correction": "Research has shown that goldfish can remember things for months and can be trained to navigate mazes and recognize their owners."}
{"misconception": "The Great Wall of China is visible from space with the naked eye.", "correction": "The Great Wall is too narrow to be seen from low Earth orbit without aid. Astronauts have confirmed this repeatedly."}
```

Each line is a JSON object with the same keys (`misconception`, `correction`) that the task's `instantiate_input_example` method expects.

## Step 2: Update the task YAML

Replace the `seed_examples` block with a `seed_datastore` reference:

```{.yaml title="tasks/public/examples/misconceptions/task.yaml" hl_lines="13-15"}
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: public/examples/misconceptions
task_description: Generate misconception-correction pairs for training a model to identify and correct misinformation.
created_by: IBM

data_builder: public/examples/misconceptions

######################################################
#                   RESERVED FIELDS
######################################################
seed_datastore:
  type: default
  data_path: ${DGT_DATA_DIR}/public/examples/misconceptions/seed_examples.jsonl
```

`${DGT_DATA_DIR}` resolves to the `data/` directory at the root of the repository by default. You can override it by setting the environment variable.

## Step 3: Run it

The run command is unchanged:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/misconceptions/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

DiGiT loads the seed examples from the file at startup and uses them exactly as it would inline examples.

## Next steps

- To switch to a different LM engine, see [Changing the Language Model Engine](changing_lm_engine.md).
- To add a validator that filters low-quality outputs, see [Creating a Validator](creating_validator.md).
