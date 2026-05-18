# Dataloaders

A Dataloader is a stateful iterator over a datastore. Where a datastore handles persistence (reading and writing files), a dataloader handles iteration: yielding one record at a time, tracking position, and optionally looping indefinitely or resuming from a checkpoint.

You do not typically instantiate dataloaders directly. The framework creates them internally when a task starts iterating over its seed or input data. Understanding dataloaders is useful when you need to control iteration behavior, apply field remapping at load time, or resume a long-running run.

## How dataloaders are used

For generation tasks, DiGiT wraps the seed datastore in a dataloader that loops continuously. Each iteration of the generation loop draws the next batch of seed examples from the dataloader, ensuring seeds are recycled as many times as needed until the target record count is reached.

For transformation tasks, the input datastore is wrapped in a dataloader that does not loop. When the dataloader is exhausted, the task is complete.

## Default dataloader

The `default` dataloader (`type: default`) wraps a datastore and yields its records one at a time:

- Tracks position as `(iterator_index, row_index)` so iteration can be resumed after a restart.
- Supports optional field remapping via the `fields` parameter.
- Loops back to the start when `loop_over: true` (the default for seed data).

The `simple` dataloader (`type: simple`) wraps an in-memory list directly, using a single integer index as state. It is used internally for seed examples provided inline in the task YAML.

## Field remapping

The `fields` parameter lets you rename or select fields as records are loaded, without modifying the source file:

```yaml
seed_datastore:
  type: default
  data_path: ${DGT_DATA_DIR}/public/examples/qa/seeds.jsonl
  fields:
    question: instruction # rename "question" to "instruction"
    answer: output # rename "answer" to "output"
```

Wildcard syntax keeps all fields while renaming specific ones:

```yaml
fields:
  "*": "*" # keep all fields as-is
  question: instruction # also rename this one
```

Nested fields are accessed with dot notation:

```yaml
fields:
  "meta.language": language # extract nested field to top level
```

## Resuming from a checkpoint

Dataloaders expose `get_state()` and `set_state(state)` for checkpointing. The state is a small dictionary recording the current position in the iteration sequence:

```python
state = dataloader.get_state()
# {"_ITER_INDEX": 0, "_ROW_INDEX": 142}

dataloader.set_state(state)  # resume from row 142
```

DiGiT persists dataloader state automatically during a run. If a run is interrupted and restarted without `--restart`, seed iteration resumes from where it left off rather than starting over from the first example.

## Implementing a custom dataloader

All dataloaders inherit from [`Dataloader`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/base/dataloader.py) and must implement:

- `__next__()`: yield the next record, raise `StopIteration` when exhausted
- `get_state()`: return a serializable state object
- `set_state(state)`: restore position from a previously saved state

Register with `@register_dataloader("my_type")` and reference by type in the task YAML datastore configuration.
