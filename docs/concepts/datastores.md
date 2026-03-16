# Datastores

A Datastore is DiGiT's storage abstraction. It knows how to read data from a file or in-memory source, write data in a specified format, and report whether previously generated data already exists (for resuming interrupted runs).

Tasks create and manage their own datastores. You do not instantiate datastores directly in most cases. Instead, you configure them through the task YAML and the framework wires them up at runtime.

## What gets stored

DiGiT creates the following stores for each task under `<output_dir>/<task_name>/`:

| Store             | File                          | Contains                                                                  |
| ----------------- | ----------------------------- | ------------------------------------------------------------------------- |
| `data`            | `data.jsonl`                  | Records generated during the main loop, appended each iteration           |
| `final_data`      | `final_data.jsonl`            | Post-processed data; identical to `data` if no postprocessors are defined |
| `postproc_data_N` | `postproc_data_1.jsonl`, etc. | Data between postprocessing epochs                                        |
| `formatted_data`  | `formatted_data.jsonl`        | Output after a formatter block runs                                       |
| `task_card`       | `task_card/task_card.jsonl`   | Run configuration and metadata                                            |
| `task_results`    | `task_results.jsonl`          | Execution metadata: start time, record counts, custom metrics             |
| `logs`            | `logs.jsonl`                  | Structured log records for this task                                      |
| Block stores      | `blocks/<block_name>/*.jsonl` | Records rejected by validator blocks                                      |

The output directory defaults to `output/` at the repository root. Set `DGT_OUTPUT_DIR` to change it, or override per task via `runner_config.output_dir` in the task YAML.

## Configuring a datastore

The default datastore type handles the common cases. Override it in the task YAML under `datastore`:

```yaml
datastore:
  type: default
  output_data_format: parquet # write final data as Parquet instead of JSONL
```

To configure where seed examples are loaded from, use `seed_datastore`:

```yaml
seed_datastore:
  type: default
  data_path: ${DGT_DATA_DIR}/public/examples/qa/seed_examples.jsonl
```

`${DGT_DATA_DIR}` resolves to the `data/` directory at the repository root by default. You can override it by setting the environment variable.

For transformation tasks, the input dataset is configured under `data`:

```yaml
data:
  type: default
  data_path: ${DGT_DATA_DIR}/public/examples/fact_triples/paragraphs.jsonl
```

## Default datastore

The `default` datastore (`type: default`) supports reading from and writing to files in the following formats:

**Readable formats**: `.jsonl`, `.json`, `.yaml`, `.parquet`, `.csv`, `.txt`, `.md`, and HuggingFace dataset directories.

**Writable formats**: `jsonl` (default), `parquet`.

Large files (JSONL, Parquet, CSV) are loaded lazily to avoid reading everything into memory at once.

Key configuration fields:

| Field                | Default | Description                                                            |
| -------------------- | ------- | ---------------------------------------------------------------------- |
| `data_path`          | None    | File path or glob pattern to read from. Supports `${VAR}` expansion.   |
| `output_data_format` | `jsonl` | Format for writing output: `jsonl` or `parquet`.                       |
| `data_formats`       | None    | Filter loaded files by extension (e.g., `[".jsonl", ".json"]`).        |
| `data_split`         | `train` | HuggingFace dataset split name, when loading from a dataset directory. |

Glob patterns are supported in `data_path` (for example, `data/chunks/**.jsonl` loads all matching files). Multiple files are concatenated in the order they are found.

## Multi-target datastore

The `multi_target` datastore writes to multiple backends simultaneously. The primary store is the read source; secondary stores receive writes on a best-effort basis (failures are logged but do not stop the run).

```yaml
formatted_datastore:
  type: multi_target
  primary:
    type: default
    output_data_format: jsonl
  additional:
    - type: default
      output_dir: /backup/outputs
      output_data_format: parquet
```

This is useful when you want to archive output to a secondary location (remote storage, a different format) without making it a hard dependency of the run.

## Implementing a custom datastore

All datastores inherit from [`Datastore`](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/base/datastore.py) and must implement:

- `exists()`: return `True` if previously persisted data is present
- `clear()`: delete all stored data
- `save_data(data)`: write records to the backing store
- `load_iterators()`: return a list of iterators over stored records
- `load_data()`: return all stored records
- `close()`: release any open handles

Register with `@register_datastore("my_type")` and reference by `type: my_type` in YAML.
